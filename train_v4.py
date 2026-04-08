"""
SmolDocling fine-tuning — v4

Key improvements over train.py:
  - QLoRA  : 4-bit base model via bitsandbytes (--qlora)
  - DoRA   : Weight-Decomposed LoRA (better than plain LoRA at same rank)
  - Targets: attention + all MLP projections (gate/up/down)
  - Flash Attention 2 (--flash_attn, requires flash-attn + bfloat16)
  - Frozen vision encoder (--freeze_vision, decoder-only gradient flow)
  - SFTTrainer from TRL (cleaner instruction-tuning loop)
  - DocTag-specific eval: tag_acc, loc_mae, tree_edit_dist

Recommendations
---------------
  lora_r       : 32 is good for DoRA; lower (16) if VRAM is tight
  lora_alpha   : keep equal to lora_r (scaling = 1.0) for stable training
  batch_size   : 1–2 for 8–16 GB GPU; 8–12 for 40–80 GB GPU
  grad_accum   : effective batch = batch_size × grad_accum × num_gpus
                 aim for effective batch ≥ 16
  eval_steps   : set to ~10 % of steps-per-epoch so you get ~10 evals
  report_to    : swap "none" for "wandb" or "tensorboard" to track runs

Usage:
    python train_v4.py --train_data training_data_train.jsonl \\
                       --val_data   training_data_val.jsonl   \\
                       --qlora --freeze_vision --flash_attn   \\
                       --output_dir doctag_model_v4
"""

import argparse
import json
import os
import re
import signal
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey-patch json so torch.dtype objects (e.g. torch.bfloat16) serialise
# without crashing – needed by older transformers config logging.
import json
_json_default = json.JSONEncoder().default

def _patched_default(self, o):
    try:
        import torch as _torch
        if isinstance(o, _torch.dtype):
            return str(o)
    except Exception:
        pass
    return _json_default(o)

json.JSONEncoder.default = _patched_default

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"
SYSTEM_PROMPT = "Convert this document page to DocTags format."

# DocTag structural tags for eval metrics
_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
    # OTSL table cell tokens – required to measure table learning in tag_acc / tree_edit_dist
    "ched", "fcel", "rhed", "ucel", "lcel", "ecel", "nl",
})


# ---------------------------------------------------------------------------
# DocTag metric helpers
# ---------------------------------------------------------------------------

def _extract_structural_tags(s: str) -> list[str]:
    return [
        ("/" if slash else "") + tag
        for slash, tag in re.findall(r'<(/?)(\w+(?:_\w+)*)(?:[^>]*)?>', s)
        if tag in _STRUCTURAL_TAGS
    ]


def _tag_accuracy(pred: str, ref: str) -> float:
    pred_tags = _extract_structural_tags(pred)
    ref_tags = _extract_structural_tags(ref)
    if not ref_tags and not pred_tags:
        return 1.0
    if not ref_tags or not pred_tags:
        return 0.0
    matches = sum(p == r for p, r in zip(pred_tags, ref_tags))
    return matches / max(len(pred_tags), len(ref_tags))


def _loc_mae(pred: str, ref: str) -> float:
    """MAE on <loc_N> values (0–500 scale). Missing pred tokens counted as 0."""
    pred_locs = [int(n) for n in re.findall(r'<loc_(\d+)>', pred)]
    ref_locs = [int(n) for n in re.findall(r'<loc_(\d+)>', ref)]
    if not ref_locs:
        return 0.0
    total = sum(
        abs(pred_locs[i] - ref_locs[i]) if i < len(pred_locs) else ref_locs[i]
        for i in range(len(ref_locs))
    )
    return total / len(ref_locs)


def _tree_edit_dist(pred: str, ref: str) -> float:
    """Normalised Zhang-Shasha tree edit distance. Falls back to tag-sequence Levenshtein."""
    try:
        import zss

        def _build(s):
            class _N:
                def __init__(self, label):
                    self.label = label
                    self.children: list = []
            root = _N("root")
            stack = [root]
            for slash, tag in re.findall(r'<(/?)(\w+(?:_\w+)*)(?:[^>]*)?>', s):
                if tag not in _STRUCTURAL_TAGS:
                    continue
                if not slash:
                    node = _N(tag)
                    stack[-1].children.append(node)
                    stack.append(node)
                elif len(stack) > 1:
                    stack.pop()
            return root

        def _to_zss(n):
            z = zss.Node(n.label)
            for c in n.children:
                z.addkid(_to_zss(c))
            return z

        def _size(n):
            return 1 + sum(_size(c) for c in n.children)

        pt = _to_zss(_build(pred))
        rt = _to_zss(_build(ref))
        dist = zss.simple_distance(pt, rt)
        max_dist = _size(pt) + _size(rt)
        return dist / max_dist if max_dist > 0 else 0.0

    except ImportError:
        pred_tags = _extract_structural_tags(pred)
        ref_tags = _extract_structural_tags(ref)
        if not pred_tags and not ref_tags:
            return 0.0
        n, m = len(pred_tags), len(ref_tags)
        if n == 0 or m == 0:
            return 1.0
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if pred_tags[i - 1] == ref_tags[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[m] / max(n, m)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DocTagDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_length: int = 4096,
        max_image_edge: int = 1024,
    ):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length
        self.max_image_edge = max_image_edge

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        if self.max_image_edge > 0 and max(image.size) > self.max_image_edge:
            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail((self.max_image_edge, self.max_image_edge), resampling.LANCZOS)
        doctag = record["doctag"]
        # Normalize <image>/<image_caption> tags to <picture> so the processor
        # doesn't count them as additional image placeholders.
        doctag = re.sub(r"<image>", "<picture>", doctag)
        doctag = re.sub(r"</image>", "</picture>", doctag)

        messages_full = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": doctag}]},
        ]
        text_full = self.processor.apply_chat_template(messages_full, add_generation_prompt=False)

        messages_prompt = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}]},
        ]
        text_prompt = self.processor.apply_chat_template(messages_prompt, add_generation_prompt=True)

        prompt_ids = self.processor.tokenizer(
            text_prompt, add_special_tokens=False, truncation=True, max_length=self.max_length,
        ).input_ids

        inputs = self.processor(
            images=image, text=text_full, return_tensors="pt",
            truncation=True, max_length=self.max_length,
            size={"longest_edge": min(max(image.size), self.max_image_edge)} if self.max_image_edge > 0 else None,
        )
        image.close()

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        labels[:len(prompt_ids)] = -100
        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def doctag_data_collator(features: list[dict]) -> dict:
    from torch.nn.utils.rnn import pad_sequence

    batch: dict = {}
    for key in features[0].keys():
        tensors = [f[key] for f in features]
        if key == "labels":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
        elif key == "attention_mask":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        elif key in ("pixel_values", "pixel_attention_mask"):
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                max_p = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_p - t.shape[0]
                    if pad_len > 0:
                        t = torch.cat([t, torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        else:
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
    return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def preprocess_logits_for_metrics(logits, labels):  # noqa: ARG001
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1).cpu()  # move to CPU immediately to avoid GPU memory accumulation


def make_compute_metrics(tokenizer):
    def _compute(eval_preds):
        import numpy as np
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.array(predictions, dtype=np.int64)
        mask = labels != -100
        token_acc = float((pred_ids == labels)[mask].mean()) if mask.any() else 0.0

        pad_id = tokenizer.pad_token_id or 0
        vocab_size = tokenizer.vocab_size or len(tokenizer)
        labels_dec = np.clip(np.where(labels == -100, pad_id, labels), 0, vocab_size - 1)
        preds_dec = np.clip(pred_ids, 0, vocab_size - 1)
        pred_strs = tokenizer.batch_decode(preds_dec.tolist(), skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(labels_dec.tolist(), skip_special_tokens=True)

        tag_acc = float(np.mean([_tag_accuracy(p, r) for p, r in zip(pred_strs, ref_strs)]))
        loc_mae = float(np.mean([_loc_mae(p, r) for p, r in zip(pred_strs, ref_strs)]))
        tree_edit = float(np.mean([_tree_edit_dist(p, r) for p, r in zip(pred_strs, ref_strs)]))

        return {
            "token_acc": token_acc,
            "tag_acc": tag_acc,
            "loc_mae": loc_mae,
            "tree_edit_dist": tree_edit,
        }
    return _compute


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------

def _install_signal_handlers(trainer) -> None:
    def _handler(signum, frame):
        print(f"\n[Signal {signal.Signals(signum).name}] Saving emergency checkpoint …")
        try:
            save_dir = os.path.join(trainer.args.output_dir, "emergency_checkpoint")
            trainer.save_model(save_dir)
            print(f"Saved to {save_dir}")
        except Exception as exc:
            print(f"Could not save: {exc}")
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SmolDocling with SFTTrainer + QLoRA/DoRA")
    parser.add_argument("--train_data", default="training_data_train.jsonl")
    parser.add_argument("--val_data", default="training_data_val.jsonl")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", default="doctag_model_v4")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size. Aim for effective batch ≥ 16 via grad_accum.")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--max_image_edge",
        type=int,
        default=1024,
        help="Cap images to this longest edge before processing; also prevents processor upscaling small images.",
    )
    # LoRA / DoRA
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank. 32 works well for DoRA; lower (16) if VRAM is tight.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha. Recommended: equal to lora_r (scaling = 1.0).")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_dora", action="store_true",
                        help="Disable DoRA and use plain LoRA (faster, slightly lower quality).")
    parser.add_argument("--adapter", type=str, default="",
                        help="Path to an existing PEFT adapter directory to resume from "
                             "(e.g. a previous best_model/ or emergency_checkpoint/). "
                             "LoRA/DoRA weights are loaded and training continues from there.")
    # Hardware
    parser.add_argument("--qlora", action="store_true",
                        help="Load base model in 4-bit NF4 for QLoRA (~60 %% less VRAM).")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Flash Attention 2 (requires: pip install flash-attn --no-build-isolation).")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Freeze vision encoder; only the language decoder trains.")
    # Eval / save
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Run eval every N optimizer steps. ~10%% of steps-per-epoch is a good target.")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    # W&B
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="Docling finetune train_v4", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity (team or username).")
    parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (auto-generated if empty).")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"], help="W&B mode.")
    args = parser.parse_args()

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    # --- W&B init ---
    if args.use_wandb:
        import importlib
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        import wandb
        import datetime
        run_name = args.wandb_run_name.strip() or (
            f"doctag-{Path(args.train_data).stem}-"
            f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                mode=args.wandb_mode,
                tags=tags or None,
            )
        except Exception as _wandb_err:
            print(
                f"[WARNING] wandb online init failed ({_wandb_err}). "
                "Falling back to offline mode. Run `wandb sync` later to upload."
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                mode="offline",
                tags=tags or None,
            )

    print(f"Loading processor and model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # --- QLoRA: 4-bit base model ---
    bnb_config = None
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",       # NF4 quantization (better than int4 for LLM weights)
            bnb_4bit_use_double_quant=True,  # quantize the quantization constants too
        )
        print("QLoRA: loading model in 4-bit NF4")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    # Prepare quantized model for gradient-based training
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("QLoRA: model prepared for kbit training")

    # --- Freeze vision encoder ---
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vision" in name or "image_model" in name:
                param.requires_grad = False
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Vision encoder frozen: {frozen:,} params  |  Trainable: {trainable:,} params")

    # --- DoRA / LoRA config: attention + all MLP projections ---
    from peft import LoraConfig, PeftModel, get_peft_model
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading existing adapter from {args.adapter} …")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print("Adapter loaded – continuing training from existing weights.")
    else:
        if args.adapter:
            print(f"[WARNING] --adapter path '{args.adapter}' not found; starting fresh.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,  # equal to r → scaling = 1.0 (recommended)
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            use_dora=not args.no_dora,       # DoRA: decomposes weight into magnitude + direction
            init_lora_weights="gaussian",    # gaussian init often outperforms default lora init
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Datasets ---
    print("Building datasets …")
    train_dataset = DocTagDataset(
        args.train_data,
        processor,
        args.max_length,
        args.max_image_edge,
    )
    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(
            args.val_data,
            processor,
            args.max_length,
            args.max_image_edge,
        )
        print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples  |  No val file at {args.val_data}")

    # --- SFTConfig ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,           # ramp LR over first 5 % of steps (more robust than fixed warmup_steps)
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=args.eval_steps if val_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",   # fused AdamW is faster and uses less memory than default
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="wandb" if args.use_wandb else "none",
        eval_accumulation_steps=1,   # move eval predictions to CPU after each step
        # SFT-specific: skip TRL's internal tokenization since DocTagDataset handles it
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
        compute_metrics=make_compute_metrics(processor.tokenizer) if val_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset is not None else None,
    )

    _install_signal_handlers(trainer)

    print("Starting training …")
    try:
        trainer.train()
    except Exception as exc:
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        print(f"\n[ERROR] Training crashed: {exc}")
        print(f"Saving emergency checkpoint to {emergency_dir} …")
        try:
            trainer.save_model(emergency_dir)
            processor.save_pretrained(emergency_dir)
            print(f"Emergency checkpoint saved to {emergency_dir}")
        except Exception as save_exc:
            print(f"[WARNING] Could not save emergency checkpoint: {save_exc}")
        raise

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
