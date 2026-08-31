"""
Fine-tune ibm-granite/granite-docling-258M on Mongolian document pages
to output DocTag format.

The script loads (image, doctag) pairs from JSONL files produced by
prepare_dataset.py, fine-tunes the vision-language model using the
HuggingFace Trainer API, and saves the result to --output_dir.

LoRA (parameter-efficient fine-tuning) is strongly recommended to reduce
VRAM requirements. Pass --lora to enable it (requires peft).

Usage:
    # 1. Build the training dataset first
    python prepare_dataset.py --xml_dir dataset/xml_annotations \\
                               --images_dir dataset/uploads

    # 2. Train
    python train.py --train_data training_data_train.jsonl \\
                    --val_data   training_data_val.jsonl   \\
                    --lora                                 \\
                    --bf16                                 \\
                    --output_dir doctag_model
"""

import argparse
import json
import os
from pathlib import Path
from signal import signal

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
import sys
import re
import signal
import math

# Default model to fine-tune
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"

SYSTEM_PROMPT = "Convert this document page to DocTags format."

_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
})

def _extract_structural_tags(s: str) -> list[str]:
    """Return ordered list of structural open/close tags from a DocTag string."""
    return [
        ("/" if slash else "") + tag
        for slash, tag in re.findall(r'<(/?)(\w+(?:_\w+)*)(?:[^>]*)?>', s)
        if tag in _STRUCTURAL_TAGS
    ]


def _tag_accuracy(pred: str, ref: str) -> float:
    """Positional accuracy of structural tags (open + close) between pred and ref."""
    pred_tags = _extract_structural_tags(pred)
    ref_tags = _extract_structural_tags(ref)
    if not ref_tags and not pred_tags:
        return 1.0
    if not ref_tags or not pred_tags:
        return 0.0
    matches = sum(p == r for p, r in zip(pred_tags, ref_tags))
    return matches / max(len(pred_tags), len(ref_tags))


def _loc_mae(pred: str, ref: str) -> float:
    """Mean absolute error on <loc_N> values (scale 0–500).
    Missing predicted tokens are treated as predicting 0."""
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
    """Normalised tree edit distance on the structural DocTag tree (0 = identical, 1 = worst).
    Uses the zss library when available; falls back to tag-sequence Levenshtein."""
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
            return 1 + sum(_size(c) for c in n.get_children())

        pt = _to_zss(_build(pred))
        rt = _to_zss(_build(ref))
        dist = zss.simple_distance(pt, rt)
        max_dist = _size(pt) + _size(rt)
        return dist / max_dist if max_dist > 0 else 0.0

    except ImportError:
        # Fallback: tag-sequence normalised edit distance
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
    
class DocTagDataset(Dataset):
    """
    Loads (image_path, doctag_string) pairs from a JSONL file and returns
    tokenised inputs ready for the Trainer.

    Label masking strategy
    ----------------------
    The full conversation (prompt + assistant response) is tokenised once.
    Tokens that belong to the *prompt* part are masked to -100 so the loss
    is computed only on the assistant's DocTag output.
    """

    def __init__(self, jsonl_path: str, processor, max_length: int = 1024):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        doctag = record["doctag"]
        # Normalise <image>...</image> structural tags to <picture>...</picture>
        # so the processor does not mistake them for image placeholders.
        doctag = doctag.replace("<image>", "<picture>").replace("</image>", "</picture>")
        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": doctag}],
            },
        ]
        text_full = self.processor.apply_chat_template(
            messages_full, add_generation_prompt=False
        )

        # --- build prompt-only conversation (to find where response starts) ---
        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, add_generation_prompt=True
        )

        # Tokenise both sequences (without images for the prompt-only pass)
        prompt_ids = self.processor.tokenizer(
            text_prompt,
            add_special_tokens=False,
            truncation=False,
            max_length=self.max_length,
        ).input_ids

        # Full pass: include the image so pixel_values are produced correctly.
        # Do NOT pass truncation to the processor: truncating inside the processor
        # can remove image placeholder tokens from the text while leaving the
        # pixel_values unchanged, causing a token-count mismatch error.
        # Instead we truncate the 1-D token tensors manually after processing.
        try:
            inputs = self.processor(
                images=image,
                text=text_full,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        except ValueError as exc:
            if "number of images" not in str(exc):
                raise
            # Fallback: repeat the single page image to match the placeholder count.
            image_token_count = text_full.count("<image>")
            if image_token_count <= 1:
                raise
            inputs = self.processor(
                images=[image] * image_token_count,
                text=text_full,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        finally:
            image.close()

        # Remove the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Truncate 1-D token tensors to max_length (pixel_values are left intact)
        # for _key in ("input_ids", "attention_mask"):
        #     if _key in inputs and inputs[_key].dim() == 1 and inputs[_key].shape[0] > self.max_length:
        #         inputs[_key] = inputs[_key][: self.max_length]

        # Build labels: mask all prompt tokens with -100
        labels = inputs["input_ids"].clone()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100

        inputs["labels"] = labels
        return inputs

def make_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function for the Trainer that computes:
    - token_acc      : fraction of non-masked tokens predicted correctly (teacher-forced)
    - tag_acc        : positional accuracy of structural DocTag element tags
    - loc_mae        : mean absolute error on <loc_N> location tokens (scale 0–500)
    - tree_edit_dist : normalised tree edit distance on the structural DocTag tree
    """
    def _compute(eval_preds):
        import numpy as np

        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.array(predictions, dtype=np.int64)

        # -- token accuracy --
        mask = labels != -100
        token_acc = float((pred_ids == labels)[mask].mean()) if mask.any() else 0.0

        # -- decode to strings --
        pad_id = tokenizer.pad_token_id or 0
        vocab_size = tokenizer.vocab_size or len(tokenizer)
        labels_for_decode = np.where(labels == -100, pad_id, labels)
        labels_for_decode = np.clip(labels_for_decode, 0, vocab_size - 1)
        pred_ids_for_decode = np.clip(pred_ids, 0, vocab_size - 1)
        pred_strs = tokenizer.batch_decode(pred_ids_for_decode.tolist(), skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(labels_for_decode.tolist(), skip_special_tokens=True)

        # -- DocTag-specific metrics --
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


def doctag_data_collator(features: list[dict]) -> dict:
    """
    Pad a batch of samples produced by DocTagDataset to the same length.
    pixel_values may have an extra patches dimension; handle both cases.
    """
    from torch.nn.utils.rnn import pad_sequence

    batch: dict = {}
    for key in features[0].keys():
        tensors = [f[key] for f in features]
        if key == "labels":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
        elif key == "attention_mask":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        elif key == "pixel_values":
            # SmolVLM pixel_values shape: [C, H, W] or [num_patches, C, patch_H, patch_W]
            # torch.stack works for both as long as all items share non-batch dims
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                # Variable patch counts: fall back to padding along dim 0
                max_patches = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_patches - t.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        elif key == "pixel_attention_mask":
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                max_patches = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_patches - t.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        else:
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)

    return batch

def _plot_training_curves(history: dict, output_dir: str) -> None:
    """Save a 2×2 grid of training-curve plots to *output_dir*."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipping training curve plot.  "
            "Install it with:  pip install matplotlib"
        )
        return

    has_train = bool(history.get("train_steps") and history.get("train_loss"))
    has_eval = bool(history.get("eval_epochs") and history.get("eval_loss"))
    has_grad = bool(history.get("train_grad_norm"))
    has_token_acc = bool(history.get("eval_token_acc"))
    has_tag_acc = bool(history.get("eval_tag_acc"))
    has_loc_mae = bool(history.get("eval_loc_mae"))
    has_tree_edit = bool(history.get("eval_tree_edit_dist"))

    if not has_train and not has_eval:
        return

    # Build subplot panels dynamically
    panels = []  # list of (title, x_data, y_data, color, xlabel, ylabel)
    if has_train:
        panels.append(("Train Loss", history["train_steps"], history["train_loss"],
                        "steelblue", "Step", "Loss"))
        panels.append(("Train Perplexity", history["train_steps"], history["train_perplexity"],
                        "steelblue", "Step", "Perplexity"))
    if has_grad:
        gsteps = history["train_steps"][:len(history["train_grad_norm"])]
        panels.append(("Gradient Norm", gsteps, history["train_grad_norm"],
                        "mediumseagreen", "Step", "Grad Norm"))
    if has_eval:
        panels.append(("Val Loss", history["eval_epochs"], history["eval_loss"],
                        "darkorange", "Epoch", "Loss"))
        panels.append(("Val Perplexity", history["eval_epochs"], history["eval_perplexity"],
                        "darkorange", "Epoch", "Perplexity"))
    if has_token_acc:
        panels.append(("Val Token Accuracy", history["eval_epochs"], history["eval_token_acc"],
                        "mediumvioletred", "Epoch", "Token Acc"))
    if has_tag_acc:
        panels.append(("Val Tag Accuracy", history["eval_epochs"], history["eval_tag_acc"],
                        "darkgreen", "Epoch", "Tag Acc"))
    if has_loc_mae:
        panels.append(("Val Loc MAE (↓ better)", history["eval_epochs"], history["eval_loc_mae"],
                        "crimson", "Epoch", "MAE (loc units)"))
    if has_tree_edit:
        panels.append(("Val Tree Edit Dist (↓ better)", history["eval_epochs"], history["eval_tree_edit_dist"],
                        "saddlebrown", "Epoch", "Norm. Tree Edit"))

    ncols = 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle("Training Progress", fontsize=14, y=1.01)

    for i, (title, xs, ys, color, xlabel, ylabel) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        marker = "o" if xlabel == "Epoch" else None
        lw = 2.0 if xlabel == "Epoch" else 1.2
        ax.plot(xs, ys, color=color, linewidth=lw, marker=marker, markersize=5, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_curves.png")
    try:
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves saved to {plot_path}")
    except Exception as exc:
        print(f"Could not save training curves: {exc}")


# ---------------------------------------------------------------------------
# Signal / interrupt handling
# ---------------------------------------------------------------------------

def _install_signal_handlers(trainer: Trainer) -> None:
    """Install SIGINT and SIGTERM handlers that save an emergency checkpoint.

    The *trainer* instance is captured in a closure so no global mutable state
    is needed.
    """

    def _handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        print(f"\n[Signal {sig_name}] Interrupted.  Saving emergency checkpoint …")
        try:
            save_dir = os.path.join(trainer.args.output_dir, "emergency_checkpoint")
            trainer.save_model(save_dir)
            print(f"Emergency checkpoint saved to: {save_dir}")
        except Exception as exc:
            print(f"Could not save emergency checkpoint: {exc}")
        sys.exit(1)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ibm-granite/granite-docling-258M for DocTag generation"
    )
    parser.add_argument(
        "--train_data",
        default="training_data_train.jsonl",
        help="Path to training JSONL (default: training_data_train.jsonl)",
    )
    parser.add_argument(
        "--val_data",
        default="training_data_val.jsonl",
        help="Path to validation JSONL; skipped if the file does not exist",
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID to fine-tune (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output_dir",
        default="doctag_model",
        help="Directory where the fine-tuned model is saved (default: doctag_model)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device training batch size (default: 1)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum token length (default: 2048)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (recommended)")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning (requires peft)",
    )
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05)",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to a previously trained LoRA adapter directory (must contain adapter_config.json). Loads adapter weights with a fresh optimizer.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="Docling finetune", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity (team or username).")
    parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (auto-generated if empty).")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb_mode", type=str, default="online", help="W&B mode: online, offline, or disabled.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume training from, or 'last' "
            "to automatically resume from the latest checkpoint in --output_dir. "
            "Restores model weights, optimizer state, scheduler, and global step."
        ),
    )
    args = parser.parse_args()

    if not Path(args.train_data).exists():
        raise FileNotFoundError(
            f"Training data not found: {args.train_data}\n"
        )
    
    print(f"Loading processor and model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    torch_dtype = (
        torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=torch_dtype)
    # Gradient checkpointing (enabled below) is incompatible with use_cache.
    # Set it explicitly here so the Trainer does not emit a warning and the
    # model behaves consistently during both training and evaluation.
    model.config.use_cache = False

    if args.lora or args.lora_adapter_path:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning. Install it with:\n"
                "  pip install peft"
            ) from exc

        if args.lora_adapter_path:
            adapter_path = Path(args.lora_adapter_path)
            if not (adapter_path / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"adapter_config.json not found in {args.lora_adapter_path}"
                )
            print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
            model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.use_wandb and args.wandb_mode != "disabled":
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

    print("Building datasets …")
    train_dataset = DocTagDataset(args.train_data, processor, args.max_length)

    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(args.val_data, processor, args.max_length)
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples | No validation file found at {args.val_data}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="steps",  
        save_steps=200, 
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if args.use_wandb else "none",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
        compute_metrics=make_compute_metrics(processor.tokenizer) if val_dataset is not None else None,

    )

    _install_signal_handlers(trainer)

    # Resolve 'last' keyword to the most recent checkpoint in output_dir.
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is not None and resume_checkpoint.lower() == "last":
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(args.output_dir)
        if last_ckpt is None:
            print(
                f"WARNING: --resume_from_checkpoint=last specified but no checkpoint "
                f"found in {args.output_dir!r}.  Starting from scratch."
            )
            resume_checkpoint = None
        else:
            resume_checkpoint = last_ckpt
            print(f"Resuming from latest checkpoint: {resume_checkpoint}")
    elif resume_checkpoint is not None:
        if not os.path.isdir(resume_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {resume_checkpoint}"
            )
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    # Detect stale checkpoint: if its global_step >= total training steps the
    # HuggingFace Trainer will declare training "done" before running a single
    # step.  When this happens, load only the model weights from the checkpoint
    # and restart training from step 0 instead of resuming trainer state.
    if resume_checkpoint is not None:
        state_path = os.path.join(resume_checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as _f:
                _ckpt_state = json.load(_f)
            _ckpt_step = _ckpt_state.get("global_step", 0)
            _steps_per_epoch = math.ceil(
                len(train_dataset) / (args.batch_size * args.grad_accum)
            )
            _total_steps = _steps_per_epoch * args.epochs
            if _ckpt_step >= _total_steps:
                print(
                    f"WARNING: checkpoint global_step ({_ckpt_step}) >= "
                    f"total training steps ({_total_steps}). "
                    "Loading model weights only and resetting training state to step 0."
                )
                # Load adapter weights into the already-wrapped PEFT model.
                try:
                    import safetensors.torch as _st
                    _adapter_path = os.path.join(
                        resume_checkpoint, "adapter_model.safetensors"
                    )
                    if os.path.exists(_adapter_path) and args.lora:
                        from peft import set_peft_model_state_dict
                        _sd = _st.load_file(_adapter_path)
                        set_peft_model_state_dict(model, _sd)
                        print(f"Loaded LoRA weights from {resume_checkpoint}")
                    else:
                        # Non-LoRA full model weights
                        _model_path = os.path.join(
                            resume_checkpoint, "model.safetensors"
                        )
                        if not os.path.exists(_model_path):
                            _model_path = os.path.join(
                                resume_checkpoint, "pytorch_model.bin"
                            )
                        if os.path.exists(_model_path):
                            if _model_path.endswith(".bin"):
                                _sd = torch.load(_model_path, map_location="cpu")
                            else:
                                _sd = _st.load_file(_model_path)
                            model.load_state_dict(_sd, strict=False)
                            print(f"Loaded model weights from {resume_checkpoint}")
                except Exception as _load_err:
                    print(f"WARNING: could not load weights from checkpoint: {_load_err}")
                resume_checkpoint = None  # do not hand stale state to Trainer

    print("Starting training …")
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.  Saving emergency checkpoint …")
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        trainer.save_model(emergency_dir)
        processor.save_pretrained(emergency_dir)
        print(f"Emergency checkpoint saved to {emergency_dir}")
        raise

    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()