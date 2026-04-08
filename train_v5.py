"""
train_v5.py — Optimised SmolDocling / Granite-Docling fine-tuning
==================================================================

Root-cause fixes over train_v4.py
----------------------------------
BUG-1  LoRA alpha == r  (scaling = α/r = 1.0)
       Previous default: r=16, α=32 → scaling=2.0 — amplified gradient updates
       and risked catastrophic forgetting of pre-trained OCR knowledge.

BUG-2  Fixed image resolution (not thumbnail).
       thumbnail() only downscales large images; small images passed through at
       native size, producing variable patch counts inside every batch.  The
       collator's zero-padding fallback injected false "black region" signal.
       All images are now resized to the model's native longest-edge resolution
       (auto-detected from processor or set via --native_resolution).

BUG-3  Frozen vision encoder  (--freeze_vision was the root cause of table
       learning failure).  The model is trained from:
           p(y | f_V(x) ; θ_dec)   where  ∂L/∂f_V = 0   (frozen)
       With Mongolian bank statement images far outside the pre-training
       distribution, the decoder LoRA cannot compensate.  This script unfreezes
       the top N ViT transformer blocks via --unfreeze_vision_layers N.

Improvements
------------
 4.  Differential learning rates — vision top-N layers train at
         lr_vision = lr * vision_lr_scale  (default 0.05×)
     so the encoder adapts without blowing away pre-trained low-level features.

 5.  LoRA rank raised to r=32 (was 16).  Table OCR over a large Mongolian
     vocabulary needs more adapter capacity.

 6.  max_length reduced to 2048 (was 8192).  Dataset p99 ≈ 1811 tokens.
     The 4× reduction cuts peak VRAM, allowing a bigger rank or batch size.

 7.  Warmup raised to 10 % of steps (was 5 %).  Larger distribution shift
     needs a gentler ramp.

 8.  Explicit gradient clipping (max_grad_norm=1.0) via SFTConfig.

 9.  Layer-wise LR decay on vision blocks:  lr_i = lr_vision × decay^(N-i)
     so the deepest unfrozen block gets the full vision LR and lower blocks
     get progressively less — protects lower-level patch features.

10.  All existing metrics retained (tag_acc, loc_mae, tree_edit_dist) plus
     new otsl_row_acc — fraction of <nl>-delimited rows in correct order.

Usage
-----
# Recommended (8–16 GB GPU, e.g. RTX 3090/4090):
python train_v5.py \\
    --train_data  training_data_train.jsonl \\
    --val_data    training_data_val.jsonl   \\
    --qlora                                 \\
    --unfreeze_vision_layers 4              \\
    --lora_r 32                             \\
    --output_dir doctag_model_v5

# Tight VRAM (8 GB, e.g. RTX 3070):
python train_v5.py \\
    --train_data  training_data_train.jsonl \\
    --val_data    training_data_val.jsonl   \\
    --qlora                                 \\
    --unfreeze_vision_layers 2              \\
    --lora_r 16                             \\
    --batch_size 1  --grad_accum 16         \\
    --output_dir doctag_model_v5_tight

# Large GPU (40–80 GB, e.g. A100/H100):
python train_v5.py \\
    --train_data  training_data_train.jsonl \\
    --val_data    training_data_val.jsonl   \\
    --unfreeze_vision_layers 8              \\
    --lora_r 64                             \\
    --batch_size 8  --grad_accum 2          \\
    --lr 2e-4       --vision_lr_scale 0.05  \\
    --output_dir doctag_model_v5_large

# Resume from checkpoint:
python train_v5.py \\
    --train_data  training_data_train.jsonl \\
    --adapter     doctag_model_v5/checkpoint-1000 \\
    --output_dir  doctag_model_v5_resumed
"""

import argparse
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Environment setup — must happen before any torch/transformers import
# ---------------------------------------------------------------------------
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey-patch json so torch.dtype objects serialise without crashing
import json as _json_mod

_json_default_orig = _json_mod.JSONEncoder().default


def _patched_json_default(self, o):
    try:
        import torch as _t
        if isinstance(o, _t.dtype):
            return str(o)
    except Exception:
        pass
    return _json_default_orig(o)


_json_mod.JSONEncoder.default = _patched_json_default

# ---------------------------------------------------------------------------
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

_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
    "ched", "fcel", "rhed", "ucel", "lcel", "ecel", "nl",
})


# ---------------------------------------------------------------------------
# Helpers: auto-detect native image resolution from processor
# ---------------------------------------------------------------------------

def get_native_resolution(processor) -> int:
    """
    Return the model's native longest-edge resolution from the image processor
    config.  Falls back to 512 if no config field is found.

    SmolVLM-256M  →  512
    SmolVLM-500M  →  512
    Granite-Docling-258M (SmolVLM base)  →  512
    """
    img_proc = getattr(processor, "image_processor", processor)
    for attr in ("size", "image_size", "crop_size"):
        v = getattr(img_proc, attr, None)
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            return v.get("longest_edge") or v.get("height") or v.get("width") or 512
    return 512


# ---------------------------------------------------------------------------
# DocTag metric helpers
# ---------------------------------------------------------------------------

def _extract_structural_tags(s: str) -> list:
    return [
        ("/" if slash else "") + tag
        for slash, tag in re.findall(r'<(/?)(\w+(?:_\w+)*)(?:[^>]*)?>', s)
        if tag in _STRUCTURAL_TAGS
    ]


def _tag_accuracy(pred: str, ref: str) -> float:
    pt = _extract_structural_tags(pred)
    rt = _extract_structural_tags(ref)
    if not rt and not pt:
        return 1.0
    if not rt or not pt:
        return 0.0
    return sum(p == r for p, r in zip(pt, rt)) / max(len(pt), len(rt))


def _loc_mae(pred: str, ref: str) -> float:
    """MAE on <loc_N> values (0–500 scale)."""
    pl = [int(n) for n in re.findall(r'<loc_(\d+)>', pred)]
    rl = [int(n) for n in re.findall(r'<loc_(\d+)>', ref)]
    if not rl:
        return 0.0
    total = sum(
        abs(pl[i] - rl[i]) if i < len(pl) else rl[i]
        for i in range(len(rl))
    )
    return total / len(rl)


def _tree_edit_dist(pred: str, ref: str) -> float:
    """Normalised Zhang-Shasha tree edit distance; falls back to Levenshtein."""
    try:
        import zss

        def _build(s):
            class _N:
                def __init__(self, label):
                    self.label, self.children = label, []

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

        pt_ = _to_zss(_build(pred))
        rt_ = _to_zss(_build(ref))
        dist = zss.simple_distance(pt_, rt_)
        mx = _size(pt_) + _size(rt_)
        return dist / mx if mx > 0 else 0.0

    except ImportError:
        pt = _extract_structural_tags(pred)
        rt = _extract_structural_tags(ref)
        if not pt and not rt:
            return 0.0
        n, m = len(pt), len(rt)
        if n == 0 or m == 0:
            return 1.0
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if pt[i - 1] == rt[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[m] / max(n, m)


def _otsl_row_accuracy(pred: str, ref: str) -> float:
    """
    NEW METRIC (v5): fraction of OTSL rows predicted in correct order.

    Each row is defined as the token sequence between consecutive <nl> markers.
    We strip loc tokens and compare row token sequences pairwise.  This directly
    measures whether the model learned the table row structure — the primary
    failure mode identified in the analysis.
    """
    _loc_re = re.compile(r'<loc_\d+>')

    def _rows(s):
        # Grab content inside <otsl>...</otsl>; fall back to whole string
        m = re.search(r'<otsl>(.*?)</otsl>', s, re.DOTALL)
        body = m.group(1) if m else s
        body = _loc_re.sub("", body)
        rows = [r.strip() for r in body.split("<nl>") if r.strip()]
        return rows

    pr = _rows(pred)
    rr = _rows(ref)
    if not rr:
        return 1.0 if not pr else 0.0
    if not pr:
        return 0.0
    matches = sum(p == r for p, r in zip(pr, rr))
    return matches / max(len(pr), len(rr))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DocTagDataset(Dataset):
    """
    Loads (image_path, doctag) pairs from a JSONL file.

    Key change from v4
    ------------------
    Images are resized to a FIXED resolution (native_resolution × native_resolution)
    using high-quality LANCZOS resampling — not thumbnail().  This guarantees:
      • Identical patch grids for every sample in every batch.
      • No collator zero-padding fallback (torch.stack always succeeds).
      • Consistent positional embeddings for the vision encoder.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_length: int = 2048,
        native_resolution: int = 512,
    ):
        self.records: list = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length
        self.native_resolution = native_resolution

        resampling_mod = getattr(Image, "Resampling", Image)
        self._lanczos = resampling_mod.LANCZOS

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")

        # FIX (BUG-2): always resize to a fixed square canvas so every sample
        # produces the same number of image patches.  Using resize() (not
        # thumbnail()) ensures upscaling of small images too.
        target = self.native_resolution
        image = image.resize((target, target), self._lanczos)

        # Normalise legacy <image>/<image_caption> tags
        doctag = record["doctag"]
        doctag = re.sub(r"<image>", "<picture>", doctag)
        doctag = re.sub(r"</image>", "</picture>", doctag)

        messages_full = [
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text",  "text": doctag}]},
        ]
        text_full = self.processor.apply_chat_template(messages_full, add_generation_prompt=False)

        messages_prompt = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}]},
        ]
        text_prompt = self.processor.apply_chat_template(messages_prompt, add_generation_prompt=True)

        prompt_ids = self.processor.tokenizer(
            text_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        # Pass the fixed-size image; explicit size hint matches native_resolution
        inputs = self.processor(
            images=image,
            text=text_full,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            size={"longest_edge": target},
        )
        image.close()

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        labels[: len(prompt_ids)] = -100   # mask the prompt tokens from loss
        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Collator — simplified (no zero-padding needed after fixed resize)
# ---------------------------------------------------------------------------

def doctag_data_collator(features: list) -> dict:
    """
    Pad token sequences to the longest sequence in the batch.

    pixel_values and pixel_attention_mask can now be stacked directly because
    every image was resized to the same resolution — no fallback zero-padding.
    """
    from torch.nn.utils.rnn import pad_sequence

    batch: dict = {}
    for key in features[0].keys():
        tensors = [f[key] for f in features]
        if key == "labels":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
        elif key == "attention_mask":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        elif key in ("pixel_values", "pixel_attention_mask"):
            # After fixed resize all tensors have identical shape — direct stack.
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError as exc:
                # Should never happen with fixed resize; surface a clear error.
                shapes = [t.shape for t in tensors]
                raise RuntimeError(
                    f"pixel_values stack failed — shapes are not uniform: {shapes}. "
                    "Ensure all images are resized to the same native_resolution."
                ) from exc
        else:
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
    return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1).cpu()


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
        preds_dec  = np.clip(pred_ids, 0, vocab_size - 1)
        pred_strs  = tokenizer.batch_decode(preds_dec.tolist(),  skip_special_tokens=True)
        ref_strs   = tokenizer.batch_decode(labels_dec.tolist(), skip_special_tokens=True)

        tag_acc    = float(np.mean([_tag_accuracy(p, r)      for p, r in zip(pred_strs, ref_strs)]))
        loc_mae    = float(np.mean([_loc_mae(p, r)            for p, r in zip(pred_strs, ref_strs)]))
        tree_edit  = float(np.mean([_tree_edit_dist(p, r)     for p, r in zip(pred_strs, ref_strs)]))
        otsl_row   = float(np.mean([_otsl_row_accuracy(p, r)  for p, r in zip(pred_strs, ref_strs)]))

        return {
            "token_acc":      token_acc,
            "tag_acc":        tag_acc,
            "loc_mae":        loc_mae,
            "tree_edit_dist": tree_edit,
            "otsl_row_acc":   otsl_row,     # NEW — primary table-learning signal
        }
    return _compute


# ---------------------------------------------------------------------------
# Vision encoder: partial unfreeze + differential LR helpers
# ---------------------------------------------------------------------------

def _collect_vision_blocks(model) -> list:
    """
    Return a list of (name_prefix, module) for each ViT transformer block in
    the vision encoder, ordered from bottom (index 0) to top (last = closest
    to the language decoder).

    Supports the common naming patterns found in SmolVLM / SigLIP / CLIP
    based models used in docling-class architectures.
    """
    candidates = []
    for name, module in model.named_modules():
        # Match patterns: vision_model.encoder.layers.N  /  image_model.encoder.layers.N
        #                 model.vision_tower.vision_model.encoder.layers.N
        if re.search(r'\.(encoder|transformer)\.(layers|blocks)\.\d+$', name):
            candidates.append((name, module))

    # Sort by layer index so index 0 = bottom, -1 = top
    def _layer_idx(name):
        m = re.search(r'\.(\d+)$', name)
        return int(m.group(1)) if m else -1

    candidates.sort(key=lambda x: _layer_idx(x[0]))
    return candidates


def setup_vision_partial_unfreeze(
    model,
    unfreeze_top_n: int,
    vision_lr: float,
    layer_decay: float = 0.85,
) -> list:
    """
    Freeze all vision encoder parameters, then selectively unfreeze the top
    `unfreeze_top_n` transformer blocks with layer-wise LR decay.

    Layer-wise decay formula:
        lr_i = vision_lr × layer_decay^(unfreeze_top_n - 1 - i)

    where i=0 is the lowest unfrozen block (gets smallest LR) and
    i=unfreeze_top_n-1 is the topmost block (gets full vision_lr).

    Returns
    -------
    List of {"params": [...], "lr": float, "name": str} dicts for AdamW.
    """
    # Step 1: freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: always keep LoRA/DoRA adapter params trainable
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "dora" in name.lower():
            param.requires_grad = True

    if unfreeze_top_n <= 0:
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Vision] All vision layers frozen.  Trainable: {total_trainable:,} params")
        return []

    # Step 3: collect and unfreeze top-N vision blocks
    all_blocks = _collect_vision_blocks(model)
    if not all_blocks:
        print("[Vision] WARNING: could not locate ViT transformer blocks.  "
              "Check model architecture.  Falling back to fully frozen encoder.")
        return []

    top_blocks = all_blocks[-unfreeze_top_n:]
    vision_param_groups = []

    for rank_from_top, (block_name, block_module) in enumerate(reversed(top_blocks)):
        # rank_from_top=0  → topmost block  → full vision_lr
        # rank_from_top=N-1 → lowest unfrozen block → vision_lr × decay^(N-1)
        block_lr = vision_lr * (layer_decay ** rank_from_top)
        block_params = []

        for pname, param in block_module.named_parameters():
            if not param.is_floating_point():
                continue  # skip quantized int weights (4-bit NF4 in QLoRA)
            if not param.requires_grad:  # not already a LoRA param
                param.requires_grad = True
            block_params.append(param)

        if block_params:
            vision_param_groups.append({
                "params": block_params,
                "lr":     block_lr,
                "name":   block_name,
            })
            n_params = sum(p.numel() for p in block_params)
            print(f"  [Vision] Unfreeze block: {block_name}  |  lr={block_lr:.2e}  |  "
                  f"{n_params:,} params")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Vision] Unfroze top-{unfreeze_top_n} blocks.  "
          f"Total trainable: {total_trainable:,} params")
    return vision_param_groups


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
            print(f"Could not save checkpoint: {exc}")
        sys.exit(1)

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolDocling / Granite-Docling (v5) — all root-cause fixes applied",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data & model ---
    parser.add_argument("--train_data",  default="training_data_train.jsonl")
    parser.add_argument("--val_data",    default="training_data_val.jsonl")
    parser.add_argument("--model_id",    default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir",  default="doctag_model_v5")
    parser.add_argument("--adapter",     type=str, default="",
                        help="Path to existing PEFT adapter directory to resume from.")

    # --- Training loop ---
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=2,
                        help="Per-device train batch size.  Aim for effective batch >= 16.")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Per-device eval batch size.  Default 1 to avoid OOM on long val sequences. "
                             "Set to 0 to use the same value as --batch_size.")
    parser.add_argument("--grad_accum",  type=int,   default=8,
                        help="Gradient accumulation steps.  effective_batch = batch_size × grad_accum × gpus")
    parser.add_argument("--max_length",  type=int,   default=2048,
                        help="Max token length.  Dataset p99 ≈ 1811 tokens — 2048 is sufficient.")

    # --- Image resolution ---
    parser.add_argument("--native_resolution", type=int, default=0,
                        help="Resize all images to this longest-edge before processing. "
                             "0 = auto-detect from processor config (recommended).")

    # --- Learning rate ---
    parser.add_argument("--lr",              type=float, default=2e-4,
                        help="Base LR for LoRA/DoRA decoder parameters.")
    parser.add_argument("--vision_lr_scale", type=float, default=0.05,
                        help="vision_lr = lr × vision_lr_scale.  "
                             "Keeps vision encoder adaptation conservative.")
    parser.add_argument("--layer_decay",     type=float, default=0.85,
                        help="Layer-wise LR decay for unfrozen vision blocks.  "
                             "lr_i = vision_lr × layer_decay^(rank_from_top).")

    # --- LoRA / DoRA --- FIX: alpha default == r default (scaling = 1.0)
    parser.add_argument("--lora_r",       type=int,   default=32,
                        help="LoRA rank.  32 is a good default for Mongolian OCR vocabulary.")
    parser.add_argument("--lora_alpha",   type=int,   default=0,
                        help="LoRA alpha.  0 = auto-set equal to lora_r (scaling=1.0, recommended).")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_dora",      action="store_true",
                        help="Disable DoRA and use plain LoRA (slightly faster, lower quality).")

    # --- Vision encoder ---
    parser.add_argument("--unfreeze_vision_layers", type=int, default=4,
                        help="Number of top ViT transformer blocks to unfreeze. "
                             "0 = fully frozen (v4 behaviour). "
                             "4 is a good default for 8–16 GB GPU with QLoRA.")

    # --- Hardware ---
    parser.add_argument("--qlora",      action="store_true",
                        help="Load base model in 4-bit NF4 (QLoRA).  Recommended for <24 GB GPU.")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Flash Attention 2 (requires: pip install flash-attn --no-build-isolation).")

    # --- Eval / save ---
    parser.add_argument("--eval_steps",       type=int, default=200)
    parser.add_argument("--save_steps",       type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Max step checkpoints to keep (default: 2 to avoid disk full).")

    # --- W&B ---
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--wandb_project",  type=str, default="Docling-finetune-v5")
    parser.add_argument("--wandb_entity",   type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags",     type=str, default="")
    parser.add_argument("--wandb_mode",     type=str, default="online",
                        choices=["online", "offline", "disabled"])

    args = parser.parse_args()

    # --- W&B init ---
    if args.use_wandb:
        import importlib
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed.  Run: pip install wandb")
        import wandb
        run_name = args.wandb_run_name.strip() or (
            f"doctag-v5-{Path(args.train_data).stem}-"
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
                config=vars(args),
            )
        except Exception as _we:
            print(f"[W&B] Online init failed ({_we}); falling back to offline mode.")
            wandb.init(project=args.wandb_project, name=run_name, mode="offline",
                        tags=tags or None, config=vars(args))

    # --- Validate data paths ---
    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    # --- Fix: alpha == r when alpha == 0 (auto) ---
    # BUG-1 FIX: ensure scaling = alpha/r = 1.0 by default
    if args.lora_alpha == 0:
        args.lora_alpha = args.lora_r
    if args.lora_alpha != args.lora_r:
        print(f"[LoRA] WARNING: lora_alpha ({args.lora_alpha}) != lora_r ({args.lora_r}).  "
              f"Scaling = {args.lora_alpha / args.lora_r:.2f}x.  "
              "Recommended: lora_alpha == lora_r (scaling = 1.0).")
    else:
        print(f"[LoRA] r={args.lora_r}  alpha={args.lora_alpha}  scaling=1.0  ✓")

    # --- W&B init ---
    if args.use_wandb:
        import importlib
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed.  Run: pip install wandb")
        import wandb, datetime
        run_name = args.wandb_run_name.strip() or (
            f"doctag-v5-{Path(args.train_data).stem}-"
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
                config=vars(args),
            )
        except Exception as _we:
            print(f"[W&B] Online init failed ({_we}); falling back to offline mode.")
            wandb.init(project=args.wandb_project, name=run_name, mode="offline",
                       tags=tags or None, config=vars(args))

    # ---------------------------------------------------------------------------
    print(f"\nLoading processor and model: {args.model_id}")
    # ---------------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)

    # BUG-2 FIX: determine the fixed resolution every image will be resized to
    if args.native_resolution > 0:
        native_res = args.native_resolution
    else:
        native_res = get_native_resolution(processor)
    print(f"[Image] All images will be resized to {native_res}×{native_res}  (fixed, no padding)")

    # --- QLoRA: 4-bit NF4 ---
    bnb_config = None
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("[QLoRA] Loading model in 4-bit NF4")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("[QLoRA] Model prepared for kbit training")

    # ---------------------------------------------------------------------------
    # BUG-3 FIX: partial vision encoder unfreeze with differential LR
    # ---------------------------------------------------------------------------
    vision_lr = args.lr * args.vision_lr_scale
    print(f"\n[Vision] unfreeze_top_n={args.unfreeze_vision_layers}  "
          f"vision_lr={vision_lr:.2e}  layer_decay={args.layer_decay}")
    vision_param_groups = setup_vision_partial_unfreeze(
        model,
        unfreeze_top_n=args.unfreeze_vision_layers,
        vision_lr=vision_lr,
        layer_decay=args.layer_decay,
    )

    # --- DoRA / LoRA config ---
    from peft import LoraConfig, PeftModel, get_peft_model

    if args.adapter and Path(args.adapter).exists():
        print(f"[Adapter] Loading existing adapter from {args.adapter} …")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print("[Adapter] Adapter loaded — continuing training.")
    else:
        if args.adapter:
            print(f"[Adapter] WARNING: path '{args.adapter}' not found; starting fresh.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,          # FIX: == r  →  scaling = 1.0
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=args.lora_dropout,
            bias="none",
            use_dora=not args.no_dora,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ---------------------------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------------------------
    print("\nBuilding datasets …")
    train_dataset = DocTagDataset(
        args.train_data, processor, args.max_length, native_res,
    )
    val_dataset: Optional[DocTagDataset] = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(
            args.val_data, processor, args.max_length, native_res,
        )
        print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples  |  No val file at {args.val_data}")

    # ---------------------------------------------------------------------------
    # Build optimizer with differential learning rates
    # ---------------------------------------------------------------------------
    # Default LoRA/DoRA decoder params use the base lr
    decoder_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(
            p is vp
            for vg in vision_param_groups
            for vp in vg["params"]
        )
    ]

    # Assemble optimizer param groups
    # vision blocks get their layer-wise LRs; decoder LoRA gets args.lr
    optimizer_param_groups = vision_param_groups + [
        {"params": decoder_params, "lr": args.lr, "name": "decoder_lora"},
    ]

    print(f"\n[Optimizer] Param groups:")
    for g in optimizer_param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:60s}  lr={g['lr']:.2e}  params={n_p:,}")

    # We pass the optimizer directly to Trainer so differential LRs are honoured.
    # SFTConfig.optim is ignored when an optimizer is supplied.
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=args.lr,          # default; each group overrides this
        weight_decay=0.01,
        fused=True if torch.cuda.is_available() else False,
    )

    # ---------------------------------------------------------------------------
    # SFTConfig
    # ---------------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # LR / scheduler — base LR for scheduler; groups override actual LR
        learning_rate=args.lr,
        warmup_ratio=0.10,                     # FIX: 10 % (was 5 %) for larger distribution shift
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,                     # FIX: explicit gradient clipping
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=args.eval_steps if val_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="otsl_row_acc",  # NEW primary metric
        greater_is_better=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="wandb" if args.use_wandb else "none",
        eval_accumulation_steps=1,
        # Skip TRL's internal tokenisation — DocTagDataset handles it
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ---------------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
        optimizers=(optimizer, None),          # (optimizer, lr_scheduler=None → auto)
        compute_metrics=make_compute_metrics(processor.tokenizer) if val_dataset else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset else None,
    )

    _install_signal_handlers(trainer)

    # ---------------------------------------------------------------------------
    # Log config summary
    # ---------------------------------------------------------------------------
    eff_batch = args.batch_size * args.grad_accum
    n_steps   = (len(train_dataset) // args.batch_size) * args.epochs // args.grad_accum
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  train_v5 — Configuration Summary
╠══════════════════════════════════════════════════════════════╣
║  Model           {args.model_id}
║  Train samples   {len(train_dataset):,}
║  Val samples     {len(val_dataset) if val_dataset else 0:,}
║  Epochs          {args.epochs}
║  Batch (device)  {args.batch_size}  ×  grad_accum={args.grad_accum}  →  eff={eff_batch}
║  Est. steps      ~{n_steps:,}
║  Image res.      {native_res}×{native_res}  (fixed, all images)
║  LoRA  r/α       {args.lora_r}/{args.lora_alpha}  (scaling={args.lora_alpha/args.lora_r:.2f})
║  Vision layers   top-{args.unfreeze_vision_layers} unfrozen  @  lr={vision_lr:.2e}
║  Decoder LR      {args.lr:.2e}
║  QLoRA           {'yes' if args.qlora else 'no'}
║  Flash Attn      {'yes' if args.flash_attn else 'no'}
║  Best metric     otsl_row_acc  (table row accuracy)
╚══════════════════════════════════════════════════════════════╝
""")

    # ---------------------------------------------------------------------------
    print("Starting training …")
    # ---------------------------------------------------------------------------
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
    finally:
        # Always save a pre-exit checkpoint regardless of how training ends
        # (clean finish, exception, or signal-triggered stop)
        pre_exit_dir = os.path.join(args.output_dir, "pre_exit_checkpoint")
        print(f"\n[Checkpoint] Saving pre-exit checkpoint to {pre_exit_dir} …")
        try:
            trainer.save_model(pre_exit_dir)
            processor.save_pretrained(pre_exit_dir)
            print(f"[Checkpoint] Pre-exit checkpoint saved to {pre_exit_dir}")
        except Exception as save_exc:
            print(f"[WARNING] Could not save pre-exit checkpoint: {save_exc}")

    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()