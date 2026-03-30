"""
Optimised fine-tuning script for ibm-granite/granite-docling-258M (DocTag generation).

This is an *alternative* to ``train.py`` that addresses common reasons
the base training loop may plateau or fail to improve the model:

Key improvements over ``train.py``
-----------------------------------
* **Image augmentation** — configurable on-the-fly transforms (colour jitter,
  small rotation, Gaussian blur, perspective warp, random erasing) prevent the
  model from memorising pixel-level details.
* **Better LoRA defaults** — higher rank (32), targets MLP projection layers
  *and* attention heads, uses rsLoRA scaling for more stable updates.
* **Cosine-with-restarts scheduler** — periodic LR re-warming helps escape
  shallow local minima.
* **Weight decay + label smoothing** — standard regularisation that
  significantly reduces over-fitting on small datasets.
* **Gradient clipping** — stabilises early training steps.
* **tf32 on Ampere+ GPUs** — free ~2× speed-up on supported hardware.
* **Configurable eval frequency** — evaluate every N steps instead of
  only per epoch, giving faster feedback on small datasets.
* **Length-sorted batching** — groups similar-length samples to reduce
  padding overhead and speed up each step.
* **torch.compile** — optional ahead-of-time compilation for an extra
  10-30 % speed-up (PyTorch ≥ 2.1).

The script is *fully compatible* with the existing JSONL dataset format
(``{"image": "...", "doctag": "..."}``), ``prepare_dataset.py``, and
``inference.py``.

Usage
-----
    # Quick start (recommended defaults)
    python train_v2.py --train_data training_data_train.jsonl \\
                       --val_data   training_data_val.jsonl   \\
                       --lora --bf16 --augment

    # Resume from a previous LoRA checkpoint
    python train_v2.py --train_data training_data_train.jsonl \\
                       --val_data   training_data_val.jsonl   \\
                       --lora --bf16 --augment \\
                       --lora_adapter_path doctag_model/checkpoint-1000

    # Full fine-tune (no LoRA) with aggressive augmentation
    python train_v2.py --train_data training_data_train.jsonl \\
                       --val_data   training_data_val.jsonl   \\
                       --bf16 --augment --augment_strength heavy \\
                       --freeze_vision --epochs 10
"""

import argparse
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import re
import signal
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ── defaults ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"
SYSTEM_PROMPT = "Convert this document page to DocTags format."
_MAX_LOSS_FOR_PERPLEXITY = 20

# DocTag structural tags — shared with train.py / metrics helpers
_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
})


# ── metric helpers (same as train.py to keep outputs compatible) ──────────

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


# ── data augmentation ─────────────────────────────────────────────────────

# Preset augmentation strengths — each maps to
# (probability that *any* augmentation is applied, and the max intensities).
_AUG_PRESETS: dict[str, dict] = {
    "light": {
        "prob": 0.3,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.1,
        "hue": 0.02,
        "rotation": 2,
        "blur_radius": 0.5,
        "perspective": 0.02,
        "erasing_prob": 0.05,
        "noise_std": 5,
    },
    "medium": {
        "prob": 0.5,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.03,
        "rotation": 3,
        "blur_radius": 1.0,
        "perspective": 0.04,
        "erasing_prob": 0.10,
        "noise_std": 10,
    },
    "heavy": {
        "prob": 0.7,
        "brightness": 0.35,
        "contrast": 0.35,
        "saturation": 0.3,
        "hue": 0.05,
        "rotation": 5,
        "blur_radius": 1.5,
        "perspective": 0.06,
        "erasing_prob": 0.15,
        "noise_std": 15,
    },
}


def _augment_image(image: Image.Image, cfg: dict) -> Image.Image:
    """Apply randomised augmentations to a PIL Image.

    The transform pipeline is *document-aware*:  we avoid aggressive crops or
    flips that would invalidate the spatial DocTag annotations (bounding-box
    coordinates).  Instead we focus on *photometric* augmentations (colour,
    noise, blur) and *very mild* geometric ones (≤5° rotation, tiny
    perspective warp) that a document scanner might naturally introduce.
    """
    if random.random() > cfg["prob"]:
        return image  # skip augmentation entirely for this sample

    # ── photometric ──────────────────────────────────────────────────
    # Brightness
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-cfg["brightness"], cfg["brightness"])
        image = ImageEnhance.Brightness(image).enhance(factor)

    # Contrast
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-cfg["contrast"], cfg["contrast"])
        image = ImageEnhance.Contrast(image).enhance(factor)

    # Saturation
    if random.random() < 0.3:
        factor = 1.0 + random.uniform(-cfg["saturation"], cfg["saturation"])
        image = ImageEnhance.Color(image).enhance(factor)

    # Gaussian blur
    if random.random() < 0.2:
        radius = random.uniform(0, cfg["blur_radius"])
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # Additive Gaussian noise (simulates scanner noise)
    if random.random() < 0.2:
        import numpy as np
        arr = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, cfg["noise_std"], arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

    # ── mild geometry ────────────────────────────────────────────────
    # Small rotation (fills white — typical for scanned documents)
    if random.random() < 0.3:
        angle = random.uniform(-cfg["rotation"], cfg["rotation"])
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    # Tiny perspective warp
    if random.random() < 0.15:
        w, h = image.size
        d = cfg["perspective"]
        dx = int(w * d)
        dy = int(h * d)
        coeffs = _perspective_coefficients(
            [(0, 0), (w, 0), (w, h), (0, h)],
            [
                (random.randint(0, dx), random.randint(0, dy)),
                (w - random.randint(0, dx), random.randint(0, dy)),
                (w - random.randint(0, dx), h - random.randint(0, dy)),
                (random.randint(0, dx), h - random.randint(0, dy)),
            ],
        )
        image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR,
                                fillcolor=(255, 255, 255))

    # Random rectangular erasing (simulates small print defects / stains)
    if random.random() < cfg["erasing_prob"]:
        import numpy as np
        arr = np.array(image)
        eh = random.randint(2, max(3, arr.shape[0] // 40))
        ew = random.randint(2, max(3, arr.shape[1] // 40))
        ey = random.randint(0, arr.shape[0] - eh)
        ex = random.randint(0, arr.shape[1] - ew)
        arr[ey:ey + eh, ex:ex + ew] = random.randint(200, 255)
        image = Image.fromarray(arr)

    return image


def _perspective_coefficients(src_pts, dst_pts):
    """Compute 8-coefficient perspective transform from 4 point pairs."""
    import numpy as np
    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for pt in src_pts for c in pt], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


# ── length-sorted sampler (reduces padding waste) ─────────────────────────

class LengthSortedSampler(Sampler):
    """Yield indices sorted by sequence length (longest first) for each epoch.

    Within each *bucket* of ``bucket_size`` samples the order is shuffled
    to retain some stochasticity while keeping sequences of similar length
    together — this minimises padding overhead and speeds up each step.
    """

    def __init__(self, lengths: list[int], bucket_size: int = 64):
        self.lengths = lengths
        self.bucket_size = max(1, bucket_size)

    def __iter__(self):
        idxs = list(range(len(self.lengths)))
        # Sort by length (descending)
        idxs.sort(key=lambda i: -self.lengths[i])
        # Shuffle within buckets
        bucketed: list[int] = []
        for start in range(0, len(idxs), self.bucket_size):
            bucket = idxs[start:start + self.bucket_size]
            random.shuffle(bucket)
            bucketed.extend(bucket)
        return iter(bucketed)

    def __len__(self) -> int:
        return len(self.lengths)


# ── dataset ───────────────────────────────────────────────────────────────

class DocTagDatasetV2(Dataset):
    """Enhanced DocTag dataset with optional on-the-fly image augmentation.

    Differences from the ``DocTagDataset`` in ``train.py``:
    * Applies a configurable augmentation pipeline to the image.
    * Pre-computes and stores token lengths for length-sorted batching.
    * Caches prompt template tokenisation (computed once, reused per sample).
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_length: int = 4096,
        augment_cfg: dict | None = None,
    ):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length
        self.augment_cfg = augment_cfg  # None → no augmentation

        # Pre-compute approximate token lengths for length-sorted batching.
        # We use char count as a fast proxy (actual token count is proportional).
        self.approx_lengths: list[int] = [len(r["doctag"]) for r in self.records]

        # Pre-tokenise the prompt template (only the text part — no image).
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        self._prompt_text = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True
        )
        self._prompt_ids = processor.tokenizer(
            self._prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).input_ids

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        doctag = record["doctag"]

        # ── augment ──────────────────────────────────────────────────
        if self.augment_cfg is not None:
            image = _augment_image(image, self.augment_cfg)

        # ── tokenise ─────────────────────────────────────────────────
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

        inputs = self.processor(
            images=image,
            text=text_full,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        image.close()

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Mask prompt tokens
        labels = inputs["input_ids"].clone()
        prompt_len = len(self._prompt_ids)
        labels[:prompt_len] = -100
        inputs["labels"] = labels
        return inputs


# ── collator (same as train.py) ──────────────────────────────────────────

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


# ── preprocess logits / compute metrics (same as train.py) ───────────────

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


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
        labels_for_decode = np.where(labels == -100, pad_id, labels)
        labels_for_decode = np.clip(labels_for_decode, 0, vocab_size - 1)
        pred_ids_for_decode = np.clip(pred_ids, 0, vocab_size - 1)
        pred_strs = tokenizer.batch_decode(pred_ids_for_decode.tolist(), skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(labels_for_decode.tolist(), skip_special_tokens=True)

        tag_acc = float(sum(_tag_accuracy(p, r) for p, r in zip(pred_strs, ref_strs)) / max(len(pred_strs), 1))
        loc_mae = float(sum(_loc_mae(p, r) for p, r in zip(pred_strs, ref_strs)) / max(len(pred_strs), 1))
        tree_edit = float(sum(_tree_edit_dist(p, r) for p, r in zip(pred_strs, ref_strs)) / max(len(pred_strs), 1))

        return {
            "token_acc": token_acc,
            "tag_acc": tag_acc,
            "loc_mae": loc_mae,
            "tree_edit_dist": tree_edit,
        }
    return _compute


# ── callbacks ─────────────────────────────────────────────────────────────

class EvalMemoryCallback(TrainerCallback):
    """Clears GPU cache around evaluation and toggles use_cache."""

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model is not None and hasattr(model, "config"):
            model.config.use_cache = True

    def on_prediction_step(self, args, state, control, **kwargs):
        pass

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "config"):
            model.config.use_cache = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TrainingMetricsCallback(TrainerCallback):
    """Logs structured metrics and saves history + plots at the end."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.history: dict[str, list] = {
            "train_steps": [],
            "train_loss": [],
            "train_perplexity": [],
            "train_lr": [],
            "train_grad_norm": [],
            "eval_steps": [],
            "eval_epochs": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "eval_token_acc": [],
            "eval_tag_acc": [],
            "eval_loc_mae": [],
            "eval_tree_edit_dist": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        epoch = state.epoch or 0.0

        if "loss" in logs and "eval_loss" not in logs:
            train_loss = logs["loss"]
            perplexity = math.exp(min(train_loss, _MAX_LOSS_FOR_PERPLEXITY))
            lr = logs.get("learning_rate", 0.0)
            grad_norm = logs.get("grad_norm")
            self.history["train_steps"].append(step)
            self.history["train_loss"].append(train_loss)
            self.history["train_perplexity"].append(perplexity)
            self.history["train_lr"].append(lr)
            if grad_norm is not None:
                self.history["train_grad_norm"].append(grad_norm)
            grad_str = f"  |  GradNorm: {grad_norm:.4f}" if grad_norm is not None else ""
            print(
                f"[Step {step:>6} | Epoch {epoch:.2f}]  "
                f"Train loss: {train_loss:.4f}  |  "
                f"Perplexity: {perplexity:.2f}  |  "
                f"LR: {lr:.2e}{grad_str}"
            )

        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            perplexity = math.exp(min(eval_loss, _MAX_LOSS_FOR_PERPLEXITY))
            token_acc = logs.get("eval_token_acc")
            tag_acc = logs.get("eval_tag_acc")
            loc_mae = logs.get("eval_loc_mae")
            tree_edit = logs.get("eval_tree_edit_dist")
            self.history["eval_steps"].append(step)
            self.history["eval_epochs"].append(epoch)
            self.history["eval_loss"].append(eval_loss)
            self.history["eval_perplexity"].append(perplexity)
            if token_acc is not None:
                self.history["eval_token_acc"].append(token_acc)
            if tag_acc is not None:
                self.history["eval_tag_acc"].append(tag_acc)
            if loc_mae is not None:
                self.history["eval_loc_mae"].append(loc_mae)
            if tree_edit is not None:
                self.history["eval_tree_edit_dist"].append(tree_edit)
            metrics_str = ""
            if token_acc is not None:
                metrics_str += f"  |  TokenAcc: {token_acc:.4f}"
            if tag_acc is not None:
                metrics_str += f"  |  TagAcc: {tag_acc:.4f}"
            if loc_mae is not None:
                metrics_str += f"  |  LocMAE: {loc_mae:.2f}"
            if tree_edit is not None:
                metrics_str += f"  |  TreeEdit: {tree_edit:.4f}"
            print(
                f"[Step {step:>6} | Epoch {epoch:.1f}]  "
                f"Val loss:   {eval_loss:.4f}  |  "
                f"Perplexity: {perplexity:.2f}{metrics_str}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        print(f"Training history saved to {history_path}")
        _plot_training_curves(self.history, self.output_dir)


# ── plotting ──────────────────────────────────────────────────────────────

def _plot_training_curves(history: dict, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.  pip install matplotlib")
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

    panels = []
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
    fig.suptitle("Training Progress (train_v2)", fontsize=14, y=1.01)

    for i, (title, xs, ys, color, xlabel, ylabel) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        marker = "o" if xlabel == "Epoch" else None
        lw = 2.0 if xlabel == "Epoch" else 1.2
        ax.plot(xs, ys, color=color, linewidth=lw, marker=marker, markersize=5, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

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


# ── signal handler ────────────────────────────────────────────────────────

def _install_signal_handlers(trainer: Trainer) -> None:
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


# ── custom Trainer subclass for length-sorted batching ────────────────────

class DocTagTrainer(Trainer):
    """Extends HF Trainer with optional length-sorted sampling."""

    def __init__(self, *args, length_sorted_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._length_sorted_sampler = length_sorted_sampler

    def _get_train_sampler(self, *args, **kwargs):
        if self._length_sorted_sampler is not None:
            return self._length_sorted_sampler
        return super()._get_train_sampler(*args, **kwargs)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimised fine-tuning for ibm-granite/granite-docling-258M. "
            "Drop-in replacement for train.py with better defaults."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── data ──
    parser.add_argument("--train_data", default="training_data_train.jsonl")
    parser.add_argument("--val_data", default="training_data_val.jsonl",
                        help="Validation JSONL; skipped if the file does not exist")

    # ── model ──
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", default="doctag_model")

    # ── augmentation ──
    parser.add_argument("--augment", action="store_true",
                        help="Enable image augmentation on training data")
    parser.add_argument("--augment_strength", choices=["light", "medium", "heavy"],
                        default="medium",
                        help="Augmentation intensity preset (default: medium)")

    # ── training hyper-parameters ──
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps → effective batch = batch_size × grad_accum (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (default: 3e-4, tuned for LoRA)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="Warmup fraction (default: 0.06)")
    parser.add_argument("--lr_scheduler", default="cosine_with_restarts",
                        choices=[
                            "linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup",
                        ],
                        help="LR scheduler (default: cosine_with_restarts)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max token length (default: 4096)")

    # ── precision & speed ──
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true",
                        help="bfloat16 mixed precision (recommended)")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 on Ampere+ GPUs for extra speed")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Flash Attention 2 (requires flash-attn package)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model for faster training (PyTorch ≥ 2.1)")
    parser.add_argument("--dataloader_workers", type=int, default=2,
                        help="Number of data-loader workers (default: 2)")

    # ── LoRA / fine-tuning mode ──
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA (parameter-efficient fine-tuning)")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank (default: 32 — higher than train.py's 8)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help=("Comma-separated list of modules to apply LoRA to. "
                              "Default: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"))
    parser.add_argument("--use_rslora", action="store_true", default=True,
                        help="Use rsLoRA scaling (default: True)")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Resume from an existing LoRA adapter directory")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Freeze vision encoder parameters")

    # ── eval ──
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Evaluate every N steps (default: once per epoch)")
    parser.add_argument("--logging_steps", type=int, default=10)

    # ── saving ──
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: each epoch)")

    # ── batching ──
    parser.add_argument("--length_sorted", action="store_true", default=True,
                        help="Sort training batches by sequence length (default: True)")
    parser.add_argument("--bucket_size", type=int, default=64,
                        help="Bucket size for length-sorted batching (default: 64)")

    args = parser.parse_args()

    # ── validate ──────────────────────────────────────────────────────
    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    if args.flash_attn and not (args.bf16 or args.fp16):
        print("WARNING: Flash Attention 2 requires --bf16 or --fp16.")

    # ── TF32 speed-up on Ampere+ ─────────────────────────────────────
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matmul and cuDNN")

    # ── model & processor ────────────────────────────────────────────
    print(f"Loading processor and model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    torch_dtype = (
        torch.bfloat16 if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    model.config.use_cache = False

    # ── freeze vision encoder ────────────────────────────────────────
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vision" in name or "image_model" in name:
                param.requires_grad = False
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Vision encoder frozen: {frozen:,} params  |  Trainable: {trainable:,} params")

    # ── LoRA ─────────────────────────────────────────────────────────
    if args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA.  pip install peft"
            ) from exc

        if args.lora_adapter_path:
            adapter_path = Path(args.lora_adapter_path)
            config_file = adapter_path / "adapter_config.json"
            if not config_file.exists() or config_file.stat().st_size == 0:
                raise FileNotFoundError(
                    f"LoRA adapter not found at {args.lora_adapter_path}"
                )
            print(f"Loading existing LoRA adapter from: {args.lora_adapter_path}")
            model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
            model.print_trainable_parameters()
        else:
            target_modules = (
                args.lora_target_modules.split(",")
                if args.lora_target_modules
                else ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                use_rslora=args.use_rslora,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    # ── torch.compile ────────────────────────────────────────────────
    if args.compile:
        print("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── datasets ─────────────────────────────────────────────────────
    print("Building datasets …")
    augment_cfg = _AUG_PRESETS[args.augment_strength] if args.augment else None

    train_dataset = DocTagDatasetV2(
        args.train_data, processor, args.max_length,
        augment_cfg=augment_cfg,
    )

    val_dataset = None
    if Path(args.val_data).exists():
        # No augmentation on validation
        val_dataset = DocTagDatasetV2(
            args.val_data, processor, args.max_length,
            augment_cfg=None,
        )
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples | No validation set")

    if augment_cfg:
        print(f"Augmentation: {args.augment_strength} preset")

    # ── length-sorted sampler ────────────────────────────────────────
    length_sampler = None
    if args.length_sorted:
        length_sampler = LengthSortedSampler(
            train_dataset.approx_lengths,
            bucket_size=args.bucket_size,
        )
        print(f"Using length-sorted batching (bucket_size={args.bucket_size})")

    # ── eval strategy ────────────────────────────────────────────────
    if val_dataset is not None:
        if args.eval_steps:
            eval_strategy = "steps"
            eval_steps = args.eval_steps
        else:
            eval_strategy = "epoch"
            eval_steps = None
    else:
        eval_strategy = "no"
        eval_steps = None

    # ── save strategy ────────────────────────────────────────────────
    if args.save_steps:
        save_strategy = "steps"
        save_steps = args.save_steps
    else:
        save_strategy = "epoch"
        save_steps = None

    # ── training arguments ───────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        report_to="none",
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_accumulation_steps=1,
        label_names=["labels"],
        # Log gradient norms for diagnostics
        logging_nan_inf_filter=True,
    )

    metrics_callback = TrainingMetricsCallback(output_dir=args.output_dir)

    trainer = DocTagTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
        compute_metrics=make_compute_metrics(processor.tokenizer) if val_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset is not None else None,
        callbacks=[EvalMemoryCallback(), metrics_callback],
        length_sorted_sampler=length_sampler,
    )

    _install_signal_handlers(trainer)

    # ── print training plan ──────────────────────────────────────────
    total_steps = (
        math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
        * args.epochs
    )
    print(f"\n{'=' * 60}")
    print(f"Training plan (train_v2)")
    print(f"{'=' * 60}")
    print(f"  Samples ........... {len(train_dataset)}")
    print(f"  Epochs ............ {args.epochs}")
    print(f"  Effective batch ... {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  ~Total steps ...... {total_steps}")
    print(f"  Peak LR ........... {args.lr}")
    print(f"  Scheduler ......... {args.lr_scheduler}")
    print(f"  Weight decay ...... {args.weight_decay}")
    print(f"  Label smoothing ... {args.label_smoothing}")
    print(f"  Max grad norm ..... {args.max_grad_norm}")
    print(f"  Augmentation ...... {'ON (' + args.augment_strength + ')' if args.augment else 'OFF'}")
    if args.lora:
        print(f"  LoRA rank ......... {args.lora_r}")
        print(f"  LoRA alpha ........ {args.lora_alpha}")
        print(f"  rsLoRA ............ {args.use_rslora}")
    print(f"{'=' * 60}\n")

    # ── train ─────────────────────────────────────────────────────────
    print("Starting training …")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted — saving emergency checkpoint …")
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        trainer.save_model(emergency_dir)
        processor.save_pretrained(emergency_dir)
        print(f"Emergency checkpoint saved to {emergency_dir}")
        raise

    # ── save ──────────────────────────────────────────────────────────
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    best_model_dir = os.path.join(args.output_dir, "best_model")
    print(f"Saving best model to {best_model_dir}")
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)

    print("Done.")


if __name__ == "__main__":
    main()
