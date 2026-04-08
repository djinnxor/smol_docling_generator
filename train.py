"""
Usage:
    python train.py --train_data training_data_train.jsonl \\
                    --val_data   training_data_val.jsonl   \\
                    --lora                                 \\
                    --bf16                                 \\
                    --output_dir doctag_model

    # Resume from a specific checkpoint:
    python train.py ... --resume_from_checkpoint doctag_model/checkpoint-500

    # Resume from the latest checkpoint automatically:
    python train.py ... --resume_from_checkpoint last --output_dir doctag_model
"""

import argparse
import datetime
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import re
import signal
import sys
from pathlib import Path

# Patch JSON encoder to handle numpy dtypes (compatibility fix for older
# transformers versions that store numpy.dtype objects in model configs).
_json_default_orig = json.JSONEncoder.default

def _json_default_patched(self, obj):
    # Handle dtype objects from numpy, torch, or any other library
    if type(obj).__name__ == "dtype" or hasattr(obj, "itemsize"):
        return str(obj)
    return _json_default_orig(self, obj)

json.JSONEncoder.default = _json_default_patched

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Default model to fine-tune
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"

SYSTEM_PROMPT = "Convert this document page to DocTags format."

# Maximum loss value used when clamping before computing perplexity (avoids
# math.exp overflow for very large loss values early in training).
_MAX_LOSS_FOR_PERPLEXITY = 20

# DocTag structural tags used for tag-accuracy and tree-edit-distance metrics.
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
            return 1 + sum(_size(c) for c in n.children)

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


import random as _random
from dataclasses import dataclass, field


@dataclass
class AugmentConfig:
    """Controls which augmentations are active and their intensities.

    All probabilities (p_*) are per-sample and independent.
    Disable individual augmentations by setting their probability to 0.
    """
    p_gaussian_noise: float = 0.3   # probability of adding Gaussian noise
    noise_std: float = 10.0         # noise standard deviation (0-255 scale)

    p_blur: float = 0.2             # probability of Gaussian blur
    blur_radius: float = 1.2        # blur kernel radius in pixels

    p_jpeg: float = 0.3             # probability of JPEG compression artifact
    jpeg_quality_min: int = 40      # lowest JPEG quality (more artifacts)
    jpeg_quality_max: int = 80      # highest JPEG quality (fewer artifacts)

    p_color_jitter: float = 0.4     # probability of color jitter
    brightness: float = 0.3        # max brightness delta (factor in [1-d, 1+d])
    contrast: float = 0.3          # max contrast delta
    saturation: float = 0.4        # max saturation delta
    hue: float = 0.03              # max hue delta (small – documents aren't colourful)

    p_grayscale: float = 0.15       # probability of converting to grayscale
    p_sharpen: float = 0.2          # probability of unsharp-mask sharpening
    sharpen_factor: float = 1.5     # sharpening strength

    p_random_erase: float = 0.15    # probability of random rectangle erasure
    erase_max_area: float = 0.04    # maximum erased area as fraction of image


def augment_document_image(image: "Image.Image", cfg: AugmentConfig) -> "Image.Image":
    """Apply document-safe augmentations to a PIL RGB image.

    Augmentations are applied independently and stochastically, so each call
    produces a (potentially) different result.  The spatial content and aspect
    ratio are never modified.
    """
    import io
    import numpy as np
    from PIL import ImageFilter, ImageEnhance

    # Work on a copy so the original is unchanged
    img = image.copy()

    # 1. Gaussian noise -------------------------------------------------------
    if _random.random() < cfg.p_gaussian_noise:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, cfg.noise_std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 2. Gaussian blur --------------------------------------------------------
    if _random.random() < cfg.p_blur:
        radius = _random.uniform(0.5, cfg.blur_radius)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 3. JPEG compression artifact --------------------------------------------
    if _random.random() < cfg.p_jpeg:
        quality = _random.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # 4. Color jitter (brightness / contrast / saturation / hue) -------------
    if _random.random() < cfg.p_color_jitter:
        if cfg.brightness > 0:
            factor = _random.uniform(1 - cfg.brightness, 1 + cfg.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if cfg.contrast > 0:
            factor = _random.uniform(1 - cfg.contrast, 1 + cfg.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if cfg.saturation > 0:
            factor = _random.uniform(1 - cfg.saturation, 1 + cfg.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        if cfg.hue > 0:
            # PIL has no direct hue control; use HSV via numpy
            arr = np.array(img, dtype=np.float32) / 255.0
            # Shift hue by adding a small constant in the red channel difference
            hue_shift = _random.uniform(-cfg.hue, cfg.hue) * 255
            arr[:, :, 0] = np.clip(arr[:, :, 0] * 255 + hue_shift, 0, 255) / 255
            img = Image.fromarray((arr * 255).astype(np.uint8))

    # 5. Grayscale ------------------------------------------------------------
    if _random.random() < cfg.p_grayscale:
        img = img.convert("L").convert("RGB")

    # 6. Sharpening -----------------------------------------------------------
    if _random.random() < cfg.p_sharpen:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(cfg.sharpen_factor * 100), threshold=3
        ))

    # 7. Random rectangle erasure ---------------------------------------------
    if _random.random() < cfg.p_random_erase:
        w, h = img.size
        max_area = cfg.erase_max_area * w * h
        ew = _random.randint(1, max(1, int((max_area ** 0.5))))
        eh = _random.randint(1, max(1, int(max_area / max(ew, 1))))
        ew, eh = min(ew, w), min(eh, h)
        x0 = _random.randint(0, w - ew)
        y0 = _random.randint(0, h - eh)
        arr = np.array(img)
        arr[y0:y0 + eh, x0:x0 + ew] = 0  # black rectangle
        img = Image.fromarray(arr)

    return img

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

    def __init__(self, jsonl_path: str, processor, max_length: int = 1024,
                 augment_cfg: "AugmentConfig | None" = None):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length
        self.augment_cfg = augment_cfg  # None = no augmentation (used for val)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        doctag = record["doctag"]
        # Idefics processors count literal "<image>" tokens in text as image placeholders.
        # Some datasets use <image>...</image> as a structural tag, so normalise it
        # to <picture>...</picture> (DocTag-compatible) before tokenisation.
        doctag = doctag.replace("<image>", "<picture>").replace("</image>", "</picture>")
        if self.augment_cfg is not None:
            image = augment_document_image(image, self.augment_cfg)
        # --- build full conversation (prompt + response) ---
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
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        # Full pass: include the image so pixel_values are produced correctly
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
            # Fallback: if extra <image> placeholders remain in text, repeat the same
            # page image so processor receives a matching image count.
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

        # Build labels: mask all prompt tokens with -100
        labels = inputs["input_ids"].clone()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100

        inputs["labels"] = labels
        return inputs


def preprocess_logits_for_metrics(logits, labels):
    """
    Called on GPU before logits are moved to CPU.
    Converts full logit tensors (batch x seq x vocab) → predicted token IDs (batch x seq)
    to avoid accumulating gigabytes of logit data in CPU memory during evaluation.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


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


# ---------------------------------------------------------------------------
# Training metrics callback
# ---------------------------------------------------------------------------

class EvalMemoryCallback(TrainerCallback):
    """
    Clears GPU cache before each evaluation pass and toggles use_cache so
    the model is memory-efficient during teacher-forced eval forward passes
    (gradient checkpointing is off during eval, so use_cache is safe there).
    """

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model is not None and hasattr(model, "config"):
            model.config.use_cache = True

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Re-disable use_cache after eval is done (Trainer calls this then
        # on_evaluate_end; we restore in on_evaluate_end instead).
        pass

    def on_evaluate_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is not None and hasattr(model, "config"):
            model.config.use_cache = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TrainingMetricsCallback(TrainerCallback):
    """
    Callback that:

    * Clearly labels every log line as *Train loss* or *Val loss* so the two
      are easy to distinguish in the console output.
    * Computes and displays perplexity alongside the loss.
    * Persists the full training history as ``training_history.json`` inside
      the output directory.
    * Generates training-curve plots (loss + perplexity) at the end of
      training.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.history: dict[str, list] = {
            "train_steps": [],
            "train_loss": [],
            "train_perplexity": [],
            "train_lr": [],
            "train_grad_norm": [],
            "eval_epochs": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "eval_token_acc": [],
            "eval_tag_acc": [],
            "eval_loc_mae": [],
            "eval_tree_edit_dist": [],
        }

    # ------------------------------------------------------------------
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not logs:
            return

        step = state.global_step
        epoch = state.epoch or 0.0

        if "loss" in logs and "eval_loss" not in logs:
            # ---- training step ----
            train_loss = logs["loss"]
            perplexity = math.exp(min(train_loss, _MAX_LOSS_FOR_PERPLEXITY))  # clamp to avoid overflow
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
            # ---- validation epoch ----
            eval_loss = logs["eval_loss"]
            perplexity = math.exp(min(eval_loss, _MAX_LOSS_FOR_PERPLEXITY))
            token_acc = logs.get("eval_token_acc")
            tag_acc = logs.get("eval_tag_acc")
            loc_mae = logs.get("eval_loc_mae")
            tree_edit = logs.get("eval_tree_edit_dist")
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
                f"[Epoch {epoch:.1f}]                "
                f"Val loss:   {eval_loss:.4f}  |  "
                f"Perplexity: {perplexity:.2f}{metrics_str}"
            )

    # ------------------------------------------------------------------
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        print(f"Training history saved to {history_path}")
        _plot_training_curves(self.history, self.output_dir)


# ---------------------------------------------------------------------------
# Training-curve visualisation
# ---------------------------------------------------------------------------

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
# Mid-epoch safety checkpoint
# ---------------------------------------------------------------------------

class MidEpochCheckpointCallback(TrainerCallback):
    """Saves a rolling checkpoint every `save_steps` optimizer steps.

    This ensures that if training stops unexpectedly (dataset error, OOM,
    hardware failure) before the end of an epoch, the last N steps of work
    are not lost.  Only one checkpoint is kept at a time (overwritten each
    interval) to avoid filling disk.
    """

    def __init__(self, save_steps: int, output_dir: str) -> None:
        self.save_steps = save_steps
        self.checkpoint_dir = os.path.join(output_dir, "mid_epoch_checkpoint")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            trainer = kwargs.get("model")  # not available here; use control
            control.should_save = False    # don't trigger the default save logic
            try:
                # Access trainer via the callback's registered trainer reference
                _trainer = getattr(self, "_trainer", None)
                if _trainer is not None:
                    _trainer.save_model(self.checkpoint_dir)
                    print(
                        f"[Step {state.global_step}] Mid-epoch checkpoint saved → "
                        f"{self.checkpoint_dir}"
                    )
            except Exception as exc:
                print(f"[WARNING] Could not save mid-epoch checkpoint: {exc}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Capture the trainer reference when training starts
        self._trainer = kwargs.get("trainer")


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


# ---------------------------------------------------------------------------

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
    parser.add_argument("--epochs", type=int, default=7, help="Number of training epochs (default: 6)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 1)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum token length (default: 4096)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (recommended)")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2 (requires flash-attn and bf16/fp16)")
    parser.add_argument("--freeze_vision", action="store_true", help="Freeze vision encoder parameters (only train the language decoder)")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning (requires peft)",
    )
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (default: 64)")
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
        help="Path to a custom LoRA adapter directory (must contain adapter_config.json). Overrides lora_r, lora_alpha, lora_dropout.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help=(
            "Maximum number of epoch checkpoints to keep inside --output_dir. "
            "Oldest checkpoints are deleted when the limit is exceeded. "
            "Default: None (keep all)."
        ),
    )
    parser.add_argument(
        "--mid_epoch_save_steps",
        type=int,
        default=200,
        help=(
            "Save a rolling mid-epoch checkpoint every N optimizer steps. "
            "Protects against unexpected stops before an epoch completes. "
            "Saved to <output_dir>/mid_epoch_checkpoint (overwritten each time). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging for hyperparameters and metrics.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Docling fine tuning",
        help="W&B project name (used when --use_wandb is set).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Optional W&B run name. Auto-generated if empty.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated W&B tags, e.g. 'craft,lora,bf16'.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (online/offline/disabled).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable on-the-fly image augmentation for training samples. "
            "Safe ops only (noise, blur, JPEG artefacts, colour jitter, "
            "grayscale, sharpen, random erase). "
            "Validation images are never augmented."
        ),
    )
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
    if args.flash_attn and torch_dtype == torch.float32:
        print("WARNING: Flash Attention 2 requires float16 or bfloat16. Pass --bf16 or --fp16.")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    # Gradient checkpointing (enabled below) is incompatible with use_cache.
    # Set it explicitly here so the Trainer does not emit a warning and the
    # model behaves consistently during both training and evaluation.
    model.config.use_cache = False

    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vision" in name or "image_model" in name:
                param.requires_grad = False
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Vision encoder frozen: {frozen:,} params  |  Trainable: {trainable:,} params")

    if args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning. Install it with:\n"
                "  pip install peft"
            ) from exc

        if args.lora_adapter_path:
            import json as _json
            from pathlib import Path as _Path
            adapter_path = _Path(args.lora_adapter_path)
            config_file = adapter_path / "adapter_config.json"
            if not config_file.exists() or config_file.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Custom LoRA adapter not found or adapter_config.json is empty at {args.lora_adapter_path}"
                )
            with open(config_file, encoding="utf-8") as f:
                adapter_cfg = _json.load(f)
            print(f"Loading custom LoRA adapter from: {args.lora_adapter_path}")
            print(f"  base model: {adapter_cfg['base_model_name_or_path']}")
            model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
            model.print_trainable_parameters()
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    print("Building datasets …")
    augment_cfg = AugmentConfig() if args.augment else None
    if augment_cfg is not None:
        print("Image augmentation enabled for training set.")
    train_dataset = DocTagDataset(args.train_data, processor, args.max_length,
                                  augment_cfg=augment_cfg)

    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(args.val_data, processor, args.max_length,
                                    augment_cfg=None)  # never augment val
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples | No validation file found at {args.val_data}")

    wandb_enabled = args.use_wandb and args.wandb_mode != "disabled"
    run_name = None
    if wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed in this environment. Install it with:\n"
                "  pip install wandb"
            ) from exc

        auto_name = (
            f"doctag-{Path(args.train_data).stem}-"
            f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        run_name = args.wandb_run_name.strip() or auto_name
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            tags=tags or None,
        )
        wandb.config.update(
            {
                "model_id": args.model_id,
                "train_data": args.train_data,
                "val_data": args.val_data,
                "output_dir": args.output_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "max_length": args.max_length,
                "fp16": args.fp16,
                "bf16": args.bf16,
                "flash_attn": args.flash_attn,
                "freeze_vision": args.freeze_vision,
                "lora": args.lora,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "save_total_limit": args.save_total_limit,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset) if val_dataset is not None else 0,
            },
            allow_val_change=True,
        )

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
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        # Keep all per-epoch checkpoints by default so no work is ever lost.
        # Pass --save_total_limit N to cap the number of saved checkpoints.
        save_total_limit=args.save_total_limit,
        # Load the best checkpoint at the end when validation is available.
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=["wandb"] if wandb_enabled else "none",
        run_name=run_name,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Move each batch's predictions to CPU immediately to avoid VRAM accumulation.
        eval_accumulation_steps=1,
        # PeftModel hides the base model's signature, so Trainer can't infer
        # label_names automatically.  Specify them explicitly to silence the warning.
        label_names=["labels"],
    )

    metrics_callback = TrainingMetricsCallback(output_dir=args.output_dir)

    callbacks = [EvalMemoryCallback(), metrics_callback]
    if args.mid_epoch_save_steps > 0:
        mid_epoch_cb = MidEpochCheckpointCallback(
            save_steps=args.mid_epoch_save_steps,
            output_dir=args.output_dir,
        )
        callbacks.append(mid_epoch_cb)
    else:
        mid_epoch_cb = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
        compute_metrics=make_compute_metrics(processor.tokenizer) if val_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset is not None else None,
        callbacks=callbacks,
    )

    # Give the mid-epoch callback a reference to the trainer so it can call save_model
    if mid_epoch_cb is not None:
        mid_epoch_cb._trainer = trainer

    # Install signal handlers (closure captures trainer, no global needed).
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
    finally:
        if wandb_enabled:
            try:
                import wandb

                history_path = os.path.join(args.output_dir, "training_history.json")
                if os.path.exists(history_path):
                    wandb.save(history_path, policy="now")

                eval_losses = metrics_callback.history.get("eval_loss", [])
                if eval_losses:
                    wandb.summary["best_eval_loss"] = min(eval_losses)

                train_losses = metrics_callback.history.get("train_loss", [])
                if train_losses:
                    wandb.summary["last_train_loss"] = train_losses[-1]

                wandb.finish()
            except Exception as exc:  # noqa: BLE001
                print(f"W&B finalization warning: {exc}")

    # Save the best (or final) model to the requested output directory.
    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Also write the best model to a dedicated sub-directory so it is always
    # easy to locate regardless of checkpoint rotation.
    best_model_dir = os.path.join(args.output_dir, "best_model")
    print(f"Saving best model to {best_model_dir}")
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)

    print("Done.")


if __name__ == "__main__":
    main()