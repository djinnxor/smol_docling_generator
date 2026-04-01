"""
Custom-loop fine-tuning for ibm-granite/granite-docling-258M (DocTag generation).

This is a *fundamentally different* approach from ``train.py`` (HF Trainer)
and ``train_v2.py`` (HF Trainer with better hyper-parameters).  It replaces
the Trainer abstraction entirely with a transparent PyTorch training loop
that adds several techniques specifically designed for vision-to-structured-
output tasks:

Why the standard approach fails
-------------------------------
1. Standard cross-entropy treats every token equally.  In DocTag output,
   ~80 % of tokens are content text (Mongolian characters, numbers) while
   the *critical* structural tokens (<doctag>, <text>, <otsl>, <loc_N>)
   make up only ~20 %.  The model optimises for average loss → memorises
   text patterns but never learns document structure.

2. Teacher-forced evaluation (argmax comparison) does not reflect real
   generation quality.  A model can have 90 % token accuracy but generate
   garbage autoregressively because errors compound.

3. Vision encoder and language decoder have very different convergence
   dynamics.  Training them with the same LR from step 0 often means the
   vision encoder diverges before the decoder learns anything.

Key techniques in this script
-----------------------------
* **Weighted token loss** — structural and location tokens get 3–5×
  higher weight in the cross-entropy loss so the model prioritises
  learning document layout over content parroting.

* **Focal loss option** — down-weights easy/confident predictions,
  focuses gradient on hard-to-predict tokens.  Particularly useful when
  the token distribution is imbalanced.

* **Two-phase training** — Phase 1 freezes the vision encoder and
  trains only the language decoder (+ LoRA) with a higher LR to learn
  DocTag grammar.  Phase 2 unfreezes the vision encoder with a 10×
  lower LR (discriminative rates) to align visual features.

* **NEFTune noise** — injects small uniform noise into the embedding
  layer during training.  Empirically improves generation quality with
  negligible overhead (Jain et al., 2024).

* **EMA model** — maintains an exponential moving average of model
  weights.  The EMA copy generalises better and is used for validation
  and final saving.

* **Generative validation** — periodically calls model.generate() on
  a small validation subset and computes edit distance against ground
  truth.  This directly measures inference quality, not teacher-forced
  accuracy.

* **OneCycleLR scheduler** — converges faster than cosine variants
  because it uses a super-convergence warmup-then-anneal pattern.

Compatibility
-------------
* Same JSONL dataset format: ``{"image": "...", "doctag": "..."}``
* Same ``inference.py`` / ``build_document.py`` pipeline
* Same LoRA adapter format (loadable by inference.py)
* Augmentation presets from train_v2.py are reused

Usage
-----
    # Recommended: two-phase with weighted loss and augmentation
    python train_v3.py --train_data training_data_train.jsonl \\
                       --val_data   training_data_val.jsonl   \\
                       --lora --bf16 --augment --two_phase

    # Focal loss (good for imbalanced token distributions)
    python train_v3.py --train_data training_data_train.jsonl \\
                       --lora --bf16 --augment --focal_loss

    # Quick single-phase with EMA
    python train_v3.py --train_data training_data_train.jsonl \\
                       --lora --bf16 --augment --ema
"""

import argparse
import copy
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import re
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── defaults ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"
SYSTEM_PROMPT = "Convert this document page to DocTags format."

# DocTag structural tags
_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
})

# ── metrics (compatible with train.py / train_v2.py) ─────────────────────


def _extract_structural_tags(s: str) -> list[str]:
    return [
        ("/" if slash else "") + tag
        for slash, tag in re.findall(r"<(/?)(\w+(?:_\w+)*)(?:[^>]*)?>"  , s)
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
    pl = [int(n) for n in re.findall(r"<loc_(\d+)>", pred)]
    rl = [int(n) for n in re.findall(r"<loc_(\d+)>", ref)]
    if not rl:
        return 0.0
    total = sum(
        abs(pl[i] - rl[i]) if i < len(pl) else rl[i] for i in range(len(rl))
    )
    return total / len(rl)


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
            for slash, tag in re.findall(r"<(/?)(\w+(?:_\w+)*)(?:[^>]*)?>"  , s):
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
        mx = _size(pt) + _size(rt)
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


# ── data augmentation (shared with train_v2.py) ──────────────────────────

_AUG_PRESETS: dict[str, dict] = {
    "light": {
        "prob": 0.3, "brightness": 0.1, "contrast": 0.1, "saturation": 0.1,
        "hue": 0.02, "rotation": 2, "blur_radius": 0.5, "perspective": 0.02,
        "erasing_prob": 0.05, "noise_std": 5,
    },
    "medium": {
        "prob": 0.5, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2,
        "hue": 0.03, "rotation": 3, "blur_radius": 1.0, "perspective": 0.04,
        "erasing_prob": 0.10, "noise_std": 10,
    },
    "heavy": {
        "prob": 0.7, "brightness": 0.35, "contrast": 0.35, "saturation": 0.3,
        "hue": 0.05, "rotation": 5, "blur_radius": 1.5, "perspective": 0.06,
        "erasing_prob": 0.15, "noise_std": 15,
    },
}


def _augment_image(image: Image.Image, cfg: dict) -> Image.Image:
    """Document-aware augmentation pipeline (no flips / heavy crops)."""
    if random.random() > cfg["prob"]:
        return image

    if random.random() < 0.5:
        f = 1.0 + random.uniform(-cfg["brightness"], cfg["brightness"])
        image = ImageEnhance.Brightness(image).enhance(f)
    if random.random() < 0.5:
        f = 1.0 + random.uniform(-cfg["contrast"], cfg["contrast"])
        image = ImageEnhance.Contrast(image).enhance(f)
    if random.random() < 0.3:
        f = 1.0 + random.uniform(-cfg["saturation"], cfg["saturation"])
        image = ImageEnhance.Color(image).enhance(f)
    if random.random() < 0.2:
        r = random.uniform(0, cfg["blur_radius"])
        image = image.filter(ImageFilter.GaussianBlur(radius=r))
    if random.random() < 0.2:
        arr = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, cfg["noise_std"], arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    if random.random() < 0.3:
        angle = random.uniform(-cfg["rotation"], cfg["rotation"])
        image = image.rotate(
            angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255)
        )
    if random.random() < 0.15:
        w, h = image.size
        d = cfg["perspective"]
        dx, dy = int(w * d), int(h * d)
        if dx > 0 and dy > 0:
            coeffs = _perspective_coefficients(
                [(0, 0), (w, 0), (w, h), (0, h)],
                [
                    (random.randint(0, dx), random.randint(0, dy)),
                    (w - random.randint(0, dx), random.randint(0, dy)),
                    (w - random.randint(0, dx), h - random.randint(0, dy)),
                    (random.randint(0, dx), h - random.randint(0, dy)),
                ],
            )
            image = image.transform(
                image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR,
                fillcolor=(255, 255, 255),
            )
    if random.random() < cfg["erasing_prob"]:
        arr = np.array(image)
        eh = random.randint(2, max(3, arr.shape[0] // 40))
        ew = random.randint(2, max(3, arr.shape[1] // 40))
        ey = random.randint(0, arr.shape[0] - eh)
        ex = random.randint(0, arr.shape[1] - ew)
        arr[ey : ey + eh, ex : ex + ew] = random.randint(200, 255)
        image = Image.fromarray(arr)
    return image


def _perspective_coefficients(src_pts, dst_pts):
    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for pt in src_pts for c in pt], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


# ── token-weight computation ─────────────────────────────────────────────


def build_token_weights(
    tokenizer,
    struct_weight: float = 5.0,
    loc_weight: float = 3.0,
    table_weight: float = 2.0,
) -> torch.Tensor:
    """Build per-token weights for the entire vocabulary.

    Tokens that encode document *structure* (element tags, location markers,
    table cell markers) receive higher weights than plain content tokens.
    This forces the model to prioritise learning the DocTag schema.

    Returns a 1-D float tensor of shape ``(vocab_size,)`` that can be passed
    to ``F.cross_entropy(weight=...)``.
    """
    vocab_size = len(tokenizer)
    weights = torch.ones(vocab_size, dtype=torch.float32)

    # Build patterns to match
    struct_patterns = set()
    for tag in _STRUCTURAL_TAGS:
        struct_patterns.add(f"<{tag}>")
        struct_patterns.add(f"</{tag}>")

    table_tokens = {"<ched>", "<rhed>", "<fcel>", "<ucel>", "<lcel>", "<xcel>",
                    "<ecel>", "<nl>"}

    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
        except Exception:
            continue

        # Structural element tags → highest weight
        if any(pat in token_str for pat in struct_patterns):
            weights[token_id] = struct_weight
        # Location tokens <loc_0> through <loc_500>
        elif re.search(r"<loc_\d+>", token_str):
            weights[token_id] = loc_weight
        # Table cell tokens
        elif any(t in token_str for t in table_tokens):
            weights[token_id] = table_weight

    return weights


# ── focal loss ────────────────────────────────────────────────────────────


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with focal modulation for hard-example mining.

    Focal loss reduces the contribution of easy (high-confidence) predictions
    and focuses the gradient on hard tokens.  Combined with per-token weights
    it is very effective for structured-output tasks.

    Parameters
    ----------
    logits : (B, S, V)
    labels : (B, S)
    gamma  : focal exponent (0 = standard CE, 2 = recommended)
    weight : (V,) per-token class weights
    label_smoothing : smoothing factor for soft labels
    ignore_index : label value to ignore (prompt tokens)
    """
    B, S, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)

    # Standard CE per token (unreduced)
    ce = F.cross_entropy(
        logits_flat, labels_flat,
        weight=weight,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    if gamma > 0:
        # Focal modulation: (1 - p_t)^gamma
        with torch.no_grad():
            log_probs = F.log_softmax(logits_flat, dim=-1)
            # Gather the log-prob of the correct token
            valid = labels_flat != ignore_index
            safe_labels = labels_flat.clone()
            safe_labels[~valid] = 0
            log_pt = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            focal_mod = (1.0 - pt) ** gamma
            focal_mod[~valid] = 0.0

        ce = focal_mod * ce

    # Mean over non-ignored tokens
    mask = labels_flat != ignore_index
    if mask.any():
        return ce[mask].mean()
    return ce.mean()


# ── NEFTune noise injection ──────────────────────────────────────────────


class NEFTuneHook:
    """Injects uniform noise into the embedding output during training.

    Implements the NEFTune technique (Jain et al., 2024) which has been
    shown to improve fine-tuning quality for language models.  The noise
    magnitude is scaled by ``alpha / sqrt(seq_len * hidden_dim)`` so it
    adapts to the sequence geometry.
    """

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha
        self._handle = None

    def _hook(self, module, input, output):
        if self.alpha > 0 and module.training:
            dims = output.shape[-1] * output.shape[-2]
            mag = self.alpha / (dims ** 0.5)
            noise = torch.zeros_like(output).uniform_(-mag, mag)
            return output + noise
        return output

    def register(self, model):
        """Find and hook the text embedding layer."""
        embed = None
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding) and "embed_tokens" in name:
                embed = mod
                break
        if embed is None:
            # Fallback: first Embedding
            for mod in model.modules():
                if isinstance(mod, torch.nn.Embedding):
                    embed = mod
                    break
        if embed is not None:
            self._handle = embed.register_forward_hook(self._hook)
            return True
        return False

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ── EMA (Exponential Moving Average) ─────────────────────────────────────


class EMAModel:
    """Maintains an exponential moving average of model parameters.

    The EMA model generalises better than the raw training model because
    it smooths out noisy gradient updates.  It is used for validation and
    as the final saved model.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: torch.nn.Module):
        """Swap model params with EMA params (for eval/save)."""
        self.backup.clear()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original params after eval."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ── dataset ───────────────────────────────────────────────────────────────


class DocTagDatasetV3(Dataset):
    """Dataset with augmentation and pre-computed token weights info."""

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
        self.augment_cfg = augment_cfg

        # Cache prompt template
        prompt_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        self._prompt_text = processor.apply_chat_template(
            prompt_msgs, add_generation_prompt=True
        )
        self._prompt_ids = processor.tokenizer(
            self._prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).input_ids

        # Approx lengths for sorting
        self.approx_lengths: list[int] = [len(r["doctag"]) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        doctag = record["doctag"]

        if self.augment_cfg is not None:
            image = _augment_image(image, self.augment_cfg)

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

        # Label masking
        labels = inputs["input_ids"].clone()
        prompt_len = len(self._prompt_ids)
        labels[:prompt_len] = -100
        inputs["labels"] = labels
        return inputs


# ── collator ──────────────────────────────────────────────────────────────


def collate_fn(features: list[dict]) -> dict:
    """Pad a batch to uniform length, handling variable pixel_values."""
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
                        padding = torch.zeros(
                            pad_len, *t.shape[1:], dtype=t.dtype
                        )
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        else:
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
    return batch


# ── length-sorted sampler ────────────────────────────────────────────────


class BucketSampler(Sampler):
    """Yield indices sorted by length with shuffled buckets."""

    def __init__(self, lengths: list[int], bucket_size: int = 64):
        self.lengths = lengths
        self.bucket_size = max(1, bucket_size)

    def __iter__(self):
        idxs = sorted(range(len(self.lengths)), key=lambda i: -self.lengths[i])
        bucketed: list[int] = []
        for start in range(0, len(idxs), self.bucket_size):
            bucket = idxs[start : start + self.bucket_size]
            random.shuffle(bucket)
            bucketed.extend(bucket)
        return iter(bucketed)

    def __len__(self) -> int:
        return len(self.lengths)


# ── generative validation ────────────────────────────────────────────────


@torch.inference_mode()
def generative_eval(
    model,
    processor,
    val_dataset: DocTagDatasetV3,
    device: str,
    max_new_tokens: int = 1536,
    num_samples: int = 8,
) -> dict:
    """Run actual model.generate() on a val subset and measure quality.

    Returns a dict with generative metrics (edit distance, tag accuracy,
    location MAE) that reflect *real* autoregressive inference quality.
    """
    model.eval()
    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))

    tag_accs, loc_maes, tree_edits = [], [], []
    gen_times = []

    for idx in indices:
        record = val_dataset.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        ref_doctag = record["doctag"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        prompt_text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(
            images=image, text=prompt_text, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image.close()

        t0 = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        gen_times.append(time.time() - t0)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        pred_doctag = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0].strip()

        tag_accs.append(_tag_accuracy(pred_doctag, ref_doctag))
        loc_maes.append(_loc_mae(pred_doctag, ref_doctag))
        tree_edits.append(_tree_edit_dist(pred_doctag, ref_doctag))

    return {
        "gen_tag_acc": float(np.mean(tag_accs)),
        "gen_loc_mae": float(np.mean(loc_maes)),
        "gen_tree_edit": float(np.mean(tree_edits)),
        "gen_time_avg": float(np.mean(gen_times)),
    }


# ── training loop ────────────────────────────────────────────────────────


def train_phase(
    *,
    phase_name: str,
    model,
    processor,
    train_loader: DataLoader,
    val_dataset: DocTagDatasetV3 | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    dtype: torch.dtype,
    epochs: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    token_weights: torch.Tensor | None,
    use_focal: bool,
    focal_gamma: float,
    label_smoothing: float,
    neftune: NEFTuneHook | None,
    ema: EMAModel | None,
    output_dir: str,
    logging_steps: int,
    eval_steps: int | None,
    gen_eval_samples: int,
    save_steps: int | None,
    history: dict,
    global_step: int,
) -> int:
    """Run one training phase (custom loop, no HF Trainer)."""

    scaler = GradScaler(enabled=(dtype == torch.float16))
    use_amp = dtype in (torch.float16, torch.bfloat16)
    amp_dtype = dtype if use_amp else torch.float32

    # Token weights to device
    tw = token_weights.to(device) if token_weights is not None else None

    model.train()
    if neftune is not None:
        neftune.register(model)

    total_steps = len(train_loader) * epochs // grad_accum_steps
    print(f"\n{'─' * 60}")
    print(f"Phase: {phase_name}")
    print(f"  Epochs .......... {epochs}")
    print(f"  Steps ........... ~{total_steps}")
    print(f"  Grad accum ...... {grad_accum_steps}")
    print(f"  Focal loss ...... {'ON (γ=' + str(focal_gamma) + ')' if use_focal else 'OFF'}")
    print(f"  Token weights ... {'ON' if tw is not None else 'OFF'}")
    print(f"  NEFTune ......... {'ON' if neftune is not None else 'OFF'}")
    print(f"  EMA ............. {'ON' if ema is not None else 'OFF'}")
    print(f"{'─' * 60}\n")

    step_in_phase = 0
    optimizer.zero_grad()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            # Forward with mixed precision
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                logits = outputs.logits

                # Custom loss
                if use_focal:
                    loss = focal_cross_entropy(
                        logits, labels,
                        gamma=focal_gamma,
                        weight=tw,
                        label_smoothing=label_smoothing,
                    )
                elif tw is not None:
                    # Weighted CE
                    B, S, V = logits.shape
                    loss = F.cross_entropy(
                        logits.reshape(-1, V),
                        labels.reshape(-1),
                        weight=tw,
                        ignore_index=-100,
                        label_smoothing=label_smoothing,
                    )
                else:
                    # Standard CE
                    B, S, V = logits.shape
                    loss = F.cross_entropy(
                        logits.reshape(-1, V),
                        labels.reshape(-1),
                        ignore_index=-100,
                        label_smoothing=label_smoothing,
                    )

                loss = loss / grad_accum_steps

            # Backward
            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Track loss
            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * grad_accum_steps * n_tokens
            epoch_tokens += n_tokens

            # Optimiser step after accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )

                if dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_in_phase += 1

                # EMA update
                if ema is not None:
                    ema.update(model)

                # Logging
                if step_in_phase % logging_steps == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    ppl = math.exp(min(avg_loss, 20))
                    print(
                        f"  [{phase_name}] Step {global_step:>5} | "
                        f"Epoch {epoch}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {ppl:.2f} | "
                        f"LR: {lr:.2e}"
                    )
                    history["train_steps"].append(global_step)
                    history["train_loss"].append(avg_loss)
                    history["train_lr"].append(lr)

                # Periodic eval
                if eval_steps and val_dataset and step_in_phase % eval_steps == 0:
                    _run_eval(
                        model, processor, val_dataset, device, ema,
                        gen_eval_samples, global_step, epoch, history,
                        phase_name,
                    )
                    model.train()

                # Periodic save
                if save_steps and step_in_phase % save_steps == 0:
                    _save_checkpoint(
                        model, processor, ema, output_dir,
                        f"checkpoint-{global_step}",
                    )

        # End-of-epoch eval
        if val_dataset:
            _run_eval(
                model, processor, val_dataset, device, ema,
                gen_eval_samples, global_step, epoch, history, phase_name,
            )
            model.train()

        # End-of-epoch save
        _save_checkpoint(
            model, processor, ema, output_dir,
            f"checkpoint-{phase_name}-epoch-{epoch}",
        )

        elapsed = time.time() - t_epoch
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        print(
            f"  [{phase_name}] Epoch {epoch} done | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

    if neftune is not None:
        neftune.remove()

    return global_step


def _run_eval(
    model, processor, val_dataset, device, ema,
    gen_eval_samples, global_step, epoch, history, phase_name,
):
    """Run both teacher-forced and generative evaluation."""
    was_training = model.training
    model.eval()

    # Apply EMA weights for eval
    if ema is not None:
        ema.apply(model)

    # Teacher-forced eval (fast: measure loss on val set subset)
    val_loss, val_tokens = 0.0, 0
    n_correct = 0
    indices = random.sample(range(len(val_dataset)), min(100, len(val_dataset)))
    with torch.inference_mode():
        for idx in indices:
            sample = val_dataset[idx]
            sample = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            labels = sample.pop("labels")
            outputs = model(**sample)
            logits = outputs.logits
            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100,
            )
            mask = labels != -100
            n_tokens = mask.sum().item()
            val_loss += loss.item() * n_tokens
            val_tokens += n_tokens
            preds = logits.argmax(dim=-1)
            n_correct += (preds == labels)[mask].sum().item()

    avg_val_loss = val_loss / max(val_tokens, 1)
    token_acc = n_correct / max(val_tokens, 1)
    ppl = math.exp(min(avg_val_loss, 20))

    # Generative eval (slower but measures real quality)
    gen_metrics = {}
    if gen_eval_samples > 0:
        # Need use_cache for generation
        if hasattr(model, "config"):
            model.config.use_cache = True
        elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = True

        gen_metrics = generative_eval(
            model, processor, val_dataset, device,
            num_samples=gen_eval_samples,
        )

        if hasattr(model, "config"):
            model.config.use_cache = False
        elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = False

    # Restore training weights
    if ema is not None:
        ema.restore(model)

    # Log
    gen_str = ""
    if gen_metrics:
        gen_str = (
            f" | GenTagAcc: {gen_metrics['gen_tag_acc']:.4f}"
            f" | GenLocMAE: {gen_metrics['gen_loc_mae']:.2f}"
            f" | GenTreeEdit: {gen_metrics['gen_tree_edit']:.4f}"
        )
    print(
        f"  [{phase_name}] EVAL Step {global_step} Epoch {epoch:.1f} | "
        f"Val Loss: {avg_val_loss:.4f} | PPL: {ppl:.2f} | "
        f"TokenAcc: {token_acc:.4f}{gen_str}"
    )

    history["eval_steps"].append(global_step)
    history["eval_loss"].append(avg_val_loss)
    history["eval_token_acc"].append(token_acc)
    if gen_metrics:
        history.setdefault("gen_tag_acc", []).append(gen_metrics["gen_tag_acc"])
        history.setdefault("gen_loc_mae", []).append(gen_metrics["gen_loc_mae"])
        history.setdefault("gen_tree_edit", []).append(gen_metrics["gen_tree_edit"])

    if was_training:
        model.train()


def _save_checkpoint(model, processor, ema, output_dir, name):
    """Save model checkpoint (with EMA weights if available)."""
    save_dir = os.path.join(output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    if ema is not None:
        ema.apply(model)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    if ema is not None:
        ema.restore(model)

    print(f"  Checkpoint saved: {save_dir}")


# ── plotting ──────────────────────────────────────────────────────────────


def _plot_training_curves(history: dict, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    panels = []
    if history.get("train_steps") and history.get("train_loss"):
        panels.append(("Train Loss", history["train_steps"], history["train_loss"],
                        "steelblue", "Step", "Loss"))
    if history.get("eval_steps") and history.get("eval_loss"):
        panels.append(("Val Loss", history["eval_steps"], history["eval_loss"],
                        "darkorange", "Step", "Loss"))
    if history.get("eval_token_acc"):
        panels.append(("Val Token Acc", history["eval_steps"][:len(history["eval_token_acc"])],
                        history["eval_token_acc"], "mediumvioletred", "Step", "Acc"))
    if history.get("gen_tag_acc"):
        panels.append(("Gen Tag Acc (real inference)", history["eval_steps"][:len(history["gen_tag_acc"])],
                        history["gen_tag_acc"], "darkgreen", "Step", "Acc"))
    if history.get("gen_loc_mae"):
        panels.append(("Gen Loc MAE ↓", history["eval_steps"][:len(history["gen_loc_mae"])],
                        history["gen_loc_mae"], "crimson", "Step", "MAE"))
    if history.get("gen_tree_edit"):
        panels.append(("Gen Tree Edit ↓", history["eval_steps"][:len(history["gen_tree_edit"])],
                        history["gen_tree_edit"], "saddlebrown", "Step", "Edit"))

    if not panels:
        return

    ncols = 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle("Training Progress (train_v3 — custom loop)", fontsize=14, y=1.01)

    for i, (title, xs, ys, color, xlabel, ylabel) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, alpha=0.85)
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


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Custom-loop fine-tuning with weighted loss, two-phase training, "
            "focal loss, NEFTune, and EMA.  Fundamentally different approach "
            "from train.py/train_v2.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument("--train_data", default="training_data_train.jsonl")
    parser.add_argument("--val_data", default="training_data_val.jsonl")

    # Model
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", default="doctag_model")

    # Augmentation
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_strength", choices=["light", "medium", "heavy"],
                        default="medium")

    # --- Core new features ---
    parser.add_argument("--weighted_loss", action="store_true", default=True,
                        help="Upweight structural/location tokens in loss (default: True)")
    parser.add_argument("--struct_weight", type=float, default=5.0,
                        help="Weight for structural tags (default: 5.0)")
    parser.add_argument("--loc_weight", type=float, default=3.0,
                        help="Weight for <loc_N> tokens (default: 3.0)")
    parser.add_argument("--table_weight", type=float, default=2.0,
                        help="Weight for table cell tokens (default: 2.0)")

    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss for hard-example mining")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")

    parser.add_argument("--two_phase", action="store_true",
                        help="Two-phase training: freeze vision → unfreeze with discriminative LR")
    parser.add_argument("--phase1_epochs", type=int, default=3,
                        help="Phase 1 epochs (decoder-only, default: 3)")
    parser.add_argument("--phase2_epochs", type=int, default=2,
                        help="Phase 2 epochs (full model, default: 2)")
    parser.add_argument("--phase2_vision_lr_scale", type=float, default=0.1,
                        help="Phase 2 vision encoder LR = base_lr × this (default: 0.1)")

    parser.add_argument("--neftune", action="store_true",
                        help="NEFTune embedding noise (improves generation quality)")
    parser.add_argument("--neftune_alpha", type=float, default=5.0,
                        help="NEFTune noise magnitude (default: 5.0)")

    parser.add_argument("--ema", action="store_true",
                        help="Exponential Moving Average of model weights")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate (default: 0.999)")

    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Total epochs (single-phase mode, default: 5)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate (default: 2e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=4096)

    # Precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")

    # LoRA
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--freeze_vision", action="store_true")

    # Eval / logging / saving
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Eval every N steps (default: end of epoch)")
    parser.add_argument("--gen_eval_samples", type=int, default=4,
                        help="Number of samples for generative eval (default: 4)")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--dataloader_workers", type=int, default=2)

    args = parser.parse_args()

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    # TF32
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        torch.bfloat16 if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )

    # ── model & processor ────────────────────────────────────────────
    print(f"Loading: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    model.config.use_cache = False

    # Freeze vision (for single-phase mode or phase 1)
    if args.freeze_vision or args.two_phase:
        for name, param in model.named_parameters():
            if "vision" in name or "image_model" in name:
                param.requires_grad = False

    # LoRA
    peft_model = None
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel

        if args.lora_adapter_path:
            model = PeftModel.from_pretrained(
                model, args.lora_adapter_path, is_trainable=True
            )
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
                use_rslora=True,
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        peft_model = model

    model = model.to(device)

    # Gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ── token weights ────────────────────────────────────────────────
    token_weights = None
    if args.weighted_loss:
        print("Building per-token weights for weighted loss …")
        token_weights = build_token_weights(
            processor.tokenizer,
            struct_weight=args.struct_weight,
            loc_weight=args.loc_weight,
            table_weight=args.table_weight,
        )
        n_weighted = (token_weights > 1.0).sum().item()
        print(f"  {n_weighted} tokens upweighted out of {len(processor.tokenizer)}")

    # ── datasets ─────────────────────────────────────────────────────
    print("Building datasets …")
    aug_cfg = _AUG_PRESETS[args.augment_strength] if args.augment else None

    train_dataset = DocTagDatasetV3(
        args.train_data, processor, args.max_length, augment_cfg=aug_cfg,
    )
    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDatasetV3(
            args.val_data, processor, args.max_length, augment_cfg=None,
        )
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    else:
        print(f"Train: {len(train_dataset)} | No validation set")

    # DataLoader
    sampler = BucketSampler(train_dataset.approx_lengths, bucket_size=64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── NEFTune ──────────────────────────────────────────────────────
    nef = NEFTuneHook(alpha=args.neftune_alpha) if args.neftune else None

    # ── training history ─────────────────────────────────────────────
    history: dict[str, list] = {
        "train_steps": [], "train_loss": [], "train_lr": [],
        "eval_steps": [], "eval_loss": [], "eval_token_acc": [],
    }

    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    # ── signal handler ───────────────────────────────────────────────
    _emergency_model = model
    _emergency_processor = processor
    _emergency_dir = args.output_dir

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] Saving emergency checkpoint …")
        try:
            save_dir = os.path.join(_emergency_dir, "emergency_checkpoint")
            _emergency_model.save_pretrained(save_dir)
            _emergency_processor.save_pretrained(save_dir)
            print(f"Saved to {save_dir}")
        except Exception as exc:
            print(f"Could not save: {exc}")
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── print plan ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training plan (train_v3 — custom loop)")
    print(f"{'=' * 60}")
    print(f"  Samples ............ {len(train_dataset)}")
    if args.two_phase:
        print(f"  Phase 1 ............ {args.phase1_epochs} epochs (decoder only)")
        print(f"  Phase 2 ............ {args.phase2_epochs} epochs (full model)")
    else:
        print(f"  Epochs ............. {args.epochs}")
    print(f"  Effective batch .... {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Peak LR ............ {args.lr}")
    print(f"  Weighted loss ...... {'ON' if args.weighted_loss else 'OFF'}")
    print(f"  Focal loss ......... {'ON (γ=' + str(args.focal_gamma) + ')' if args.focal_loss else 'OFF'}")
    print(f"  NEFTune ............ {'ON (α=' + str(args.neftune_alpha) + ')' if args.neftune else 'OFF'}")
    print(f"  EMA ................ {'ON (decay=' + str(args.ema_decay) + ')' if args.ema else 'OFF'}")
    print(f"  Augmentation ....... {'ON (' + args.augment_strength + ')' if args.augment else 'OFF'}")
    print(f"  Gen eval samples ... {args.gen_eval_samples}")
    if args.lora:
        print(f"  LoRA rank .......... {args.lora_r}")
    print(f"{'=' * 60}\n")

    # ── TWO-PHASE TRAINING ───────────────────────────────────────────
    if args.two_phase:
        # ─── PHASE 1: Vision frozen, train decoder ───────────────────
        ema1 = EMAModel(model, decay=args.ema_decay) if args.ema else None

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer1 = AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        total_steps_p1 = (
            len(train_loader) * args.phase1_epochs // args.grad_accum
        )
        scheduler1 = OneCycleLR(
            optimizer1,
            max_lr=args.lr,
            total_steps=max(total_steps_p1, 1),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        global_step = train_phase(
            phase_name="Phase1-DecoderOnly",
            model=model,
            processor=processor,
            train_loader=train_loader,
            val_dataset=val_dataset,
            optimizer=optimizer1,
            scheduler=scheduler1,
            device=device,
            dtype=torch_dtype,
            epochs=args.phase1_epochs,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            token_weights=token_weights,
            use_focal=args.focal_loss,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            neftune=nef,
            ema=ema1,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            gen_eval_samples=args.gen_eval_samples,
            save_steps=args.save_steps,
            history=history,
            global_step=global_step,
        )

        # ─── PHASE 2: Unfreeze vision with discriminative LR ────────
        print("\nUnfreezing vision encoder for Phase 2 …")
        vision_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if "vision" in name or "image_model" in name:
                param.requires_grad = True
                vision_params.append(param)
            elif param.requires_grad:
                decoder_params.append(param)

        phase2_lr = args.lr * 0.5  # Lower base LR for phase 2
        ema2 = EMAModel(model, decay=args.ema_decay) if args.ema else None

        optimizer2 = AdamW(
            [
                {"params": decoder_params, "lr": phase2_lr},
                {"params": vision_params, "lr": phase2_lr * args.phase2_vision_lr_scale},
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        total_steps_p2 = (
            len(train_loader) * args.phase2_epochs // args.grad_accum
        )
        scheduler2 = OneCycleLR(
            optimizer2,
            max_lr=[phase2_lr, phase2_lr * args.phase2_vision_lr_scale],
            total_steps=max(total_steps_p2, 1),
            pct_start=0.05,
            anneal_strategy="cos",
        )

        global_step = train_phase(
            phase_name="Phase2-FullModel",
            model=model,
            processor=processor,
            train_loader=train_loader,
            val_dataset=val_dataset,
            optimizer=optimizer2,
            scheduler=scheduler2,
            device=device,
            dtype=torch_dtype,
            epochs=args.phase2_epochs,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            token_weights=token_weights,
            use_focal=args.focal_loss,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            neftune=nef,
            ema=ema2,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            gen_eval_samples=args.gen_eval_samples,
            save_steps=args.save_steps,
            history=history,
            global_step=global_step,
        )

        # Save final with EMA
        ema_final = ema2
    else:
        # ─── SINGLE-PHASE TRAINING ──────────────────────────────────
        ema_single = EMAModel(model, decay=args.ema_decay) if args.ema else None

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        total_steps = len(train_loader) * args.epochs // args.grad_accum
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=max(total_steps, 1),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        global_step = train_phase(
            phase_name="SinglePhase",
            model=model,
            processor=processor,
            train_loader=train_loader,
            val_dataset=val_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dtype=torch_dtype,
            epochs=args.epochs,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            token_weights=token_weights,
            use_focal=args.focal_loss,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            neftune=nef,
            ema=ema_single,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            gen_eval_samples=args.gen_eval_samples,
            save_steps=args.save_steps,
            history=history,
            global_step=global_step,
        )
        ema_final = ema_single

    # ── save final model ─────────────────────────────────────────────
    print(f"\nSaving final model to {args.output_dir}")
    if ema_final is not None:
        ema_final.apply(model)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    best_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)

    if ema_final is not None:
        ema_final.restore(model)

    # ── save history & plots ─────────────────────────────────────────
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    _plot_training_curves(history, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
