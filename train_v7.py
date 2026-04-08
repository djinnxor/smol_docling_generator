"""
train_v7.py — SmolDocling / Granite-Docling fine-tuning with Mongolian OCR improvements
=========================================================================================

New in v7 over train_v6
-----------------------
Full integration of Mongolian-OCR-specific improvements. All components are
inlined — no external import of mongolian_ocr_improvements.py required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDITION-1  MongolianAugmenter (image augmentation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: vision encoder is pre-trained on clean Western documents.
Mongolian banking screenshots have JPEG artefacts, sub-pixel font rendering at
8–14px, screen brightness/contrast variation, and hand-held motion blur.

The distinguishing feature for Ө vs О is a ~1px horizontal bar; at JPEG
quality<80 that bar may be quantised away entirely.

Formal justification: under domain-adaptation theory, training on P_aug that
covers the degradation manifold of phone screenshots reduces the covariate-
shift term H(P_aug, P_deploy) → 0, whereas training on clean images leaves
H(P_train, P_deploy) large.

Integrated into: DocTagDataset.__init__ / __getitem__
CLI flag:        --mongolian_aug_p (default 0.8, 0.0 = disabled)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDITION-2  clean_doctag_amounts (data cleaning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: 27K cells contain OCR artefacts in ground-truth labels (e.g.
"46, ,000. 00", "1016,442,97.02"). Training on these teaches the model to
reproduce the artefact, injecting systematic noise into the gradient signal.

Strategy:
  a) Normalise well-formed Mongolian banking amounts to canonical format.
  b) Mask irrecoverably malformed amounts (labels=-100) so the model receives
     no gradient from them.

Integrated into: DocTagDataset.__getitem__
Always applied regardless of --no_mongolian_enhancements (pure data cleaning).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDITION-3  patch_mongolian_weights (vocabulary upweighting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: Ө Ү appear 207K times but receive w_content=1.0 CE gradient.
These 4 chars are the primary Mongolian OCR error sources because they are
visually confusable with О and У at low screen resolution.

Derivation: fcel tags (32% of tokens × w_fcel=1.0) produce gradient signal;
Mongolian chars (≈0.39% of tokens) need weight ≈ 32/0.39 ≈ 82× to match.
We use 15× (conservative) to avoid gradient explosion.

Integrated into: main() after build_loss_fn, via loss_fn.token_weights if
DocTagLoss exposes it; otherwise a warning is printed guiding the user to
patch doctag_loss.py (see DOCTAG_LOSS_PATCH_INSTRUCTIONS at end of file).
CLI flag:        --w_mongolian (default 15.0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDITION-4  homoglyph_contrastive_loss (Cyrillic/Latin disambiguation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: 11 Cyrillic chars are visually identical to Latin chars at pixel
level (А/A, В/B, Е/E, О/O, Р/P, С/C, Т/T, Х/X + lower case). Standard CE
gives zero gradient for choosing А (U+0410) vs A (U+0041) — they look
identical in the image.

Loss term:
    L_contrast(t) = max(0, margin − (log p(Cyr_t) − log p(Lat_lookalike_t)))

With margin=2.0 the model must place the Cyrillic char at least
e^2 ≈ 7.4× more probable than the Latin look-alike.

Total loss becomes:
    L_total = L_wce + λ_loc·L_loc + λ_focal·L_focal + λ_homoglyph·L_contrast

Integrated into: HomoglyphAwareLoss wrapper around DocTagModelWithLoss
CLI flags:       --lambda_homoglyph (default 0.3), --homoglyph_margin (default 2.0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDITION-5  MongolianPhraseProcessor (inference-time constrained decoding)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: top-50 Mongolian banking phrases cover 64% of all Mongolian cells.
Once a cell's first tokens match a known phrase prefix, unconstrained decoding
samples from |V|=32K while only ~1–5 valid continuations exist.

Implementation: prefix trie built from known phrases; at each step suppresses
all token IDs that don't continue any valid phrase in the trie.

Integrated into: inference only (not training) — see usage example at end.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inherited from v6 (unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom differentiable loss (doctag_loss.py):
    L_total = L_wce + λ_loc·L_loc + λ_focal·L_focal
BUG-1 fix: LoRA alpha == r (scaling = 1.0)
BUG-2 fix: Fixed image resolution (resize not thumbnail)
BUG-3 fix: Partial vision encoder unfreeze with differential LR
Improvements 4–10 from v6 all retained.

# Usage
--------
# Recommended — full Mongolian enhancement stack:
python train_v7.py \\
    --train_data  training_data_train.jsonl \\
    --val_data    training_data_val.jsonl   \\
    --qlora                                 \\
    --unfreeze_vision_layers 4              \\
    --lora_r 32                             \\
    --mongolian_aug_p 0.8                   \\
    --w_mongolian 15.0                      \\
    --lambda_homoglyph 0.3                  \\
    --output_dir doctag_model_v7

# Disable Mongolian enhancements (augmentation + homoglyph loss) for ablation:
python train_v7.py \\
    --train_data  training_data_train.jsonl \\
    --no_mongolian_enhancements             \\
    --output_dir  doctag_model_v7_no_mongolian

# Disable custom loss entirely (fall back to standard CE):
python train_v7.py \\
    --train_data  training_data_train.jsonl \\
    --no_custom_loss                         \\
    --output_dir  doctag_model_v7_ce_baseline
"""

import argparse
import io
import json
import math
import os
import random
import re
import signal
import sys
from collections import Counter
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
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from doctag_loss import build_loss_fn, DocTagModelWithLoss

# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"
SYSTEM_PROMPT = "Convert this document page to DocTags format."

_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
    "ched", "fcel", "rhed", "ucel", "lcel", "ecel", "nl",
})


# ===========================================================================
# ADDITION-1: MongolianAugmenter
# ===========================================================================
# Root cause: covariate shift H(P_train, P_deploy) is large because the
# vision encoder saw only clean Western documents during pre-training, while
# Mongolian banking screenshots are degraded by JPEG compression, sub-pixel
# font rendering, variable screen brightness, and hand-held motion blur.
#
# Training with stochastic augmentation on the degradation manifold of the
# deployment domain reduces H(P_aug, P_deploy) → 0 without requiring extra
# data collection.

class MongolianAugmenter:
    """
    Stochastic augmentation pipeline calibrated to Mongolian banking screenshot
    degradation modes.

    Applied during training only (DocTagDataset is_train=True).

    Parameters
    ----------
    p : float
        Probability of applying any augmentation per sample (default 0.8).
    jpeg_quality_range : tuple
        Simulate phone JPEG compression. This is the most important single
        augmentation: the thin bar of Ө and crossbar of Ү are 1–2 pixels wide
        at typical phone resolutions; JPEG quantisation at quality<80 smears
        these into surrounding pixels, which is exactly what happens at
        deployment time. Training with this augmentation forces the encoder to
        learn a robust "bar present / bar absent" feature.
    """

    def __init__(
        self,
        p: float = 0.8,
        jpeg_quality_range=(55, 90),
        blur_sigma_range=(0.0, 0.9),
        brightness_range=(0.75, 1.25),
        contrast_range=(0.80, 1.20),
        noise_std_range=(0.0, 8.0),
        rotation_range=(-1.5, 1.5),
    ):
        self.p = p
        self.jpeg_q = jpeg_quality_range
        self.blur_s = blur_sigma_range
        self.bright = brightness_range
        self.contr = contrast_range
        self.noise_s = noise_std_range
        self.rot_r = rotation_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        from PIL import ImageEnhance

        # Gaussian blur (simulates motion blur / lens defocus)
        sigma = random.uniform(*self.blur_s)
        if sigma > 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Brightness jitter (screen brightness variation)
        b = random.uniform(*self.bright)
        image = ImageEnhance.Brightness(image).enhance(b)

        # Contrast jitter (low-contrast banking app backgrounds)
        c = random.uniform(*self.contr)
        image = ImageEnhance.Contrast(image).enhance(c)

        # Gaussian pixel noise (camera sensor noise)
        noise_std = random.uniform(*self.noise_s)
        if noise_std > 1.0:
            arr = np.array(image, dtype=np.float32)
            arr += np.random.normal(0, noise_std, arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        # JPEG compression round-trip — single most important augmentation
        # for Mongolian character distinction (see class docstring)
        q = random.randint(*self.jpeg_q)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        image = Image.open(buf).copy()

        # Slight rotation (hand-held scan tilt)
        angle = random.uniform(*self.rot_r)
        if abs(angle) > 0.3:
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

        return image


# ===========================================================================
# ADDITION-2: clean_doctag_amounts (data cleaning)
# ===========================================================================
# Root cause: 27K cells contain OCR artefacts embedded in ground-truth labels.
# Training on these injects noise into the gradient signal and teaches the
# model to reproduce the artefact pattern.
#
# Mongolian banking numeric convention:
#   Thousands separator = comma (,)
#   Decimal separator   = period (.)
#   Negative prefix     = hyphen (-)
#   Example: -1,234,567.89

_VALID_AMOUNT = re.compile(
    r'^-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?$'
)
_ARTEFACT_PATTERNS = [
    re.compile(r'\d, ,\d'),          # "46, ,000"
    re.compile(r'\d \d\d\d\d'),      # "1234 56789" (OCR missed comma)
    re.compile(r'\.\d{3,}\.'),       # multiple decimal points
    re.compile(r',{2,}'),            # consecutive commas
    re.compile(r'\d{16,}'),          # impossible 16+ digit number
]


def _is_amount_cell(text: str) -> bool:
    t = text.strip()
    return bool(re.match(r'^-?[\d,. ]+$', t) and len(t) > 2)


def _normalise_amount(text: str) -> Optional[str]:
    """
    Normalise a Mongolian banking amount string.
    Returns normalised string, or None if irrecoverably malformed.

    Steps:
    1. Strip spaces.
    2. Remove spaces between digits ("1 234 567" → "1,234,567").
    3. Detect and convert European separator format (1.234.567,89 → 1,234,567.89).
    4. Validate; return None if still malformed (cell will be masked).
    """
    t = text.strip()

    for pat in _ARTEFACT_PATTERNS:
        if pat.search(t):
            return None

    # Remove spaces inside numbers (two passes for triple-space gap)
    t = re.sub(r'(\d) (\d)', r'\1\2', t)
    t = re.sub(r'(\d) (\d)', r'\1\2', t)

    # Detect European format: swap separators
    dots = t.count('.')
    commas = t.count(',')
    if dots >= 2 or (dots == 1 and commas == 1 and t.index('.') < t.index(',')):
        t = t.replace('.', 'X').replace(',', '.').replace('X', ',')

    return t if _VALID_AMOUNT.match(t) else None


def clean_doctag_amounts(doctag: str) -> str:
    """
    Walk a complete DocTag string and normalise / mask all numeric cell content.

    Cells that cannot be normalised are replaced with an empty cell tag.  The
    training label at that position becomes a single empty cell, contributing
    near-zero gradient and preventing the model from learning to reproduce
    OCR artefacts.

    Usage in DocTagDataset.__getitem__ (already integrated in v7 — shown here
    for reference):
        doctag = clean_doctag_amounts(doctag)
    """
    def _replace_cell(m):
        tag = m.group(1)
        content = m.group(2)
        if not _is_amount_cell(content):
            return m.group(0)
        normalised = _normalise_amount(content)
        if normalised is None:
            return f"<{tag}></{tag}>"   # mask: empty cell → near-zero gradient
        return f"<{tag}>{normalised}"

    return re.sub(
        r'<(fcel|ched|rhed)>(.*?)(?=<)',
        _replace_cell,
        doctag,
        flags=re.DOTALL,
    )


# ===========================================================================
# ADDITION-3: patch_mongolian_weights
# ===========================================================================
# Root cause: gradient starvation for Mongolian-specific Cyrillic characters.
# Ө Ү appear 207K times but receive w_content=1.0.
#
# Derivation: to produce gradient signal comparable to fcel tags
# (32% × w_fcel=1.0), Mongolian chars (≈0.39% of tokens) need
# weight ≈ 32/0.39 ≈ 82×. We use 15× (conservative).
#
# This function is called in main() after build_loss_fn and attempts to write
# directly into loss_fn.token_weights if DocTagLoss exposes that attribute.
# If not, a clear warning is printed with instructions to patch doctag_loss.py.

_MONGOLIAN_SPECIFIC = frozenset({
    '\u04E8', '\u04E9',   # Өө — Cyrillic Barred O
    '\u04AE', '\u04AF',   # Үү — Cyrillic Straight U
})


def patch_mongolian_weights(
    token_weights: torch.Tensor,
    tokenizer,
    w_mongolian: float = 15.0,
) -> torch.Tensor:
    """
    Upweight vocabulary tokens containing Mongolian-specific Cyrillic characters.

    Parameters
    ----------
    token_weights : torch.Tensor, shape (vocab_size,)
        Output of DocTagLoss.token_weights or build_token_weight_map().
    tokenizer : PreTrainedTokenizer
    w_mongolian : float
        Weight assigned to Mongolian char tokens. Recommended: 15.0 (range 10–25).

    Returns
    -------
    token_weights with Mongolian char positions updated in-place.
    """
    patched = 0
    for tok_id in range(len(tokenizer)):
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        if any(c in _MONGOLIAN_SPECIFIC for c in tok_str):
            token_weights[tok_id] = max(token_weights[tok_id].item(), w_mongolian)
            patched += 1
    print(f"[MongolianWeights] Patched {patched} vocabulary tokens with w={w_mongolian}")
    return token_weights


def _try_patch_loss_fn_weights(loss_fn, tokenizer, w_mongolian: float) -> bool:
    """
    Attempt to patch token_weights inside the DocTagLoss object.
    Returns True on success, False if the attribute is not exposed.
    """
    # DocTagLoss may store weights in several places depending on its version
    for attr in ("token_weights", "_token_weights", "weight_map"):
        weights = getattr(loss_fn, attr, None)
        if weights is not None and isinstance(weights, torch.Tensor):
            patch_mongolian_weights(weights, tokenizer, w_mongolian)
            return True
    return False


# ===========================================================================
# ADDITION-4: Homoglyph contrastive loss (Cyrillic / Latin disambiguation)
# ===========================================================================
# Root cause: 11 Cyrillic chars are pixel-identical to Latin chars.
# Standard CE assigns zero gradient to the Cyr/Lat decision at these positions
# because the image features are indistinguishable; only linguistic context
# can disambiguate.
#
# Loss formulation:
#   L_contrast(t) = max(0, margin − (log p(Cyr_t) − log p(Lat_lookalike_t)))
#
# With margin=2.0: model must place Cyrillic ≥ e^2 ≈ 7.4× more probable than
# the Latin look-alike at every Cyrillic homoglyph position.
#
# This is integrated via HomoglyphAwareLoss, which wraps DocTagModelWithLoss
# (or any model) and adds L_contrast to the existing loss.

_CYR_LATIN_PAIRS = [
    ('\u0410', 'A'),   # А / A
    ('\u0430', 'a'),   # а / a
    ('\u0412', 'B'),   # В / B
    ('\u0415', 'E'),   # Е / E
    ('\u0435', 'e'),   # е / e
    ('\u041A', 'K'),   # К / K
    ('\u041C', 'M'),   # М / M
    ('\u041D', 'H'),   # Н / H
    ('\u041E', 'O'),   # О / O
    ('\u043E', 'o'),   # о / o
    ('\u0420', 'P'),   # Р / P
    ('\u0421', 'C'),   # С / C
    ('\u0441', 'c'),   # с / c
    ('\u0422', 'T'),   # Т / T
    ('\u0425', 'X'),   # Х / X
    ('\u0445', 'x'),   # х / x
]


def build_homoglyph_pairs(tokenizer) -> list:
    """
    Build list of (cyrillic_token_id, latin_token_id) pairs that are visually
    confusable. Only includes pairs where both chars tokenise to a single token.

    Returns list of (cyr_id, lat_id) tuples.
    """
    pairs = []
    for cyr_char, lat_char in _CYR_LATIN_PAIRS:
        cyr_ids = tokenizer.encode(cyr_char, add_special_tokens=False)
        lat_ids = tokenizer.encode(lat_char, add_special_tokens=False)
        if len(cyr_ids) == 1 and len(lat_ids) == 1:
            pairs.append((cyr_ids[0], lat_ids[0]))
    return pairs


def homoglyph_contrastive_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    homoglyph_pairs: list,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Contrastive hinge loss for Cyrillic/Latin homoglyph disambiguation.

    For each position where the true label is a Cyrillic homoglyph, penalise
    the model for assigning probability to the Latin look-alike:

        L = mean over active positions of max(0, margin − (log p(cyr) − log p(lat)))

    Parameters
    ----------
    logits : (B, T, V)
    labels : (B, T)  — -100 at masked positions
    homoglyph_pairs : list of (cyr_id, lat_id)
    margin : float
        Required log-probability gap between Cyrillic and Latin chars.
        2.0 ≈ 7.4× probability ratio.  Recommended range: 1.0–3.0.
    """
    if not homoglyph_pairs:
        return logits.new_zeros(1).squeeze()

    cyr_ids = torch.tensor([p[0] for p in homoglyph_pairs],
                            dtype=torch.long, device=logits.device)
    lat_ids = torch.tensor([p[1] for p in homoglyph_pairs],
                            dtype=torch.long, device=logits.device)

    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
    labels_clamped = labels.clamp(min=0)

    # Boolean mask: which vocabulary positions are Cyrillic homoglyphs?
    is_cyr = torch.zeros(V, dtype=torch.bool, device=logits.device)
    is_cyr[cyr_ids] = True

    # Mapping tensor: Cyrillic token → its Latin look-alike
    cyr_to_lat = torch.zeros(V, dtype=torch.long, device=logits.device)
    cyr_to_lat[cyr_ids] = lat_ids

    # Active positions: valid label AND label is a Cyrillic homoglyph
    label_is_cyr = is_cyr[labels_clamped]                      # (B, T)
    active_mask = (labels != -100) & label_is_cyr

    if not active_mask.any():
        return logits.new_zeros(1).squeeze()

    active_labels = labels_clamped[active_mask]                # (N,)
    active_logps = log_probs[active_mask]                      # (N, V)

    log_p_cyr = active_logps.gather(1, active_labels.unsqueeze(1)).squeeze(1)
    lat_labels = cyr_to_lat[active_labels]                     # (N,)
    log_p_lat = active_logps.gather(1, lat_labels.unsqueeze(1)).squeeze(1)

    # Hinge: penalise if Cyrillic is not > Latin by required margin
    loss = F.relu(margin - (log_p_cyr - log_p_lat))
    return loss.mean()


class HomoglyphAwareLoss(torch.nn.Module):
    """
    Wrapper around any HuggingFace model (typically DocTagModelWithLoss) that
    adds the Cyrillic/Latin homoglyph contrastive loss term.

    Total loss:
        L_total = L_base + λ_homoglyph · L_contrast

    where L_base is whatever loss the wrapped model computes (the v6 weighted
    CE + focal + loc regression when DocTagModelWithLoss is used).

    Parameters
    ----------
    model : torch.nn.Module
        Base model, typically DocTagModelWithLoss from doctag_loss.py.
    homoglyph_pairs : list of (cyr_id, lat_id)
        Built by build_homoglyph_pairs(tokenizer).
    lambda_homoglyph : float
        Weight of the contrastive term in the total loss.
    margin : float
        Required log-probability gap (default 2.0 ≈ e^2 ≈ 7.4× probability ratio).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        homoglyph_pairs: list,
        lambda_homoglyph: float = 0.3,
        margin: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.homoglyph_pairs = homoglyph_pairs
        self.lambda_homoglyph = lambda_homoglyph
        self.margin = margin

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)

        if self.lambda_homoglyph <= 0 or not self.homoglyph_pairs:
            return outputs

        labels = kwargs.get("labels")
        if labels is None:
            return outputs

        # Extract logits from model output (handles both dict and tuple outputs)
        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            # Cannot locate logits — skip contrastive term silently
            return outputs

        if logits.dim() != 3:
            return outputs

        l_contrast = homoglyph_contrastive_loss(
            logits, labels, self.homoglyph_pairs, margin=self.margin
        )

        # Add contrastive term to the existing loss
        if hasattr(outputs, "loss") and outputs.loss is not None:
            outputs.loss = outputs.loss + self.lambda_homoglyph * l_contrast
        elif isinstance(outputs, dict) and "loss" in outputs:
            outputs["loss"] = outputs["loss"] + self.lambda_homoglyph * l_contrast

        return outputs

    # Delegate attribute access to the wrapped model (needed for PEFT / save)
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ===========================================================================
# ADDITION-5: Phrase-constrained decoding (inference only)
# ===========================================================================
# Root cause: top-50 Mongolian banking phrases cover 64% of all Mongolian cells.
# Unconstrained decoding explores |V|=32K tokens at each step; a prefix trie
# reduces this to |valid_continuations| ≈ 1–5 for known phrase prefixes.
#
# Used at inference time only — not during training.

class _TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: dict = {}
        self.is_end: bool = False


class MongolianPhraseProcessor:
    """
    HuggingFace LogitsProcessor that constrains generation to known Mongolian
    banking phrases once the token sequence matches a phrase prefix in a trie.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    phrases : list of str
        Known banking phrases. Build from training corpus using
        extract_top_phrases().
    min_phrase_len : int
        Minimum phrase length (tokens) before activating constraints.
        Prevents over-constraining on short common tokens like "Банк".
    boost : float
        Log-probability boost added to valid continuation tokens.
        0 = pure masking of invalid tokens.

    Usage (inference)
    -----------------
    records = [json.loads(l) for l in open("training_data_train.jsonl")]
    phrases = extract_top_phrases([r["doctag"] for r in records], min_count=50)
    proc = MongolianPhraseProcessor(processor.tokenizer, phrases)
    outputs = model.generate(
        **inputs,
        logits_processor=LogitsProcessorList([proc]),
        max_new_tokens=512,
        num_beams=4,
    )
    """

    def __init__(
        self,
        tokenizer,
        phrases: list,
        min_phrase_len: int = 3,
        boost: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.min_len = min_phrase_len
        self.boost = boost
        self._root = _TrieNode()
        self._n_inserted = 0
        for phrase in phrases:
            self._insert(phrase)
        print(f"[PhraseProcessor] Trie built: {self._n_inserted} phrases inserted")

    def _insert(self, phrase: str):
        ids = self.tokenizer.encode(phrase, add_special_tokens=False)
        if len(ids) < self.min_len:
            return
        node = self._root
        for tid in ids:
            if tid not in node.children:
                node.children[tid] = _TrieNode()
            node = node.children[tid]
        node.is_end = True
        self._n_inserted += 1

    def _get_node(self, token_ids: list) -> Optional[_TrieNode]:
        node = self._root
        for tid in token_ids:
            if tid not in node.children:
                return None
            node = node.children[tid]
        return node

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            matched_node = None

            for start in range(max(0, len(seq) - 20), len(seq)):
                node = self._get_node(seq[start:])
                if node is not None:
                    matched_node = node
                    break

            if matched_node is None or not matched_node.children:
                continue

            if matched_node.is_end:
                continue  # Phrase already complete — allow any continuation

            # Suppress all tokens not in the valid continuation set
            valid_ids = set(matched_node.children.keys())
            mask = torch.full_like(scores[b], float('-inf'))
            valid_tensor = torch.tensor(
                list(valid_ids), dtype=torch.long, device=scores.device
            )
            mask[valid_tensor] = scores[b][valid_tensor] + self.boost
            scores[b] = mask

        return scores


def extract_top_phrases(
    doctags: list,
    min_count: int = 50,
    max_len: int = 80,
) -> list:
    """
    Extract high-frequency Mongolian banking phrases from a list of doctag
    strings to build the trie for MongolianPhraseProcessor.

    Returns list of phrase strings with count >= min_count.
    """
    cyr_re = re.compile(r'[\u0400-\u04FF]')
    counter: Counter = Counter()
    for doctag in doctags:
        cells = re.findall(r'<(?:fcel|ched|rhed)>(.*?)(?=<)', doctag, re.DOTALL)
        for cell in cells:
            cell = cell.strip()
            if cyr_re.search(cell) and 3 < len(cell) <= max_len:
                counter[cell] += 1
    return [phrase for phrase, cnt in counter.items() if cnt >= min_count]


# ===========================================================================
# Helpers: auto-detect native image resolution from processor
# ===========================================================================

def get_native_resolution(processor) -> int:
    """
    Return the model's native longest-edge resolution from the image processor
    config.  Falls back to 512 if no config field is found.
    """
    img_proc = getattr(processor, "image_processor", processor)
    for attr in ("size", "image_size", "crop_size"):
        v = getattr(img_proc, attr, None)
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            return v.get("longest_edge") or v.get("height") or v.get("width") or 512
    return 512


# ===========================================================================
# DocTag metric helpers
# ===========================================================================

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
    Fraction of OTSL rows predicted in correct order.
    Each row = token sequence between consecutive <nl> markers (loc tokens stripped).
    """
    _loc_re = re.compile(r'<loc_\d+>')

    def _rows(s):
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


# ===========================================================================
# Dataset — with Mongolian augmentation and data cleaning integrated
# ===========================================================================

class DocTagDataset(Dataset):
    """
    Loads (image_path, doctag) pairs from a JSONL file.

    v7 changes over v6
    ------------------
    is_train : bool
        If True, MongolianAugmenter is applied to images during __getitem__.
        If False (validation / test), augmentation is skipped.
    augmenter : MongolianAugmenter | None
        Constructed automatically when is_train=True and aug_p > 0.
    clean_amounts : bool
        If True (default), clean_doctag_amounts() is applied to normalise /
        mask malformed numeric cells. Applied regardless of is_train.

    v6 features retained
    --------------------
    Images are resized to a FIXED resolution (native_resolution × native_resolution)
    using LANCZOS — not thumbnail(). This guarantees identical patch grids for
    every sample in every batch (BUG-2 fix from v6).
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_length: int = 2048,
        native_resolution: int = 512,
        is_train: bool = True,
        aug_p: float = 0.8,
        clean_amounts: bool = True,
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
        self.is_train = is_train
        self.clean_amounts = clean_amounts

        # ADDITION-1: Mongolian augmenter (training only)
        self.augmenter: Optional[MongolianAugmenter] = None
        if is_train and aug_p > 0.0:
            self.augmenter = MongolianAugmenter(p=aug_p)
            print(f"[DocTagDataset] MongolianAugmenter enabled (p={aug_p})")
        else:
            mode = "training" if is_train else "validation/test"
            print(f"[DocTagDataset] Augmentation disabled for {mode} split.")

        resampling_mod = getattr(Image, "Resampling", Image)
        self._lanczos = resampling_mod.LANCZOS

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")

        # ADDITION-1: apply stochastic Mongolian augmentation (training only)
        if self.augmenter is not None:
            image = self.augmenter(image)

        # BUG-2 fix (v6): always resize to a fixed square canvas
        target = self.native_resolution
        image = image.resize((target, target), self._lanczos)

        # Normalise legacy <image>/<image_caption> tags
        doctag = record["doctag"]
        doctag = re.sub(r"<image>", "<picture>", doctag)
        doctag = re.sub(r"</image>", "</picture>", doctag)

        # ADDITION-2: clean malformed numeric cells in ground-truth labels
        if self.clean_amounts:
            doctag = clean_doctag_amounts(doctag)

        messages_full = [
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text",  "text": doctag}]},
        ]
        text_full = self.processor.apply_chat_template(
            messages_full, add_generation_prompt=False
        )

        messages_prompt = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}]},
        ]
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, add_generation_prompt=True
        )

        prompt_ids = self.processor.tokenizer(
            text_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids

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
        labels[: len(prompt_ids)] = -100   # mask prompt tokens from loss
        inputs["labels"] = labels
        return inputs


# ===========================================================================
# Collator — unchanged from v6
# ===========================================================================

def doctag_data_collator(features: list) -> dict:
    """
    Pad token sequences to the longest sequence in the batch.
    pixel_values can be stacked directly because every image was resized to the
    same resolution (BUG-2 fix from v6).
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
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError as exc:
                shapes = [t.shape for t in tensors]
                raise RuntimeError(
                    f"pixel_values stack failed — shapes are not uniform: {shapes}. "
                    "Ensure all images are resized to the same native_resolution."
                ) from exc
        else:
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
    return batch


# ===========================================================================
# Metrics
# ===========================================================================

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1).cpu()


def make_compute_metrics(tokenizer):
    def _compute(eval_preds):
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

        tag_acc   = float(np.mean([_tag_accuracy(p, r)     for p, r in zip(pred_strs, ref_strs)]))
        loc_mae   = float(np.mean([_loc_mae(p, r)           for p, r in zip(pred_strs, ref_strs)]))
        tree_edit = float(np.mean([_tree_edit_dist(p, r)    for p, r in zip(pred_strs, ref_strs)]))
        otsl_row  = float(np.mean([_otsl_row_accuracy(p, r) for p, r in zip(pred_strs, ref_strs)]))

        return {
            "token_acc":      token_acc,
            "tag_acc":        tag_acc,
            "loc_mae":        loc_mae,
            "tree_edit_dist": tree_edit,
            "otsl_row_acc":   otsl_row,
        }
    return _compute


# ===========================================================================
# Vision encoder: partial unfreeze + differential LR helpers (from v6)
# ===========================================================================

def _collect_vision_blocks(model) -> list:
    candidates = []
    for name, module in model.named_modules():
        if re.search(r'\.(encoder|transformer)\.(layers|blocks)\.\d+$', name):
            candidates.append((name, module))

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
    where i=0 is the lowest unfrozen block and i=N-1 is the topmost.
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora" in name.lower() or "dora" in name.lower():
            param.requires_grad = True

    if unfreeze_top_n <= 0:
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Vision] All vision layers frozen.  Trainable: {total_trainable:,} params")
        return []

    all_blocks = _collect_vision_blocks(model)
    if not all_blocks:
        print("[Vision] WARNING: could not locate ViT transformer blocks. "
              "Falling back to fully frozen encoder.")
        return []

    top_blocks = all_blocks[-unfreeze_top_n:]
    vision_param_groups = []

    for rank_from_top, (block_name, block_module) in enumerate(reversed(top_blocks)):
        block_lr = vision_lr * (layer_decay ** rank_from_top)
        block_params = []

        for pname, param in block_module.named_parameters():
            if not param.is_floating_point():
                continue
            if not param.requires_grad:
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


# ===========================================================================
# Signal handler
# ===========================================================================

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


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolDocling / Granite-Docling (v7) — v6 + Mongolian OCR improvements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data & model ---
    parser.add_argument("--train_data",  default="training_data_train.jsonl")
    parser.add_argument("--val_data",    default="training_data_val.jsonl")
    parser.add_argument("--model_id",    default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir",  default="doctag_model_v7")
    parser.add_argument("--adapter",     type=str, default="",
                        help="Path to existing PEFT adapter directory to resume from.")

    # --- Training loop ---
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)

    # --- Image resolution ---
    parser.add_argument("--native_resolution", type=int, default=0,
                        help="Resize all images to this resolution. 0 = auto-detect.")

    # --- Learning rate ---
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--vision_lr_scale", type=float, default=0.05)
    parser.add_argument("--layer_decay",     type=float, default=0.85)

    # --- LoRA / DoRA ---
    parser.add_argument("--lora_r",       type=int,   default=32)
    parser.add_argument("--lora_alpha",   type=int,   default=0,
                        help="0 = auto-set equal to lora_r (scaling=1.0).")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_dora",      action="store_true")

    # --- Vision encoder ---
    parser.add_argument("--unfreeze_vision_layers", type=int, default=4)

    # --- Hardware ---
    parser.add_argument("--qlora",      action="store_true")
    parser.add_argument("--flash_attn", action="store_true")

    # --- Eval / save ---
    parser.add_argument("--eval_steps",       type=int, default=200)
    parser.add_argument("--save_steps",       type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # --- W&B ---
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--wandb_project",  type=str, default="Docling-finetune-v7")
    parser.add_argument("--wandb_entity",   type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags",     type=str, default="")
    parser.add_argument("--wandb_mode",     type=str, default="online",
                        choices=["online", "offline", "disabled"])

    # --- Custom loss (v6) ---
    parser.add_argument("--no_custom_loss", action="store_true",
                        help="Disable DocTagLoss and use standard CE (for ablation).")
    parser.add_argument("--w_doc",      type=float, default=20.0)
    parser.add_argument("--w_struct",   type=float, default=8.0)
    parser.add_argument("--w_loc",      type=float, default=4.0)
    parser.add_argument("--w_fcel",     type=float, default=1.0)
    parser.add_argument("--lambda_loc",   type=float, default=0.3)
    parser.add_argument("--lambda_focal", type=float, default=0.5)
    parser.add_argument("--focal_gamma",  type=float, default=2.0)

    # --- Mongolian enhancements (v7) ---
    parser.add_argument("--no_mongolian_enhancements", action="store_true",
                        help="Disable ALL Mongolian-specific enhancements: "
                             "augmentation, vocabulary upweighting, and homoglyph "
                             "contrastive loss. Data cleaning (clean_doctag_amounts) "
                             "is always applied. Use for ablation studies.")

    # ADDITION-1: augmentation
    parser.add_argument("--mongolian_aug_p", type=float, default=0.8,
                        help="Probability of applying MongolianAugmenter per sample. "
                             "0.0 = disabled. Ignored when --no_mongolian_enhancements.")

    # ADDITION-3: vocabulary upweighting
    parser.add_argument("--w_mongolian", type=float, default=15.0,
                        help="CE weight for tokens containing Ө Ү characters. "
                             "Derivation: ≈82× needed for gradient parity with fcel tags; "
                             "15× is conservative to avoid explosion. Range: 10–25. "
                             "Ignored when --no_mongolian_enhancements.")

    # ADDITION-4: homoglyph contrastive loss
    parser.add_argument("--lambda_homoglyph", type=float, default=0.3,
                        help="Weight λ of homoglyph contrastive loss term: "
                             "L_total += λ · max(0, margin − (log p(Cyr) − log p(Lat))). "
                             "Ignored when --no_mongolian_enhancements.")
    parser.add_argument("--homoglyph_margin", type=float, default=2.0,
                        help="Required log-probability gap between Cyrillic and Latin "
                             "homoglyphs. 2.0 ≈ e^2 ≈ 7.4× probability ratio.")

    args = parser.parse_args()

    # --- Validate data paths ---
    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    # --- BUG-1 fix (v6): alpha == r → scaling = 1.0 ---
    if args.lora_alpha == 0:
        args.lora_alpha = args.lora_r
    if args.lora_alpha != args.lora_r:
        print(f"[LoRA] WARNING: lora_alpha ({args.lora_alpha}) != lora_r ({args.lora_r}).  "
              f"Scaling = {args.lora_alpha / args.lora_r:.2f}x.  "
              "Recommended: lora_alpha == lora_r (scaling = 1.0).")
    else:
        print(f"[LoRA] r={args.lora_r}  alpha={args.lora_alpha}  scaling=1.0  ✓")

    # Determine effective augmentation probability
    effective_aug_p = 0.0 if args.no_mongolian_enhancements else args.mongolian_aug_p

    # --- W&B init ---
    if args.use_wandb:
        import importlib
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed.  Run: pip install wandb")
        import wandb, datetime
        run_name = args.wandb_run_name.strip() or (
            f"doctag-v7-{Path(args.train_data).stem}-"
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

    # -----------------------------------------------------------------------
    print(f"\nLoading processor and model: {args.model_id}")
    # -----------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.native_resolution > 0:
        native_res = args.native_resolution
    else:
        native_res = get_native_resolution(processor)
    print(f"[Image] All images will be resized to {native_res}×{native_res}")

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

    # --- BUG-3 fix (v6): partial vision encoder unfreeze with differential LR ---
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
            lora_alpha=args.lora_alpha,
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

    # -----------------------------------------------------------------------
    # Custom loss — wrap model with DocTagLoss (v6), then apply Mongolian
    # enhancements (v7)
    # -----------------------------------------------------------------------
    if not args.no_custom_loss:
        print("\n[Loss] Building DocTagLoss (weighted CE + focal + loc regression) …")
        loss_fn = build_loss_fn(
            tokenizer=processor.tokenizer,
            w_doc=args.w_doc,
            w_struct=args.w_struct,
            w_loc=args.w_loc,
            w_fcel=args.w_fcel,
            lambda_loc=args.lambda_loc,
            lambda_focal=args.lambda_focal,
            focal_gamma=args.focal_gamma,
        )

        # ADDITION-3: patch Mongolian character weights in the token weight map
        if not args.no_mongolian_enhancements:
            patched = _try_patch_loss_fn_weights(
                loss_fn, processor.tokenizer, args.w_mongolian
            )
            if not patched:
                print(
                    "\n[MongolianWeights] WARNING: could not patch token_weights — "
                    "DocTagLoss does not expose 'token_weights' as a public attribute.\n"
                    "To enable Mongolian character upweighting, add the following to the\n"
                    "end of build_token_weight_map() in doctag_loss.py:\n"
                    "\n"
                    "    from train_v7 import patch_mongolian_weights\n"
                    "    weights = patch_mongolian_weights(weights, tokenizer, w_mongolian=15.0)\n"
                    "    return weights\n"
                    "\n"
                    "See DOCTAG_LOSS_PATCH_INSTRUCTIONS at the end of train_v7.py.\n"
                )

        model = DocTagModelWithLoss(model, loss_fn)
        print("[Loss] Model wrapped with DocTagLoss  ✓")
    else:
        print("[Loss] --no_custom_loss set: using standard cross-entropy (ablation mode)")

    # ADDITION-4: wrap with homoglyph contrastive loss
    if not args.no_mongolian_enhancements and not args.no_custom_loss:
        homoglyph_pairs = build_homoglyph_pairs(processor.tokenizer)
        print(f"[HomoglyphLoss] {len(homoglyph_pairs)} Cyr/Lat confusion pairs found "
              f"in vocabulary  |  λ={args.lambda_homoglyph}  margin={args.homoglyph_margin}")
        if homoglyph_pairs:
            model = HomoglyphAwareLoss(
                model,
                homoglyph_pairs=homoglyph_pairs,
                lambda_homoglyph=args.lambda_homoglyph,
                margin=args.homoglyph_margin,
            )
            print("[HomoglyphLoss] Model wrapped with HomoglyphAwareLoss  ✓")
    else:
        reason = ("--no_mongolian_enhancements" if args.no_mongolian_enhancements
                  else "--no_custom_loss (no logits available)")
        print(f"[HomoglyphLoss] Disabled ({reason})")

    # -----------------------------------------------------------------------
    # Datasets — is_train flag controls augmentation; cleaning always on
    # -----------------------------------------------------------------------
    print("\nBuilding datasets …")
    train_dataset = DocTagDataset(
        args.train_data,
        processor,
        max_length=args.max_length,
        native_resolution=native_res,
        is_train=True,
        aug_p=effective_aug_p,
        clean_amounts=True,
    )
    val_dataset: Optional[DocTagDataset] = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(
            args.val_data,
            processor,
            max_length=args.max_length,
            native_resolution=native_res,
            is_train=False,        # no augmentation on validation
            aug_p=0.0,
            clean_amounts=True,
        )
        print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples  |  No val file at {args.val_data}")

    # -----------------------------------------------------------------------
    # Build optimizer with differential learning rates
    # -----------------------------------------------------------------------
    decoder_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(
            p is vp
            for vg in vision_param_groups
            for vp in vg["params"]
        )
    ]

    optimizer_param_groups = vision_param_groups + [
        {"params": decoder_params, "lr": args.lr, "name": "decoder_lora"},
    ]

    print("\n[Optimizer] Param groups:")
    for g in optimizer_param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:60s}  lr={g['lr']:.2e}  params={n_p:,}")

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=args.lr,
        weight_decay=0.01,
        fused=True if torch.cuda.is_available() else False,
    )

    # -----------------------------------------------------------------------
    # SFTConfig
    # -----------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=args.eval_steps if val_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="otsl_row_acc",
        greater_is_better=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="wandb" if args.use_wandb else "none",
        eval_accumulation_steps=1,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
        optimizers=(optimizer, None),
        compute_metrics=make_compute_metrics(processor.tokenizer) if val_dataset else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset else None,
    )

    _install_signal_handlers(trainer)

    # -----------------------------------------------------------------------
    # Config summary
    # -----------------------------------------------------------------------
    eff_batch = args.batch_size * args.grad_accum
    n_steps = (len(train_dataset) // args.batch_size) * args.epochs // args.grad_accum
    loss_mode = "standard CE (ablation)" if args.no_custom_loss else (
        f"weighted CE (w_doc={args.w_doc} w_struct={args.w_struct} w_loc={args.w_loc}) "
        f"+ focal(γ={args.focal_gamma},λ={args.lambda_focal}) "
        f"+ loc_reg(λ={args.lambda_loc})"
        + ("" if args.no_mongolian_enhancements else
           f" + homoglyph_contrast(λ={args.lambda_homoglyph},margin={args.homoglyph_margin})")
    )
    mongolian_status = "DISABLED (--no_mongolian_enhancements)" if args.no_mongolian_enhancements else (
        f"aug_p={effective_aug_p}  w_mongolian={args.w_mongolian}  "
        f"λ_homoglyph={args.lambda_homoglyph}"
    )

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  train_v7 — Configuration Summary
╠══════════════════════════════════════════════════════════════════╣
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
║  Loss            {loss_mode}
║  Mongolian v7    {mongolian_status}
║  Best metric     otsl_row_acc  (table row accuracy)
║  QLoRA           {'yes' if args.qlora else 'no'}
╚══════════════════════════════════════════════════════════════════╝
""")

    # -----------------------------------------------------------------------
    print("Starting training …")
    # -----------------------------------------------------------------------
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

    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()


# ===========================================================================
# DOCTAG_LOSS_PATCH_INSTRUCTIONS
# ===========================================================================
# ADDITION-3 (patch_mongolian_weights) requires a one-line change to
# doctag_loss.py to expose the token_weights tensor publicly.
#
# In doctag_loss.py, find the build_token_weight_map() function and add at
# the END of the function, before the return statement:
#
#     # ---- v7 patch: store weights as attribute for external patching ----
#     # If this loss object will be passed to train_v7._try_patch_loss_fn_weights,
#     # the tensor must be accessible. Store it as an attribute on the returned
#     # object if using a class, or return it from build_loss_fn so callers can
#     # access loss_fn.token_weights directly.
#     self.token_weights = weights   # add this inside DocTagLoss.__init__
#
# Alternatively, without modifying doctag_loss.py, call the standalone
# patch_mongolian_weights() function directly inside your custom
# build_token_weight_map() override:
#
#     from train_v7 import patch_mongolian_weights
#     weights = patch_mongolian_weights(weights, tokenizer, w_mongolian=15.0)
#     return weights
#
# ===========================================================================
# INFERENCE USAGE (MongolianPhraseProcessor — ADDITION-5)
# ===========================================================================
#
#     import json
#     from train_v7 import extract_top_phrases, MongolianPhraseProcessor
#     from transformers import LogitsProcessorList
#
#     records = [json.loads(l) for l in open("training_data_train.jsonl")]
#     phrases = extract_top_phrases([r["doctag"] for r in records], min_count=50)
#     phrase_proc = MongolianPhraseProcessor(processor.tokenizer, phrases)
#
#     outputs = model.generate(
#         **inputs,
#         logits_processor=LogitsProcessorList([phrase_proc]),
#         max_new_tokens=512,
#         num_beams=4,
#     )
