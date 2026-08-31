"""
doctag_loss.py — Custom differentiable loss for SmolDocling / Granite-Docling
==============================================================================

Mathematical specification
--------------------------
Total loss:

    L_total = L_wce + λ_focal · L_focal + λ_loc · L_loc

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_wce — Weighted Cross-Entropy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: without weighting, doc-structure tags receive only 0.87% of the
total CE gradient signal because they are rare relative to content tokens.
This causes the model to over-optimise content transcription while under-
learning structure prediction — the primary failure mode on Mongolian banking
statement tables.

Definition:
    L_wce = -1/Z · Σ_{t: y_t ≠ -100} w(y_t) · log p_θ(y_t | context)
    Z = Σ_{t: y_t ≠ -100} w(y_t)    (normalisation by total active weight)

Per-token weight w(y_t):
    w_doc    (default 20) — opening/closing tags for document-level structure:
                            <doctag>, <text>, <title>, <caption>, <picture>,
                            <otsl>, <section_header_level_1/2/3>, <list_item>,
                            <formula>, <page_header>, <page_footer>, <code>,
                            <footnote>, and their </…> closing counterparts.
    w_struct (default 8)  — OTSL table-structure tokens:
                            <ched>, <rhed>, <ucel>, <lcel>, <ecel>, <nl>,
                            and their closing tags.
    w_loc    (default 4)  — bounding-box tokens <loc_0> … <loc_500>
    w_fcel   (default 1)  — cell-content token <fcel> / </fcel>
    1.0                   — all other vocabulary tokens (default background)

Implementation note: F.cross_entropy with the weight parameter computes the
sum-weighted loss normalised by the sum of weights of non-ignored tokens:
    L = -Σ_t w(y_t) log p(y_t) / Σ_t w(y_t)
which matches the definition above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_focal — Focal Loss on Structural-Tag Positions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: for structural tags the model quickly reaches high confidence on
common tags (e.g. <fcel> in long tables). Standard CE's gradient from easy
examples overwhelms the gradient from rare hard tags (e.g. <ched>, <otsl>).

Focal loss (Lin et al., 2017) modulates CE by (1-p_t)^γ so already-confident
predictions contribute diminishing gradient and hard misses dominate:

    L_focal = -1/N_struct · Σ_{t: y_t ∈ struct_ids} (1 - p_t)^γ · log p_t

where p_t = softmax(logits_t)[y_t] is the model's probability of the true
structural token, γ (focal_gamma, default 2.0) controls the modulation
strength, and N_struct is the number of active structural positions.

Limiting cases:
    γ=0 → focal = standard CE (no modulation)
    γ→∞ → focal collapses to hard mining (only the worst position)

Implementation: log p_t is computed via log_softmax for numerical stability.
p_t = exp(log p_t). The double application of exp(log(·)) is stable because
log_softmax is computed once; exp is applied only to the scalar at the true
token position, whose value is in (-∞, 0], so exp is in (0, 1].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_loc — Differentiable Huber Regression on Bounding-Box Tokens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root cause: bounding-box coordinates are encoded as discrete tokens
<loc_0> … <loc_500> on a 0–500 integer scale.  Standard CE treats these as
independent classes: predicting <loc_49> when the ground truth is <loc_50>
receives the same gradient as predicting <loc_499>. There is no ordinality
signal — the model cannot learn that nearby coordinates are better than distant
ones, only that they are wrong.

Fix: compute a soft expected coordinate as a differentiable expectation over
the 501 loc-token distribution, then apply Huber regression against the true
coordinate:

    E_θ[loc] = Σ_{k=0}^{500} k · p_θ(loc_k | context)

where p_θ(loc_k | context) is the softmax probability computed over only the
501 loc-token vocabulary entries (not the full 32K vocabulary). This gives a
continuous, differentiable "predicted coordinate" in [0, 500].

    L_loc = 1/N_loc · Σ_{t: y_t ∈ loc_ids} Huber(E_θ[loc_t], k_t, δ=25)

where k_t ∈ {0, …, 500} is the true coordinate of <loc_k_t> and

    Huber(ŷ, y, δ) = 0.5·(ŷ-y)²           if |ŷ-y| ≤ δ
                     δ·(|ŷ-y| - 0.5·δ)     otherwise

δ=25 corresponds to 5% of the 0–500 coordinate range.  Below this threshold
the loss behaves as MSE (smooth, fast convergence for close predictions);
above it the linear regime prevents gradient explosion from large outliers
(e.g. when the model initially places a coordinate on the wrong side of the
page).

The gradient ∂L_loc/∂logits flows through the softmax into all 501 loc-token
logit positions simultaneously — giving a continuous learning signal for the
spatial layout even from a single misprediction, unlike CE which updates only
the argmax.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Public API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    loss_fn = build_loss_fn(tokenizer, w_doc, w_struct, w_loc, w_fcel,
                            lambda_loc, lambda_focal, focal_gamma)

    model = DocTagModelWithLoss(peft_model, loss_fn)

    # External patching (train_v7 Mongolian upweighting):
    loss_fn.token_weights[some_tok_id] = new_weight  # in-place, safe

    # Forward (called automatically by HuggingFace Trainer):
    outputs = model(**batch)   # outputs.loss  = L_total (scalar)
                               # outputs.logits = raw logits (B, T, V)
"""

import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tag category definitions
# Mirrors _STRUCTURAL_TAGS in train_v6/v7 but split into weight groups.
# ---------------------------------------------------------------------------

# Document-level open/close structure tags → w_doc
_DOC_STRUCT_TAGS: frozenset = frozenset({
    "doctag",
    "text",
    "section_header_level_1",
    "section_header_level_2",
    "section_header_level_3",
    "title",
    "caption",
    "picture",
    "otsl",
    "list_item",
    "formula",
    "page_header",
    "page_footer",
    "code",
    "footnote",
})

# OTSL table-structure tags → w_struct
_OTSL_STRUCT_TAGS: frozenset = frozenset({
    "ched",
    "rhed",
    "ucel",
    "lcel",
    "ecel",
    "nl",
})

# Cell-content tag → w_fcel
_FCEL_TAGS: frozenset = frozenset({"fcel"})

# All structural tags combined (used for focal loss mask)
_ALL_STRUCT_TAGS: frozenset = _DOC_STRUCT_TAGS | _OTSL_STRUCT_TAGS | _FCEL_TAGS

# loc token range: <loc_0> … <loc_500>, coordinates on a 0–500 integer scale
_LOC_RANGE: int = 501   # indices 0 … 500 inclusive
_HUBER_DELTA: float = 25.0  # 5% of 0–500 range; below = MSE regime, above = linear


# ===========================================================================
# Token discovery helpers
# ===========================================================================

def build_token_weight_map(
    tokenizer,
    w_doc: float = 20.0,
    w_struct: float = 8.0,
    w_loc: float = 4.0,
    w_fcel: float = 1.0,
) -> torch.Tensor:
    """
    Build a per-token-ID weight tensor of shape (vocab_size,).

    Strategy: encode each expected tag string and loc-N string with
    add_special_tokens=False.  If the tokenizer returns a single token ID
    for a given string, that token is a special token and receives the
    appropriate weight.  If the tokenizer splits it into multiple sub-tokens
    (rare; only occurs if the tag was not added to the vocabulary as a
    special token), it is ignored and falls back to the default weight of 1.0.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    w_doc     : CE weight for document-structure open/close tags (default 20).
    w_struct  : CE weight for OTSL table-structure tags          (default 8).
    w_loc     : CE weight for <loc_N> bounding-box tokens        (default 4).
    w_fcel    : CE weight for <fcel> / </fcel> tokens            (default 1).

    Returns
    -------
    weights : torch.Tensor, dtype=float32, shape (vocab_size,)
        Public attribute of DocTagLoss.  Can be patched in-place
        (e.g. by train_v7.patch_mongolian_weights) after construction.
    """
    vocab_size = len(tokenizer)
    weights = torch.ones(vocab_size, dtype=torch.float32)

    def _assign(tag_set: frozenset, w: float):
        for tag in tag_set:
            for variant in (f"<{tag}>", f"</{tag}>"):
                try:
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                        weights[ids[0]] = w
                except Exception:
                    pass

    _assign(_DOC_STRUCT_TAGS, w_doc)
    _assign(_OTSL_STRUCT_TAGS, w_struct)
    _assign(_FCEL_TAGS, w_fcel)

    # loc tokens: <loc_0> … <loc_500>
    n_loc = 0
    for k in range(_LOC_RANGE):
        try:
            ids = tokenizer.encode(f"<loc_{k}>", add_special_tokens=False)
            if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                weights[ids[0]] = w_loc
                n_loc += 1
        except Exception:
            pass

    print(
        f"[DocTagLoss] Token weight map built:  vocab={vocab_size}  "
        f"loc_tokens_found={n_loc}/{_LOC_RANGE}"
    )
    return weights


def build_loc_token_table(tokenizer):
    """
    Build data structures needed for the differentiable loc regression term.

    Returns
    -------
    loc_ids : torch.LongTensor, shape (N_loc,)
        Vocabulary IDs for each discovered loc token, in k-order
        (loc_ids[i] is the token ID for <loc_{k_values[i]}> ).
    k_values : torch.FloatTensor, shape (N_loc,)
        Coordinate values [0, 1, 2, ..., 500] corresponding to each loc_id.
        Stored as float for use in the expectation sum.
    id_to_k : dict[int, float]
        Reverse mapping: vocabulary ID → coordinate value.
        Used to look up the true coordinate k_t from a ground-truth label.
    """
    loc_ids: list = []
    k_vals: list = []
    id_to_k: dict = {}
    vocab_size = len(tokenizer)

    for k in range(_LOC_RANGE):
        try:
            ids = tokenizer.encode(f"<loc_{k}>", add_special_tokens=False)
            if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                tok_id = ids[0]
                loc_ids.append(tok_id)
                k_vals.append(float(k))
                id_to_k[tok_id] = float(k)
        except Exception:
            pass

    if not loc_ids:
        # Fallback: vocabulary scan for tokens that decode as <loc_N>
        _loc_pat = re.compile(r'^<loc_(\d+)>$')
        for tok_id in range(vocab_size):
            try:
                s = tokenizer.decode([tok_id], skip_special_tokens=False).strip()
                m = _loc_pat.match(s)
                if m:
                    k = int(m.group(1))
                    loc_ids.append(tok_id)
                    k_vals.append(float(k))
                    id_to_k[tok_id] = float(k)
            except Exception:
                pass
        # Sort by coordinate value
        paired = sorted(zip(k_vals, loc_ids))
        if paired:
            k_vals_sorted, loc_ids_sorted = zip(*paired)
            loc_ids = list(loc_ids_sorted)
            k_vals = list(k_vals_sorted)
        print(f"[DocTagLoss] Loc token fallback scan: found {len(loc_ids)} tokens")

    return (
        torch.tensor(loc_ids,  dtype=torch.long),
        torch.tensor(k_vals,   dtype=torch.float32),
        id_to_k,
    )


def build_struct_token_ids(tokenizer) -> torch.LongTensor:
    """
    Return a LongTensor of all vocabulary IDs that correspond to structural
    tag tokens (both opening and closing variants).

    These IDs are used to build the boolean mask for the focal loss term.
    """
    ids: list = []
    seen: set = set()
    vocab_size = len(tokenizer)

    for tag in _ALL_STRUCT_TAGS:
        for variant in (f"<{tag}>", f"</{tag}>"):
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    tok_id = enc[0]
                    if 0 <= tok_id < vocab_size and tok_id not in seen:
                        ids.append(tok_id)
                        seen.add(tok_id)
            except Exception:
                pass

    return torch.tensor(ids, dtype=torch.long)


# ===========================================================================
# DocTagLoss — core loss module
# ===========================================================================

class DocTagLoss(nn.Module):
    """
    Composite differentiable loss for DocTag sequence prediction.

        L_total = L_wce + λ_focal · L_focal + λ_loc · L_loc

    All mathematical derivations are in the module docstring above.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    w_doc, w_struct, w_loc, w_fcel : float
        Per-category CE weights (see build_token_weight_map).
    lambda_loc : float
        Weight of the differentiable loc regression term (default 0.3).
    lambda_focal : float
        Weight of the focal loss term (default 0.5).
    focal_gamma : float
        Focal modulation exponent γ (default 2.0).

    Public attributes
    -----------------
    token_weights : torch.Tensor, shape (vocab_size,)
        Per-token CE weight vector. Exposed as a public attribute so external
        code (e.g. train_v7.patch_mongolian_weights) can modify it in-place
        after construction to upweight language-specific characters.
    """

    def __init__(
        self,
        tokenizer,
        w_doc: float = 20.0,
        w_struct: float = 8.0,
        w_loc: float = 4.0,
        w_fcel: float = 1.0,
        lambda_loc: float = 0.3,
        lambda_focal: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.lambda_loc   = lambda_loc
        self.lambda_focal = lambda_focal
        self.focal_gamma  = focal_gamma

        # --- Token weight map (registered buffer: moves with model.to(device)) ---
        # Exposed as self.token_weights for external patching.
        weights = build_token_weight_map(tokenizer, w_doc, w_struct, w_loc, w_fcel)
        self.register_buffer("token_weights", weights)

        # --- Loc token table ---
        loc_ids, k_values, id_to_k = build_loc_token_table(tokenizer)
        self.register_buffer("_loc_ids",    loc_ids)   # (N_loc,) token IDs
        self.register_buffer("_k_values",  k_values)  # (N_loc,) coordinate floats

        # id_to_k is a plain dict; used on CPU to build the true-coord tensor
        self._id_to_k: dict = id_to_k

        # --- Structural tag token IDs (for focal loss mask) ---
        struct_ids = build_struct_token_ids(tokenizer)
        self.register_buffer("_struct_ids", struct_ids)   # (N_struct,)

        # Pre-build a boolean lookup vector over the full vocabulary for
        # fast structural-position detection during forward:
        #   _is_struct[tok_id] = True  iff  tok_id ∈ struct_ids
        vocab_size = len(tokenizer)
        is_struct = torch.zeros(vocab_size, dtype=torch.bool)
        if len(struct_ids) > 0:
            is_struct[struct_ids] = True
        self.register_buffer("_is_struct", is_struct)     # (vocab_size,)

        # Same for loc tokens
        is_loc = torch.zeros(vocab_size, dtype=torch.bool)
        if len(loc_ids) > 0:
            is_loc[loc_ids] = True
        self.register_buffer("_is_loc", is_loc)           # (vocab_size,)

        print(
            f"[DocTagLoss] Ready.  "
            f"struct_tokens={len(struct_ids)}  "
            f"loc_tokens={len(loc_ids)}  "
            f"λ_focal={lambda_focal}  γ={focal_gamma}  "
            f"λ_loc={lambda_loc}  δ_huber={_HUBER_DELTA}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,   # (B, T, V)  raw model logits
        labels: torch.Tensor,   # (B, T)     ground-truth token IDs, -100 at masked positions
    ) -> dict:
        """
        Compute the composite DocTag loss.

        Returns a dict:
            "loss"       — L_total (scalar, used by Trainer)
            "loss_wce"   — weighted CE component (detached, for logging)
            "loss_focal" — focal component        (detached, for logging)
            "loss_loc"   — loc regression component (detached, for logging)
        """
        B, T, V = logits.shape
        device  = logits.device

        # Move buffers to current device (safety: handles accelerate placement)
        # Clamp token_weights to actual logits vocab size — guards against
        # off-by-one when len(tokenizer) != model.config.vocab_size (e.g. when
        # a checkpoint adapter added an extra special token that was never
        # embedded into the model matrix).
        token_weights = self.token_weights[:V].to(device)
        is_struct     = self._is_struct.to(device)
        is_loc        = self._is_loc.to(device)
        loc_ids       = self._loc_ids.to(device)
        k_values      = self._k_values.to(device)

        # ----------------------------------------------------------------
        # L_wce — weighted cross-entropy
        # F.cross_entropy expects (N, V) logits and (N,) labels.
        # ----------------------------------------------------------------
        logits_2d = logits.reshape(-1, V)        # (B*T, V)
        labels_1d = labels.reshape(-1)           # (B*T,)

        # Guard: if every position is masked, CE returns nan (mean of empty set).
        # Return zero loss for fully-masked batches to avoid NaN propagation.
        if not (labels_1d != -100).any():
            zero = logits_2d.new_zeros(1).squeeze()
            return {"loss": zero, "loss_wce": zero, "loss_focal": zero, "loss_loc": zero}

        l_wce = F.cross_entropy(
            logits_2d,
            labels_1d,
            weight=token_weights,
            ignore_index=-100,
            reduction="mean",
        )

        # ----------------------------------------------------------------
        # L_focal — focal loss at structural-tag positions
        # ----------------------------------------------------------------
        l_focal = self._focal_loss(logits_2d, labels_1d, is_struct, device)

        # ----------------------------------------------------------------
        # L_loc — differentiable Huber regression on loc tokens
        # ----------------------------------------------------------------
        l_loc = self._loc_regression_loss(logits_2d, labels_1d, is_loc,
                                          loc_ids, k_values, device)

        # ----------------------------------------------------------------
        # Combine
        # ----------------------------------------------------------------
        l_total = l_wce + self.lambda_focal * l_focal + self.lambda_loc * l_loc

        return {
            "loss":       l_total,
            "loss_wce":   l_wce.detach(),
            "loss_focal": l_focal.detach(),
            "loss_loc":   l_loc.detach(),
        }

    # ------------------------------------------------------------------
    # Internal sub-loss methods
    # ------------------------------------------------------------------

    def _focal_loss(
        self,
        logits_2d: torch.Tensor,   # (N, V)
        labels_1d: torch.Tensor,   # (N,)
        is_struct:  torch.Tensor,  # (V,) bool
        device:     torch.device,
    ) -> torch.Tensor:
        """
        Focal loss restricted to structural-tag positions.

            L_focal = mean_{t ∈ struct, y_t ≠ -100} [(1-p_t)^γ · (-log p_t)]

        Numerical approach:
            log_p_t = log_softmax(logits)[y_t]        # scalar in (-∞, 0]
            p_t      = exp(log_p_t)                   # scalar in (0, 1]
            focal_wt = (1 - p_t)^γ
            loss_t   = -focal_wt · log_p_t
        """
        if len(self._struct_ids) == 0 or self.lambda_focal == 0.0:
            return logits_2d.new_zeros(1).squeeze()

        # Active mask: valid label AND label is a structural tag
        valid_mask  = labels_1d != -100                       # (N,)
        labels_safe = labels_1d.clamp(min=0)                 # avoid OOB index for is_struct lookup
        struct_mask = is_struct[labels_safe] & valid_mask    # (N,)

        if not struct_mask.any():
            return logits_2d.new_zeros(1).squeeze()

        # Extract logits and labels at structural positions
        active_logits = logits_2d[struct_mask]               # (M, V)
        active_labels = labels_safe[struct_mask]             # (M,)

        # log p_t at the true structural token
        log_probs = F.log_softmax(active_logits, dim=-1)     # (M, V), stable
        log_p_t   = log_probs.gather(
            1, active_labels.unsqueeze(1)
        ).squeeze(1)                                          # (M,)

        # Focal modulation: (1 - p_t)^γ
        # p_t = exp(log_p_t); safe because log_p_t ≤ 0 → p_t ∈ (0, 1]
        p_t       = log_p_t.exp()                            # (M,)  ∈ (0, 1]
        focal_wt  = (1.0 - p_t).pow(self.focal_gamma)        # (M,)
        loss_vals = -focal_wt * log_p_t                      # (M,)

        return loss_vals.mean()

    def _loc_regression_loss(
        self,
        logits_2d: torch.Tensor,   # (N, V)
        labels_1d: torch.Tensor,   # (N,)
        is_loc:     torch.Tensor,  # (V,) bool
        loc_ids:    torch.Tensor,  # (N_loc,) vocabulary IDs of loc tokens in k-order
        k_values:   torch.Tensor,  # (N_loc,) coordinate floats in k-order
        device:     torch.device,
    ) -> torch.Tensor:
        """
        Differentiable Huber regression on bounding-box (loc) token positions.

        For each position t where y_t ∈ {loc_0, …, loc_500}:

            p_loc_t  = softmax(logits_t[loc_ids])        # (N_loc,) distribution over coords
            E_θ[loc] = Σ_k k · p_loc_t[k]               # soft expected coordinate ∈ [0, 500]
            k_true   = coordinate of <loc_{k_true}>      # true coordinate ∈ {0, …, 500}
            L_t      = Huber(E_θ[loc], k_true, δ=25)

        The gradient ∂L_t/∂logits_t flows through the softmax into all 501
        loc-token logit dimensions simultaneously, providing an ordinal gradient
        signal that standard CE cannot give.
        """
        if len(loc_ids) < 2 or self.lambda_loc == 0.0:
            return logits_2d.new_zeros(1).squeeze()

        # Active mask: valid label AND label is a loc token
        valid_mask  = labels_1d != -100                     # (N,)
        labels_safe = labels_1d.clamp(min=0)
        loc_mask    = is_loc[labels_safe] & valid_mask      # (N,)

        if not loc_mask.any():
            return logits_2d.new_zeros(1).squeeze()

        # Extract logits at loc positions
        active_logits = logits_2d[loc_mask]                  # (M, V)
        active_labels = labels_safe[loc_mask]                # (M,)  vocab IDs

        # Sub-logits over the loc-token sub-vocabulary only
        loc_logits = active_logits[:, loc_ids]               # (M, N_loc)

        # Soft expected coordinate: E_θ[loc] = Σ_k k · softmax(loc_logits)[k]
        loc_probs = F.softmax(loc_logits, dim=-1)            # (M, N_loc)
        E_coord   = (loc_probs * k_values.unsqueeze(0)).sum(dim=-1)  # (M,)

        # True coordinates: map vocab IDs → k values using id_to_k dict
        # We build a full vocab-sized lookup tensor on device for vectorised gather.
        # This tensor is constructed once per forward at loc positions only (cheap).
        id_to_k_tensor = torch.zeros(logits_2d.shape[1],
                                     dtype=torch.float32, device=device)
        id_to_k_tensor[loc_ids] = k_values                  # fill known loc positions
        true_coords = id_to_k_tensor[active_labels]          # (M,)

        # Huber regression: smooth for |E - k_true| ≤ δ, linear beyond
        l_loc = F.huber_loss(E_coord, true_coords,
                             reduction="mean", delta=_HUBER_DELTA)
        return l_loc


# ===========================================================================
# DocTagModelWithLoss — HuggingFace-compatible model wrapper
# ===========================================================================

class DocTagModelWithLoss(nn.Module):
    """
    Wraps a (PEFT/QLoRA) HuggingFace model with the DocTagLoss.

    The HuggingFace Trainer/SFTTrainer calls model(**batch) and reads
    outputs.loss for gradient computation.  This wrapper:

        1. Pops 'labels' from kwargs (avoids computing the default CE loss).
        2. Calls the underlying model to obtain logits.
        3. Computes DocTagLoss(logits, labels).
        4. Sets outputs.loss = L_total.
        5. Returns the (modified) model output so Trainer can read loss and
           logits normally.

    Attribute delegation via __getattr__ ensures that PEFT methods
    (save_pretrained, print_trainable_parameters, generate, …) all pass
    through to the wrapped model transparently.

    Stacking with HomoglyphAwareLoss (train_v7)
    -------------------------------------------
    HomoglyphAwareLoss wraps DocTagModelWithLoss and adds a contrastive
    term to outputs.loss after this class computes L_total.  The two
    wrappers form a composition chain:

        SFTTrainer → HomoglyphAwareLoss → DocTagModelWithLoss → PEFT model

    The HomoglyphAwareLoss reads outputs.logits (set by this class) to
    compute its contrastive term, so logits must be available in the output.
    """

    def __init__(self, model: nn.Module, loss_fn: DocTagLoss):
        super().__init__()
        self.base_model = model     # registered submodule: .to(device) propagates
        self.loss_fn    = loss_fn   # registered submodule: buffers move with model

    def forward(self, **kwargs) -> object:
        # Pop labels before the base model call — prevents it from computing
        # its own default CE loss (which we replace with DocTagLoss).
        labels = kwargs.pop("labels", None)

        # Forward pass through the underlying model (logits only, no default loss)
        outputs = self.base_model(**kwargs)

        if labels is None:
            # Inference mode — no loss needed
            return outputs

        # Extract logits (handle both attribute and tuple-output models)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        if logits is None:
            # Cannot compute loss — return outputs unchanged with a warning
            import warnings
            warnings.warn(
                "[DocTagModelWithLoss] Could not locate logits in model output. "
                "Skipping custom loss. Ensure the model returns outputs.logits."
            )
            return outputs
        if logits.dim() != 3:
            # Some generation-phase calls return 2D logits; skip custom loss
            return outputs

        # Compute composite DocTag loss
        loss_dict = self.loss_fn(logits, labels)
        total_loss = loss_dict["loss"]

        # Write custom loss back into the model output object.
        # ModelOutput.__setattr__ only syncs to the underlying dict when
        # the key already exists in .keys(); since the base model was called
        # *without* labels, outputs["loss"] is absent from the dict (None values
        # are excluded at construction time).  Using subscript assignment
        # (__setitem__) always updates both the dict and the Python attribute.
        try:
            outputs["loss"] = total_loss
        except (TypeError, AttributeError):
            # Fallback: wrap in a lightweight container that the Trainer accepts
            outputs = _DocTagOutputWrapper(
                loss=total_loss,
                logits=logits,
                original=outputs,
            )

        return outputs

    # ------------------------------------------------------------------
    # Attribute delegation to the wrapped model
    # Allows code that does model.save_pretrained(...), model.generate(...),
    # model.print_trainable_parameters(), etc. to work transparently.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        try:
            # First look in nn.Module's own registry (parameters, buffers, modules)
            return super().__getattr__(name)
        except AttributeError:
            # Then delegate to the wrapped model
            return getattr(self.base_model, name)


class _DocTagOutputWrapper:
    """
    Minimal output container for the rare case where outputs.loss is not
    directly assignable (e.g. frozen dataclass).  Provides .loss and .logits
    attribute access so the Trainer and HomoglyphAwareLoss can both read them.
    """
    __slots__ = ("loss", "logits", "_original")

    def __init__(self, loss, logits, original):
        self.loss      = loss
        self.logits    = logits
        self._original = original

    def __getattr__(self, name: str):
        return getattr(self._original, name)


# Python <3.11 doesn't have FrozenInstanceError as a built-in; define a
# fallback so the except clause above doesn't NameError.
try:
    FrozenInstanceError  # type: ignore[used-before-def]
except NameError:
    class FrozenInstanceError(AttributeError):  # type: ignore[no-redef]
        pass


# ===========================================================================
# Factory function
# ===========================================================================

def build_loss_fn(
    tokenizer,
    w_doc: float = 20.0,
    w_struct: float = 8.0,
    w_loc: float = 4.0,
    w_fcel: float = 1.0,
    lambda_loc: float = 0.3,
    lambda_focal: float = 0.5,
    focal_gamma: float = 2.0,
) -> DocTagLoss:
    """
    Construct and return a DocTagLoss instance.

    This is the primary entry point used by train_v6 and train_v7:

        loss_fn = build_loss_fn(
            tokenizer=processor.tokenizer,
            w_doc=20.0,
            w_struct=8.0,
            w_loc=4.0,
            w_fcel=1.0,
            lambda_loc=0.3,
            lambda_focal=0.5,
            focal_gamma=2.0,
        )
        model = DocTagModelWithLoss(peft_model, loss_fn)

    After construction, loss_fn.token_weights (shape: vocab_size) can be
    modified in-place for language-specific upweighting:

        # Mongolian-specific characters (train_v7 ADDITION-3):
        from train_v7 import patch_mongolian_weights
        patch_mongolian_weights(loss_fn.token_weights, tokenizer, w_mongolian=15.0)

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Must already have DocTag special tokens in its vocabulary (loc_0..500
        and all structural tag tokens).
    w_doc : float
        CE weight for document-structure open/close tags (default 20).
        Derivation: these tags receive 0.87% of unweighted CE gradient;
        ×20 brings them to ~17.4% — comparable to fcel at 32% × 1.0.
    w_struct : float
        CE weight for OTSL table-structure tags (default 8).
    w_loc : float
        CE weight for <loc_N> bounding-box tokens (default 4).
    w_fcel : float
        CE weight for <fcel> cell-content tokens (default 1.0).
    lambda_loc : float
        Weight of the Huber loc regression term in L_total (default 0.3).
    lambda_focal : float
        Weight of the focal loss term in L_total (default 0.5).
    focal_gamma : float
        Focal modulation exponent γ (default 2.0; standard value from Lin 2017).

    Returns
    -------
    DocTagLoss instance with:
        .token_weights    — public buffer, shape (vocab_size,), patchable in-place
        .lambda_loc       — float
        .lambda_focal     — float
        .focal_gamma      — float
    """
    return DocTagLoss(
        tokenizer=tokenizer,
        w_doc=w_doc,
        w_struct=w_struct,
        w_loc=w_loc,
        w_fcel=w_fcel,
        lambda_loc=lambda_loc,
        lambda_focal=lambda_focal,
        focal_gamma=focal_gamma,
    )
