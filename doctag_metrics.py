"""DocTag comparison metrics.

    from doctag_metrics import tag_accuracy, loc_mae, tree_edit_dist
"""

import re
from collections import Counter

_STRUCTURAL_TAGS = frozenset({
    "doctag", "text", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "title", "caption", "picture", "otsl",
    "list_item", "formula", "page_header", "page_footer", "code", "footnote",
    # granite-docling also emits these; leaving them out silently discarded
    # real structure the model produced (list wrappers, form checkboxes) and
    # made the tag scores flatter than the models actually were.
    "unordered_list", "ordered_list",
    "checkbox_unselected", "checkbox_selected",
})

# OTSL cell tags, plus every structural tag above.  Matching only known names
# matters: this corpus contains inequalities like "МНЗ<ӨЭК" and "А1<П1", and a
# generic <[^>]*> pattern swallows from that '<' to the next '>', destroying
# both the real closing tag and the text in between.
_CELL_TAGS = frozenset({"fcel", "ched", "rhed", "ecel", "nl", "srow",
                        "stamp", "signature"})
_KNOWN_TAGS = _STRUCTURAL_TAGS | _CELL_TAGS
_TAG_RE = re.compile(
    r"</?(?:" + "|".join(sorted(_KNOWN_TAGS, key=len, reverse=True)) + r")>"
    r"|<loc_\d+>")


def extract_structural_tags(s: str) -> list[str]:
    """Return ordered list of structural open/close tags from a DocTag string."""
    return [
        ("/" if slash else "") + tag
        for slash, tag in re.findall(
            r'<(/?)(' + "|".join(sorted(_KNOWN_TAGS, key=len, reverse=True)) + r')>', s)
        if tag in _STRUCTURAL_TAGS
    ]


def tag_accuracy(pred: str, ref: str) -> float:
    """Positional accuracy of structural tags (open + close) between pred and ref."""
    pred_tags = extract_structural_tags(pred)
    ref_tags = extract_structural_tags(ref)
    if not ref_tags and not pred_tags:
        return 1.0
    if not ref_tags or not pred_tags:
        return 0.0
    matches = sum(p == r for p, r in zip(pred_tags, ref_tags))
    return matches / max(len(pred_tags), len(ref_tags))


def tag_precision(pred: str, ref: str) -> float:
    """Precision over structural tag counts. Higher is better; 1.0 means no extra tags."""
    pred_counts = Counter(tag for tag in extract_structural_tags(pred) if not tag.startswith("/"))
    ref_counts = Counter(tag for tag in extract_structural_tags(ref) if not tag.startswith("/"))
    all_tags = set(ref_counts) | set(pred_counts)
    tp = sum(min(pred_counts.get(tag, 0), ref_counts.get(tag, 0)) for tag in all_tags)
    fp = sum(max(pred_counts.get(tag, 0) - ref_counts.get(tag, 0), 0) for tag in all_tags)
    denom = tp + fp
    return 1.0 if denom == 0 else tp / denom


def tag_count_accuracy(pred: str, ref: str) -> float:
    """Count-level tag accuracy based on absolute mismatch across structural tag types."""
    pred_counts = Counter(tag for tag in extract_structural_tags(pred) if not tag.startswith("/"))
    ref_counts = Counter(tag for tag in extract_structural_tags(ref) if not tag.startswith("/"))
    all_tags = set(ref_counts) | set(pred_counts)
    if not all_tags:
        return 1.0
    total = sum(max(ref_counts.get(tag, 0), pred_counts.get(tag, 0)) for tag in all_tags)
    if total == 0:
        return 1.0
    mismatches = sum(abs(pred_counts.get(tag, 0) - ref_counts.get(tag, 0)) for tag in all_tags)
    return 1.0 - (mismatches / total)


def loc_mae(pred: str, ref: str) -> float:
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


def tree_edit_dist(pred: str, ref: str) -> float:
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
            for slash, tag in re.findall(
                    r'<(/?)(' + "|".join(sorted(_KNOWN_TAGS, key=len, reverse=True)) + r')>', s):
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

        pred_tree, ref_tree = _build(pred), _build(ref)
        dist = zss.simple_distance(_to_zss(pred_tree), _to_zss(ref_tree))
        max_dist = _size(pred_tree) + _size(ref_tree)
        return dist / max_dist if max_dist > 0 else 0.0

    except ImportError:
        # Fallback: tag-sequence normalised edit distance
        pred_tags = extract_structural_tags(pred)
        ref_tags = extract_structural_tags(ref)
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

__all__ = [
    "extract_structural_tags",
    "tag_accuracy",
    "tag_precision",
    "tag_count_accuracy",
    "loc_mae",
    "tree_edit_dist",
]
