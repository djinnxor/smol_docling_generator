"""Render a deterministic scorecard PNG for DocTags evaluations.

This module does not use any model output or random generation to draw the
image. It only consumes exact evaluation metrics (tag accuracy, CER, WER,
per-page scores, and expected/predicted tag counts) and renders them using
Matplotlib. The output is therefore reproducible for any model evaluation as
long as the same statistics are passed in.

One image per model, deliberately: the two models produce wildly different
magnitudes (a degenerate model can exceed CER 1.0 several times over, and emit
hundreds of spurious text blocks), so sharing an axis would squash the good
model flat. Each card is self-contained, and the expected/ground-truth counts
are drawn on every card so two cards can still be compared by eye.

Axes are linear throughout. A bar encodes magnitude by its length from zero,
so a log axis would break the encoding it depends on.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Palette: validated categorical slots 1-2, plus chrome/ink and status tokens.
# Status colours always ship with a word beside them, never colour alone.
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_MODEL = "#2a78d6"    # slot 1 - this model
SERIES_EXPECT = "#eb6834"   # slot 2 - ground truth
STATUS = {"good": "#0ca30c", "fair": "#fab219", "poor": "#d03b3b"}

REPORT_TAGS = [
    "title", "section_header_level_1", "text", "list_item", "page_header",
    "page_footer", "footnote", "otsl", "picture", "unordered_list",
    "checkbox_unselected",
]


def _band(value: float, good: float, fair: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "good" if value >= good else "fair" if value >= fair else "poor"
    return "good" if value <= good else "fair" if value <= fair else "poor"


def _tile(ax, label, value, direction, band, kind) -> None:
    """A stat tile: the number is the chart."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.015, 0.02), 0.97, 0.96,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1, edgecolor=GRID, facecolor=SURFACE,
        transform=ax.transAxes, clip_on=False))
    ax.text(0.08, 0.76, label, color=INK_2, fontsize=11, weight="medium")
    ax.text(0.08, 0.36, f"{value:.3f}", color=INK, fontsize=26, weight="bold")
    ax.text(0.08, 0.17, f"{kind} · {direction}", color=MUTED, fontsize=8.5)
    ax.plot([0.90], [0.82], marker="o", markersize=8, color=STATUS[band],
            transform=ax.transAxes, clip_on=False)
    ax.text(0.90, 0.68, band, color=INK_2, fontsize=8.5, ha="center")


def _style(ax, *, xlabel="", title=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, weight="bold",
                     loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def render_scorecard(name: str, agg: dict, scores: list[dict],
                     pages: list[int], path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 10.6), dpi=110, facecolor=PLANE)
    gs = fig.add_gridspec(
        3, 6, height_ratios=[0.62, 1.05, 1.30],
        hspace=0.62, wspace=0.28,
        left=0.075, right=0.965, top=0.875, bottom=0.105)

    fig.text(0.075, 0.950, f"DocTags evaluation — {name}",
             color=INK, fontsize=17, weight="bold")
    fig.text(0.075, 0.921,
             f"Zaavar.pdf · {len(pages)} pages · scored against hand-built ground truth",
             color=MUTED, fontsize=10)

    # ---- row 1: the four headline numbers -------------------------------
    tiles = [
        ("Tag accuracy", agg["tag_acc"], "higher is better",
         _band(agg["tag_acc"], 0.75, 0.45, True), "structure"),
        ("Tag precision", agg.get("precision", agg["tag_acc"]), "higher is better",
         _band(agg.get("precision", agg["tag_acc"]), 0.75, 0.45, True), "structure"),
        ("Tag count accuracy", agg.get("accuracy", agg["tag_acc"]), "higher is better",
         _band(agg.get("accuracy", agg["tag_acc"]), 0.75, 0.45, True), "structure"),
        ("Tree edit dist", agg["tree_edit"], "lower is better",
         _band(agg["tree_edit"], 0.20, 0.45, False), "structure"),
        ("CER", agg["cer"], "lower is better",
         _band(agg["cer"], 0.10, 0.30, False), "reading"),
        ("WER", agg["wer"], "lower is better",
         _band(agg["wer"], 0.20, 0.50, False), "reading"),
    ]
    for i, (lbl, val, dirn, band, kind) in enumerate(tiles):
        _tile(fig.add_subplot(gs[0, i]), lbl, val, dirn, band, kind)

    # ---- row 2: per page, structure and reading side by side ------------
    xs = list(range(len(pages)))
    labels = [f"page {p}" for p in pages]

    ax = fig.add_subplot(gs[1, :3])
    _style(ax, title="Structure per page — tag accuracy")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.bar(xs, [s["tag_acc"] for s in scores], width=0.52, color=SERIES_MODEL)
    for x, s in zip(xs, scores):
        ax.text(x, s["tag_acc"] + 0.035, f"{s['tag_acc']:.2f}",
                ha="center", color=INK_2, fontsize=9.5)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.6, len(pages) - 0.4)
    ax.set_ylim(0, 1.10)
    ax.text(-0.55, 1.04, "higher is better", color=MUTED, fontsize=8.5)

    ax = fig.add_subplot(gs[1, 3:])
    cers = [s["cer"] for s in scores]
    top = max(1.25, max(cers) * 1.30)
    _style(ax, title="Reading per page — CER")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.bar(xs, cers, width=0.52, color=SERIES_MODEL)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.55, 1.02, "1.0 = one error per character",
            color=MUTED, fontsize=8, ha="left", va="bottom")
    for x, c in zip(xs, cers):
        ax.text(x, c + top * 0.035, f"{c:.2f}",
                ha="center", color=INK_2, fontsize=9.5)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.6, len(pages) - 0.4)
    ax.set_ylim(0, top)
    ax.text(-0.55, top * 0.955, "lower is better", color=MUTED, fontsize=8.5)

    # ---- row 3: element counts, expected vs produced --------------------
    ax = fig.add_subplot(gs[2, :])
    tags = [t for t in REPORT_TAGS
            if agg["ref_tags"].get(t, 0) or agg["pred_tags"].get(t, 0)]
    exp = [agg["ref_tags"].get(t, 0) for t in tags]
    got = [agg["pred_tags"].get(t, 0) for t in tags]
    ys = list(range(len(tags)))
    off = 0.20
    hi = max(exp + got) or 1

    _style(ax, title="Element counts — what the document has vs what the model produced")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.barh([y - off for y in ys], exp, height=0.34, color=SERIES_EXPECT)
    ax.barh([y + off for y in ys], got, height=0.34, color=SERIES_MODEL)

    ax.set_yticks(ys, tags)
    ax.set_ylim(len(tags) - 0.45, -0.55)
    ax.set_xlim(0, hi * 1.20)

    pad = hi * 0.012
    for y, e, g in zip(ys, exp, got):
        ax.text(e + pad, y - off, str(e), va="center", color=INK_2, fontsize=9)
        note = f"{g}   ← missing" if (e and not g) else str(g)
        ax.text(g + pad, y + off, note, va="center", color=INK_2, fontsize=9)

    ax.legend(handles=[
        plt.Line2D([], [], color=SERIES_EXPECT, linewidth=7,
                   solid_capstyle="butt", label="expected (ground truth)"),
        plt.Line2D([], [], color=SERIES_MODEL, linewidth=7,
                   solid_capstyle="butt", label=f"produced by {name}"),
    ], loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2)

    fig.text(0.075, 0.020,
             "Tag accuracy, precision, and count accuracy measure STRUCTURE only.  "
             "CER and WER measure TEXT only.\nA change in one and not the other "
             "tells you which skill moved.",
             color=MUTED, fontsize=9, linespacing=1.5)

    fig.savefig(path, facecolor=PLANE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Render a DocTags evaluation scorecard from numeric statistics. "
            "No model output or AI generation is involved; the chart is produced "
            "directly from the metrics passed in."
        )
    )
    ap.add_argument("--model", required=True, help="Model name shown on the chart")
    ap.add_argument("--tag-acc", type=float, required=True, help="Tag accuracy")
    ap.add_argument("--tree-edit", type=float, required=True, help="Tree edit distance")
    ap.add_argument("--cer", type=float, required=True, help="CER")
    ap.add_argument("--wer", type=float, required=True, help="WER")
    ap.add_argument("--out", required=True, help="Destination PNG path")
    ap.add_argument("--pages", nargs="+", type=int, default=[1],
                    help="Pages represented in the per-page chart")
    ap.add_argument("--ref-tags", nargs="*", default=[],
                    help="Ground-truth tag counts as key=value pairs, e.g. title=3 text=10")
    ap.add_argument("--pred-tags", nargs="*", default=[],
                    help="Predicted tag counts as key=value pairs, e.g. title=3 text=9")
    args = ap.parse_args()

    def _parse_tag_counts(items):
        counts = {}
        for item in items:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"Invalid tag count '{item}'. Expected key=value.")
            counts[key] = int(value)
        return counts

    ref_tags = _parse_tag_counts(args.ref_tags)
    pred_tags = _parse_tag_counts(args.pred_tags)
    scores = [{"tag_acc": args.tag_acc, "cer": args.cer} for _ in args.pages]
    agg = {
        "tag_acc": args.tag_acc,
        "tree_edit": args.tree_edit,
        "cer": args.cer,
        "wer": args.wer,
        "ref_tags": ref_tags,
        "pred_tags": pred_tags,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_scorecard(args.model, agg, scores, args.pages, out)
    print(f"scorecard written: {out}")


if __name__ == "__main__":
    main()
