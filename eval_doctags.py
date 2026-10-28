"""
Score DocTags predictions against ground-truth labels, reporting structure
(tags) and reading (text) accuracy as two independent numbers.

The two are deliberately measured in isolation so a regression can be
attributed.  A model that has forgotten tables but still reads Cyrillic
cleanly scores badly on TAG and well on TEXT; a model that has drifted into
gibberish but keeps the right skeleton does the opposite.

    # score one model
    python eval_doctags.py --labels labels/zaavar_labels.jsonl \
                           --pred finetuned.json

    # compare fine-tuned against the base model side by side
    python eval_doctags.py --labels labels/zaavar_labels.jsonl \
                           --pred base.json --pred finetuned.json \
                           --names base,finetuned

    # write the scores to a file for plotting / diffing
    python eval_doctags.py --labels labels/zaavar_labels.jsonl \
                           --pred base.json --pred finetuned.json \
                           --names base,finetuned --json reports/scores.json

Predictions are the .json written by `inference.py --output pred.json`.
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from doctag_metrics import (
    _TAG_RE,
    extract_structural_tags,
    tag_accuracy,
    tag_count_accuracy,
    tag_precision,
    tree_edit_dist,
)

# Element tags whose presence/absence is a structural fact worth reporting on
# its own line, so "tables disappeared" is visible rather than averaged away.
REPORT_TAGS = [
    "title", "section_header_level_1", "text", "list_item",
    "page_header", "page_footer", "footnote", "otsl", "picture",
    "unordered_list", "ordered_list", "checkbox_unselected", "checkbox_selected",
]


# ---------------------------------------------------------------- extraction

def strip_tags(doctag: str) -> str:
    """Everything the model claims to have read, with markup removed.

    Only *recognised* markup is stripped.  A generic <[^>]*> would run from any
    literal '<' in the page text to the next '>', eating both the real closing
    tag and everything between -- which silently deletes content on documents
    full of inequalities ("МНЗ<ӨЭК", "А1<П1") and inflates CER for both the
    reference and the prediction.
    """
    return " ".join(_TAG_RE.sub(" ", doctag).split())


def tag_counts(doctag: str) -> Counter:
    """How many of each structural element the document contains."""
    return Counter(t for t in extract_structural_tags(doctag)
                   if not t.startswith("/"))


# ------------------------------------------------------------------ distance

def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over two sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[len(b)]


def cer(ref: str, hyp: str) -> float:
    """Character error rate.  0.0 is a perfect read, 1.0 is total failure."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(list(ref), list(hyp)) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word error rate over whitespace-separated tokens."""
    r = ref.split()
    if not r:
        return 0.0 if not hyp.split() else 1.0
    return _edit_distance(r, hyp.split()) / len(r)


# -------------------------------------------------------------------- scoring

def score_page(ref_doctag: str, pred_doctag: str) -> dict:
    ref_text, pred_text = strip_tags(ref_doctag), strip_tags(pred_doctag)
    precision = tag_precision(pred_doctag, ref_doctag)
    accuracy = tag_count_accuracy(pred_doctag, ref_doctag)
    return {
        # structure
        "tag_acc": tag_accuracy(pred_doctag, ref_doctag),
        "accuracy": accuracy,
        "precision": precision,
        "tree_edit": tree_edit_dist(pred_doctag, ref_doctag),
        # reading
        "cer": cer(ref_text, pred_text),
        "wer": wer(ref_text, pred_text),
        "ref_chars": len(ref_text),
        "pred_chars": len(pred_text),
        "ref_tags": tag_counts(ref_doctag),
        "pred_tags": tag_counts(pred_doctag),
    }


def load(path: str) -> dict[int, str]:
    """Read a predictions .json or a labels .jsonl into {page: doctag}."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]
    return {int(r.get("page", i + 1)): r["doctag"] for i, r in enumerate(rows)}


def aggregate(pages: list[dict]) -> dict:
    """Average the per-page rates, weighting text rates by reference length.

    Weighting matters: a near-empty page would otherwise pull the corpus CER
    around as hard as a dense one.
    """
    total_ref = sum(p["ref_chars"] for p in pages) or 1
    ref_tags, pred_tags = Counter(), Counter()
    for p in pages:
        ref_tags += p["ref_tags"]
        pred_tags += p["pred_tags"]
    return {
        "tag_acc": sum(p["tag_acc"] for p in pages) / len(pages),
        "accuracy": sum(p["accuracy"] for p in pages) / len(pages),
        "precision": sum(p["precision"] for p in pages) / len(pages),
        "tree_edit": sum(p["tree_edit"] for p in pages) / len(pages),
        "cer": sum(p["cer"] * p["ref_chars"] for p in pages) / total_ref,
        "wer": sum(p["wer"] * p["ref_chars"] for p in pages) / total_ref,
        "ref_tags": ref_tags,
        "pred_tags": pred_tags,
    }


# --------------------------------------------------------------------- report

def _bar(value: float, width: int = 18) -> str:
    filled = max(0, min(width, round(value * width)))
    return "█" * filled + "·" * (width - filled)


def report(names: list[str], results: list[dict], per_page: list[list[dict]],
           pages: list[int]) -> None:
    w = max(12, max(len(n) for n in names) + 2)

    print("\n" + "=" * (24 + w * len(names)))
    print("STRUCTURE (tags)      " + "".join(f"{n:>{w}}" for n in names))
    print("=" * (24 + w * len(names)))
    print(f"{'tag accuracy ↑':<22}" +
          "".join(f"{r['tag_acc']:>{w}.3f}" for r in results))
    print(f"{'tag precision ↑':<22}" +
          "".join(f"{r['precision']:>{w}.3f}" for r in results))
    print(f"{'tag count accuracy ↑':<22}" +
          "".join(f"{r['accuracy']:>{w}.3f}" for r in results))
    print(f"{'tree edit dist ↓':<22}" +
          "".join(f"{r['tree_edit']:>{w}.3f}" for r in results))

    print("\n" + "-" * (24 + w * len(names)))
    print("element counts        " + f"{'expected':>{w}}" +
          "".join(f"{n:>{w}}" for n in names))
    print("-" * (24 + w * len(names)))
    ref = results[0]["ref_tags"]
    for tag in REPORT_TAGS:
        exp = ref.get(tag, 0)
        got = [r["pred_tags"].get(tag, 0) for r in results]
        if exp == 0 and not any(got):
            continue
        flag = "  <-- MISSING" if exp and not any(got) else ""
        print(f"{tag:<22}{exp:>{w}}" + "".join(f"{g:>{w}}" for g in got) + flag)

    print("\n" + "=" * (24 + w * len(names)))
    print("READING (text)        " + "".join(f"{n:>{w}}" for n in names))
    print("=" * (24 + w * len(names)))
    print(f"{'CER ↓':<22}" + "".join(f"{r['cer']:>{w}.3f}" for r in results))
    print(f"{'WER ↓':<22}" + "".join(f"{r['wer']:>{w}.3f}" for r in results))

    print("\nper page")
    for i, pg in enumerate(pages):
        print(f"  page {pg}")
        for n, pp in zip(names, per_page):
            s = pp[i]
            print(f"    {n:<14} tag {s['tag_acc']:.3f} {_bar(s['tag_acc'])}"
                  f"   CER {s['cer']:.3f} {_bar(1 - min(1, s['cer']))}")

    # Attribute any regression when two models are compared.
    if len(results) == 2:
        a, b = results
        dtag = b["tag_acc"] - a["tag_acc"]
        dcer = a["cer"] - b["cer"]          # positive = second model reads better
        print("\n" + "=" * 40)
        print(f"VERDICT  ({names[1]} vs {names[0]})")
        print("=" * 40)
        print(f"  structure  {dtag:+.3f} tag accuracy   "
              f"{'better' if dtag > 0.01 else 'worse' if dtag < -0.01 else 'unchanged'}")
        print(f"  reading    {dcer:+.3f} CER (lower=better) "
              f"{'better' if dcer > 0.01 else 'worse' if dcer < -0.01 else 'unchanged'}")
        lost = [t for t in REPORT_TAGS
                if a["pred_tags"].get(t, 0) > 0 and b["pred_tags"].get(t, 0) == 0]
        if lost:
            print(f"  element types {names[1]} stopped emitting: {', '.join(lost)}")


# ----------------------------------------------------------------- json export

# Each metric carries its own direction so a reader -- or a chart -- never has
# to guess whether a bigger number is a better one.
METRIC_META = {
    "structure": {
        "tag_accuracy": ("tag accuracy", True),
        "tag_precision": ("tag precision", True),
        "tag_count_accuracy": ("tag count accuracy", True),
        "tree_edit_distance": ("tree edit distance", False),
    },
    "reading": {
        "cer": ("character error rate", False),
        "wer": ("word error rate", False),
    },
}

# Aggregate key -> exported key.  Element counts (ref_tags/pred_tags) and the
# per-page scores are deliberately left out: this file is the summary.
_EXPORT_KEYS = {
    "structure": [("tag_acc", "tag_accuracy"),
                  ("precision", "tag_precision"),
                  ("accuracy", "tag_count_accuracy"),
                  ("tree_edit", "tree_edit_distance")],
    "reading": [("cer", "cer"), ("wer", "wer")],
}


def build_export(labels_path: str, pages: list[int], names: list[str],
                 pred_paths: list[str], results: list[dict],
                 missing: list[list[int]]) -> dict:
    models: dict[str, dict] = {}
    for name, pred_path, agg, absent in zip(names, pred_paths, results, missing):
        # Two --pred files can stem to the same name; keep both rather than
        # letting the second silently overwrite the first.
        key, n = name, 2
        while key in models:
            key, n = f"{name}#{n}", n + 1
        models[key] = {
            "predictions": pred_path,
            # Pages absent from this model's predictions were still averaged
            # in, scored as an empty page -- page_count is the denominator.
            "pages_missing": absent,
            **{group: {out: round(agg[src], 4) for src, out in keys}
               for group, keys in _EXPORT_KEYS.items()},
        }
    return {
        "schema": "doctag-eval/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": labels_path,
        "page_count": len(pages),
        "metrics": {
            group: {k: {"label": label, "higher_is_better": up}
                    for k, (label, up) in group_meta.items()}
            for group, group_meta in METRIC_META.items()
        },
        "models": models,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score DocTags predictions on structure and text separately")
    ap.add_argument("--labels", required=True,
                    help="Ground-truth .jsonl (e.g. labels/zaavar_labels.jsonl)")
    ap.add_argument("--pred", action="append", required=True,
                    help="Predictions .json from inference.py --output. "
                         "Repeat to compare models side by side.")
    ap.add_argument("--names", default="",
                    help="Comma-separated column labels, one per --pred")
    ap.add_argument("--json", metavar="PATH",
                    help="Also write the aggregate scores per model to a JSON "
                         "file (summary only -- no per-page or element counts)")
    ap.add_argument("--charts", metavar="DIR",
                    help="Also render a scorecard PNG per model into DIR "
                         "(one image per model, named eval_<model>.png)")
    args = ap.parse_args()

    labels = load(args.labels)
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(names) != len(args.pred):
        names = [Path(p).stem for p in args.pred]

    preds = [load(p) for p in args.pred]
    pages = sorted(labels)

    per_page, results, missing_pages = [], [], []
    for pred in preds:
        scores = [score_page(labels[pg], pred.get(pg, "")) for pg in pages]
        missing = [pg for pg in pages if pg not in pred]
        if missing:
            print(f"WARNING: pages absent from predictions, scored as empty: {missing}")
        per_page.append(scores)
        results.append(aggregate(scores))
        missing_pages.append(missing)

    report(names, results, per_page, pages)

    if args.json:
        out = Path(args.json)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        export = build_export(args.labels, pages, names, args.pred, results,
                              missing_pages)
        out.write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"\n  scores -> {out}")

    if args.charts:
        from doctag_charts import render_scorecard

        out = Path(args.charts)
        out.mkdir(parents=True, exist_ok=True)
        for name, agg, scores in zip(names, results, per_page):
            path = out / f"eval_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.png"
            render_scorecard(name, agg, scores, pages, path)
            print(f"  chart -> {path}")


if __name__ == "__main__":
    main()
