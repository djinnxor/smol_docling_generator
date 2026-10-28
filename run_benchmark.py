"""
Run every model registered in the MLOps project over a benchmark document and
score them together.

The roster is read from the project file rather than hard-coded, so a model
registered later is picked up without editing anything here.  Each experiment
node contributes its ``output_dir``; the base model is added separately because
it has no training run of its own.

Outputs follow the layout the earlier runs established:

    output/<key>/<key>-<dataset>.json    predictions, as inference.py writes
    output/<key>/<key>-<dataset>.pdf     the rendered pages
    reports/<dataset>/eval_<key>.png     per-model scorecard
    reports/<dataset>/scores.json        every model, doctag-eval/1 schema

That scores.json is what the dashboard imports, so one run produces both the
artefacts to look at and the numbers to compare.

    # everything, then import the scores into the project
    python run_benchmark.py --import-mlops

    # one model, re-running even though its output exists
    python run_benchmark.py --only exp-008 --force

Inference is skipped for a model whose predictions are already on disk, so an
interrupted sweep resumes where it stopped.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = "/home/rookie/mlops_docs/granite-docling.json"
BASE_MODEL = "ibm-granite/granite-docling-258M"

# Experiments that share an output_dir describe the same weights: the run and
# the import of its result.  Keep the one whose key matches the existing
# evaluation rows so a new report lines up with the old ones.
PREFER = {"doctag_restart_textonly": "exp-002", "updated_table": "exp-010"}


def load_roster(project: str, include_base: bool = True) -> list[dict]:
    """Every registered model with usable weights on disk, newest id last."""
    data = json.loads(Path(project).read_text(encoding="utf-8"))
    by_dir: dict[str, list[dict]] = {}
    for node in data.get("nodes", []):
        if node.get("type") != "experiment":
            continue
        out = (node.get("data", {}).get("parameters", {}) or {}).get("output_dir", "")
        if out:
            by_dir.setdefault(out, []).append(node)

    roster = []
    if include_base:
        roster.append({"key": "base", "node": None, "path": BASE_MODEL,
                       "title": BASE_MODEL})

    for out, nodes in sorted(by_dir.items()):
        ids = [n["id"] for n in nodes]
        pick = next((n for n in nodes if n["id"] == PREFER.get(out)), nodes[-1])
        path = ROOT / out
        weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
        # A directory of checkpoints with no merged model at the top is a run
        # that has not been finalised.  Resolve which checkpoint to use at the
        # moment the model runs, not now: a training job writing into the same
        # directory rotates old checkpoints away, so a path picked here can be
        # deleted before its turn comes -- which is exactly what happened to
        # updated_table on the first sweep.
        latest = not weights
        if latest and not any(p.is_dir() for p in path.glob("checkpoint-*")):
            print(f"  SKIP {','.join(ids):18} {out}: no weights found",
                  file=sys.stderr)
            continue
        roster.append({
            "key": pick["id"], "node": pick["id"], "path": str(path),
            "latest_checkpoint": latest,
            "title": pick.get("title", ""), "shared_with": ids if len(ids) > 1 else [],
        })
    return roster


def resolve_path(model: dict) -> str | None:
    """Where this model's weights are, as of right now."""
    if not model.get("latest_checkpoint"):
        return model["path"]
    ckpts = sorted((p for p in Path(model["path"]).glob("checkpoint-*") if p.is_dir()),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        print(f"  {model['key']:9} no checkpoint left in {model['path']}",
              file=sys.stderr, flush=True)
        return None
    return str(ckpts[-1])


def eval_name(key: str) -> str:
    """The short label the earlier reports use: exp-008 -> 008, base -> base."""
    return key.split("-")[-1] if key.startswith("exp-") else key


def trim_pdf(pdf: str, first: int, last: int, dest: Path) -> Path:
    """A PDF of just pages first..last, built with poppler.

    Only the labelled pages are worth inferring: an unlabelled page cannot be
    scored, and on this document the unlabelled half is scanned, which is
    exactly where a model generates until it hits the token cap.  Running the
    whole file costs more than twice the GPU time for output nothing can grade.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    stem = dest.parent / f".{dest.stem}-part"
    subprocess.run(["pdfseparate", "-f", str(first), "-l", str(last), pdf,
                    f"{stem}-%d.pdf"], check=True)
    parts = sorted(stem.parent.glob(f"{stem.name}-*.pdf"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    subprocess.run(["pdfunite", *[str(p) for p in parts], str(dest)], check=True)
    for p in parts:
        p.unlink()
    return dest


def renumber(pred: Path, offset: int) -> None:
    """Shift predicted page numbers so they line up with the label pages.

    Inference on a trimmed PDF numbers its pages from 1; the labels keep the
    numbering of the full document.  eval_doctags.py matches on the page
    field, so without this every page would score as missing.
    """
    rows = json.loads(pred.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return
    if min(int(r.get("page", 1)) for r in rows) > offset:
        return                                  # already renumbered
    for r in rows:
        r["page"] = int(r.get("page", 1)) + offset
    pred.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def run_inference(model: dict, pdf: str, dataset: str, extra: list[str],
                  force: bool) -> Path | None:
    out_dir = ROOT / "output" / model["key"]
    pred = out_dir / f"{model['key']}-{dataset}.json"
    render = out_dir / f"{model['key']}-{dataset}.pdf"
    if pred.exists() and not force:
        print(f"  {model['key']:9} predictions already present, skipping", flush=True)
        return pred
    out_dir.mkdir(parents=True, exist_ok=True)

    path = resolve_path(model)
    if path is None:
        return None
    is_adapter = (Path(path) / "adapter_config.json").exists()
    cmd = [sys.executable, str(ROOT / "inference.py"),
           "--input", pdf,
           "--checkpoint" if is_adapter else "--model_id", path,
           "--output", str(pred),
           "--render_output", str(render)] + extra

    print(f"  {model['key']:9} {path}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  {model['key']:9} FAILED (exit {result.returncode})",
              file=sys.stderr, flush=True)
        return None
    print(f"  {model['key']:9} done in {time.time() - started:.0f}s -> {pred}",
          flush=True)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score every registered model on a benchmark document")
    ap.add_argument("--pdf", default="/home/rookie/Downloads/merged_benchmark.pdf")
    ap.add_argument("--labels", default="labels/benchmark_labels.jsonl")
    ap.add_argument("--dataset", default="benchmark",
                    help="Names the output files and the reports/<dataset>/ folder")
    ap.add_argument("--pages", default="16-28",
                    help="Page range to infer, matching the labelled pages. "
                         "Predictions are renumbered back to the full "
                         "document's page numbers. Use 'all' for every page.")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--only", action="append", default=[],
                    help="Run just these keys (exp-008, base). Repeatable.")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run inference even where predictions exist")
    ap.add_argument("--import-mlops", action="store_true",
                    help="Add the scores to the MLOps project as an evaluation")
    ap.add_argument("--inference-args", default="",
                    help="Extra flags passed through to inference.py, e.g. "
                         "\"--device cuda --max_new_tokens 2048\"")
    args = ap.parse_args()

    labels = ROOT / args.labels
    if not labels.exists():
        raise SystemExit(f"{labels}: no labels. Run labels/make_benchmark_labels.py")

    roster = load_roster(args.project, include_base=not args.skip_base)
    if args.only:
        roster = [m for m in roster if m["key"] in args.only]
    if not roster:
        raise SystemExit("no models to run")

    print(f"{len(roster)} model(s) on {Path(args.pdf).name}:")
    for m in roster:
        shared = f"  (shares weights with {', '.join(m['shared_with'])})" \
            if m.get("shared_with") else ""
        print(f"  {m['key']:9} {m['path']}{shared}")

    pdf = args.pdf
    offset = 0
    if args.pages != "all":
        first, _, last = args.pages.partition("-")
        first, last = int(first), int(last or first)
        offset = first - 1
        pdf = str(trim_pdf(args.pdf, first, last,
                           ROOT / "output" / f"_{args.dataset}_p{first}-{last}.pdf"))
        print(f"\ninferring pages {first}-{last} only ({pdf})")

    extra = args.inference_args.split()
    print("\ninference")
    preds: list[tuple[str, Path]] = []
    for m in roster:
        pred = run_inference(m, pdf, args.dataset, extra, args.force)
        if pred and pred.exists():
            if offset:
                renumber(pred, offset)
            preds.append((eval_name(m["key"]), pred))

    if not preds:
        raise SystemExit("no predictions were produced; nothing to score")

    reports = ROOT / "reports" / args.dataset
    reports.mkdir(parents=True, exist_ok=True)
    scores = reports / "scores.json"

    cmd = [sys.executable, str(ROOT / "eval_doctags.py"),
           "--labels", str(labels), "--charts", str(reports), "--json", str(scores),
           "--names", ",".join(n for n, _ in preds)]
    for _, p in preds:
        cmd += ["--pred", str(p)]

    print(f"\nscoring {len(preds)} model(s)", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit("evaluation failed")

    # Record where each model's artefacts live, next to its scores.  The
    # dashboard shows a row's metadata as chips that open the file, so this is
    # what turns a number in the table into the page you can actually look at.
    report = json.loads(scores.read_text(encoding="utf-8"))
    for name, pred in preds:
        entry = report.get("models", {}).get(name)
        if entry is None:
            continue
        render = pred.with_suffix(".pdf")
        card = reports / f"eval_{name}.png"
        if render.exists():
            entry["render"] = str(render.relative_to(ROOT))
        if card.exists():
            entry["scorecard"] = str(card.relative_to(ROOT))
    report["document"] = str(Path(args.pdf))
    report["pages"] = args.pages
    scores.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")

    print(f"\n  predictions  output/<key>/<key>-{args.dataset}.json")
    print(f"  renders      output/<key>/<key>-{args.dataset}.pdf")
    print(f"  scorecards   {reports}/eval_<key>.png")
    print(f"  scores       {scores}")

    if args.import_mlops:
        cmd = [sys.executable, str(ROOT / "import_eval_to_mlops.py"),
               "--scores", str(scores), "--project", args.project,
               "--name", args.dataset]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit("import into the MLOps project failed")


if __name__ == "__main__":
    main()
