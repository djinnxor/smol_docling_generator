"""
Import an eval_doctags.py scores.json into the MLOps project so the dashboard
shows it.

The dashboard already reads any report in the doctag-eval/1 schema and renders
every stored report, so nothing in the dashboard needs changing to add a new
benchmark -- the report declares its own metrics and arrives as another table.
This script is the one missing step: putting the file into the project
document and linking each row to the experiment that produced it.

Re-importing under the same name replaces the previous copy rather than
stacking a second one, so re-running a benchmark refreshes the table and any
model-to-node links you fixed by hand are preserved.

    python import_eval_to_mlops.py --scores reports/benchmark/scores.json \
                                   --name benchmark

    # see what it would do without writing
    python import_eval_to_mlops.py --scores reports/benchmark/scores.json --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

MLOPS_PACKAGE = "/home/rookie/Documents/Minje/mlops"
DEFAULT_PROJECT = "/home/rookie/mlops_docs/granite-docling.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add a doctag-eval report to the MLOps project document")
    ap.add_argument("--scores", default="reports/benchmark/scores.json")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help="Project JSON to update")
    ap.add_argument("--name", default=None,
                    help="Report name shown on the dashboard "
                         "(default: the scores file's parent folder)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the resulting table without writing anything")
    args = ap.parse_args()

    scores = Path(args.scores)
    if not scores.exists():
        raise SystemExit(f"{scores}: no scores file. Run run_benchmark.py first.")

    if MLOPS_PACKAGE not in sys.path:
        sys.path.insert(0, MLOPS_PACKAGE)
    try:
        from mlops.store import load_project, save_project
    except ImportError as exc:
        raise SystemExit(
            f"the mlops package is not importable from {MLOPS_PACKAGE}: {exc}")

    project = Path(args.project)
    root, slug = project.parent, project.stem
    name = args.name or scores.parent.name

    writer = load_project(slug, root)
    before = {e["id"] for e in writer.evaluations}
    report = writer.add_evaluation(scores, name=name)

    verb = "replaced" if report["id"] in before else "added"
    linked = sum(1 for row in report["rows"] if row.get("node"))
    print(f"{verb} {report['id']} \"{report['name']}\": "
          f"{len(report['rows'])} model(s), {len(report['columns'])} metric(s), "
          f"{linked} linked to an experiment")

    for row in report["rows"]:
        link = row["node"] or "-- unlinked, link it on the dashboard"
        print(f"  {row['key']:8} -> {link}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    path = save_project(writer, root)
    print(f"\nwrote {path}")
    print("Open the dashboard with:  mlops-web")


if __name__ == "__main__":
    main()
