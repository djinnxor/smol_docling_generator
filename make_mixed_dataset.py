"""
    # 25% of the mixed set is text (default)
    python make_mixed_dataset.py

    # sweep the requested 20-30% band
    python make_mixed_dataset.py --ratio 0.20 --out-prefix mixed20
    python make_mixed_dataset.py --ratio 0.30 --out-prefix mixed30
"""

import argparse
import json
import random
import sys
from pathlib import Path

REQUIRED_KEYS = {"image", "doctag"}


def read_jsonl(path: str) -> list[dict]:
    """Load a .jsonl, failing loudly on malformed or incomplete rows."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: malformed JSON ({exc})") from exc
            missing = REQUIRED_KEYS - row.keys()
            if missing:
                raise SystemExit(
                    f"{path}:{lineno}: row is missing {sorted(missing)}; "
                    f"expected keys {sorted(REQUIRED_KEYS)}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows found")
    return rows


def write_jsonl(rows: list[dict], path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists -- refusing to overwrite. "
            f"Pass --force, or choose another --out-prefix.")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_quota(n_table: int, n_text_available: int, ratio: float, mode: str,
               split: str) -> int:
    """How many text examples to draw, under either reading of the ratio."""
    if mode == "share":
        # text / (table + text) == ratio   ->   text = table * r/(1-r)
        want = round(n_table * ratio / (1.0 - ratio))
    else:  # fraction of the available text pool
        want = round(n_text_available * ratio)

    if want > n_text_available:
        print(f"WARNING [{split}]: wanted {want} text examples but only "
              f"{n_text_available} are available; using all of them.",
              file=sys.stderr)
        want = n_text_available
    return want


def build_split(table_rows: list[dict], text_rows: list[dict], ratio: float,
                mode: str, seed: int, split: str, tag: bool) -> list[dict]:
    quota = text_quota(len(table_rows), len(text_rows), ratio, mode, split)

    # Sample without replacement from a copy, then shuffle the union.  Seeding
    # per split keeps train and val independent but both reproducible.
    rng = random.Random(f"{seed}:{split}")
    picked = rng.sample(text_rows, quota)

    mixed = [dict(r) for r in table_rows] + [dict(r) for r in picked]
    if tag:
        for r in mixed[:len(table_rows)]:
            r["source"] = "table_only"
        for r in mixed[len(table_rows):]:
            r["source"] = "text_only"
    rng.shuffle(mixed)

    overlap = ({r["image"] for r in table_rows} & {r["image"] for r in picked})
    if overlap:
        print(f"WARNING [{split}]: {len(overlap)} image path(s) appear in both "
              f"source datasets, e.g. {sorted(overlap)[:3]}", file=sys.stderr)

    share = quota / len(mixed) if mixed else 0.0
    print(f"  {split:5s}: {len(table_rows):5d} table + {quota:5d} text "
          f"= {len(mixed):5d} rows  ({share:.1%} text, "
          f"{quota / len(text_rows):.1%} of the text pool)")
    return mixed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mix text-only examples into the table-only dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--table-train", default="mixed30_train.jsonl")
    ap.add_argument("--table-val", default="mixed30_val.jsonl")
    ap.add_argument("--text-train", default="dataset_train.jsonl")
    ap.add_argument("--text-val", default="dataset_val.jsonl")
    ap.add_argument("--ratio", type=float, default=0.40,
                    help="Mixing ratio; the requested band is 0.20-0.40")
    ap.add_argument("--mode", choices=("share", "fraction"), default="share",
                    help="'share': text is RATIO of the final mixed set. "
                         "'fraction': add RATIO of the available text examples.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sampling and shuffling; same seed = same files")
    ap.add_argument("--out-prefix", default="table_text_mixeddoc",
                    help="Output files are <prefix>_train.jsonl / <prefix>_val.jsonl")
    ap.add_argument("--out-dir", default=".", help="Directory for the new files")
    ap.add_argument("--tag-source", action="store_true",
                    help="Add a \"source\" field to each row recording which "
                         "dataset it came from (changes the schema)")
    ap.add_argument("--check-images", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Verify every referenced image path exists. On by "
                         "default: this script copies path references, it does "
                         "not create images, so a dataset whose images are "
                         "absent looks fine here and only fails at training.")
    ap.add_argument("--rewrite-root", metavar="OLD=NEW", action="append",
                    default=[],
                    help="Rewrite image path prefixes, e.g. "
                         "'/media/rookie/Storage1=/mnt/data'. Repeatable. "
                         "Applied before the existence check.")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="Write the output even if referenced images are absent")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting existing output files")
    args = ap.parse_args()

    if not 0.0 < args.ratio < 1.0:
        raise SystemExit(f"--ratio must be between 0 and 1, got {args.ratio}")
    if not 0.20 <= args.ratio <= 0.30:
        print(f"NOTE: --ratio {args.ratio} is outside the usual 0.20-0.30 band.",
              file=sys.stderr)

    rewrites = []
    for spec in args.rewrite_root:
        if "=" not in spec:
            raise SystemExit(f"--rewrite-root expects OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        rewrites.append((old, new))

    def load(path: str) -> list[dict]:
        rows = read_jsonl(path)
        for old, new in rewrites:
            for r in rows:
                if r["image"].startswith(old):
                    r["image"] = new + r["image"][len(old):]
        return rows

    table_train = load(args.table_train)
    table_val = load(args.table_val)
    text_train = load(args.text_train)
    text_val = load(args.text_val)

    print(f"Source datasets")
    print(f"  table: {len(table_train):5d} train / {len(table_val):5d} val")
    print(f"  text : {len(text_train):5d} train / {len(text_val):5d} val")
    print(f"Mixing at ratio={args.ratio} mode={args.mode} seed={args.seed}")

    mixed_train = build_split(table_train, text_train, args.ratio, args.mode,
                              args.seed, "train", args.tag_source)
    mixed_val = build_split(table_val, text_val, args.ratio, args.mode,
                            args.seed, "val", args.tag_source)

    # Train/val pools are disjoint by construction; assert it rather than trust it.
    leaked = {r["image"] for r in mixed_train} & {r["image"] for r in mixed_val}
    if leaked:
        raise SystemExit(
            f"ABORT: {len(leaked)} image(s) appear in both the mixed train and "
            f"val sets, e.g. {sorted(leaked)[:3]}. The source splits overlap.")

    if args.check_images:
        total = len(mixed_train) + len(mixed_val)
        missing = [r["image"] for r in mixed_train + mixed_val
                   if not Path(r["image"]).exists()]
        if not missing:
            print(f"  all {total} referenced images exist")
        else:
            # Group by root so a whole unmounted volume reads as one line
            # rather than thousands of individual paths.
            by_root: dict[str, int] = {}
            for p in missing:
                by_root["/".join(p.split("/")[:6])] = \
                    by_root.get("/".join(p.split("/")[:6]), 0) + 1
            print(f"\n{len(missing)} of {total} referenced images are MISSING:",
                  file=sys.stderr)
            for root, n in sorted(by_root.items(), key=lambda kv: -kv[1]):
                print(f"  {n:6d} under {root}", file=sys.stderr)
            print("\nThis script references images, it does not create them. "
                  "Mount the volume, pass --rewrite-root OLD=NEW if the files "
                  "moved, or re-render them from their doctags.",
                  file=sys.stderr)
            if not args.allow_missing_images:
                raise SystemExit(
                    "ABORT: refusing to write a dataset whose images are "
                    "absent -- training would fail on the first such example. "
                    "Pass --allow-missing-images to write it anyway.")
            print("Writing anyway (--allow-missing-images).", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{args.out_prefix}_train.jsonl"
    val_path = out_dir / f"{args.out_prefix}_val.jsonl"

    write_jsonl(mixed_train, train_path, args.force)
    write_jsonl(mixed_val, val_path, args.force)

    print("Wrote (originals untouched)")
    print(f"  {train_path}  {train_path.stat().st_size / 1e6:.1f} MB")
    print(f"  {val_path}  {val_path.stat().st_size / 1e6:.1f} MB")
    print("\nTrain with:")
    print(f"  python train_origin.py --train_data {train_path} "
          f"--val_data {val_path} --lora --bf16 --output_dir doctag_mixed")


if __name__ == "__main__":
    main()
