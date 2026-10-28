"""
Mix any number of DocTag sources into one train/val pair, at shares you state.

make_mixed_dataset.py blends exactly two sources, so a four-way mix had to be
built by chaining it -- and the second pass then reweighted the first pass's
output instead of hitting the intended proportions.  This takes every source at
once, so the numbers you ask for are the numbers you get.

Each --source is NAME SPEC PATH(S), where SPEC is one of:

    all      use the whole pool
    1500     use exactly this many rows
    0.12     this fraction of the final dataset

Fixed sources (all / a count) are taken first; the fractional ones are then
sized against the total they imply, so "all of A plus 12% B" means B ends up as
12% of everything, not 12% of A.

    python make_dataset_mix.py \\
        --source rich  all  "/mnt/Transcend/updated_table doc/dataset.jsonl" \\
        --source mixed all  dataset_train.jsonl,dataset_val.jsonl \\
        --source text  0.12 text_only.jsonl \\
        --source table 0.08 table_only.jsonl \\
        --out-prefix richmix

Train/val is split per source, so both halves carry the same composition, and
no image is allowed to appear in both.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = {"image", "doctag"}

# Tags worth reporting on: the rare ones are what a narrow source cannot teach,
# so seeing their per-row density is how you tell a rich mix from a padded one.
REPORT_TAGS = ["otsl", "section_header_level_2", "caption", "picture",
               "key_value_region", "checkbox_unselected", "formula"]


def read_jsonl(path: str) -> list[dict]:
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
                raise SystemExit(f"{path}:{lineno}: row is missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows found")
    return rows


def density(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {t: 0.0 for t in REPORT_TAGS}
    c = Counter()
    for r in rows:
        for t in REPORT_TAGS:
            c[t] += r["doctag"].count(f"<{t}>")
    return {t: c[t] / len(rows) for t in REPORT_TAGS}


def resolve_counts(specs: list[tuple[str, str]], pools: dict[str, int]) -> dict[str, int]:
    """Turn each source's spec into a row count."""
    fixed: dict[str, int] = {}
    shares: dict[str, float] = {}
    for name, spec in specs:
        if spec == "all":
            fixed[name] = pools[name]
        elif "." in spec:
            shares[name] = float(spec)
        else:
            fixed[name] = int(spec)

    share_sum = sum(shares.values())
    if share_sum >= 1.0:
        raise SystemExit(f"fractional shares total {share_sum:.2f}; they must be < 1.0 "
                         "so the fixed sources have room")
    if not fixed and not shares:
        raise SystemExit("no sources given")

    if fixed:
        total = round(sum(fixed.values()) / (1.0 - share_sum))
    else:
        total = min(int(pools[n] / s) for n, s in shares.items())

    counts = dict(fixed)
    for name, share in shares.items():
        want = round(total * share)
        if want > pools[name]:
            print(f"WARNING [{name}]: wanted {want} rows for a {share:.0%} share but "
                  f"only {pools[name]} are available; using all of them.",
                  file=sys.stderr)
            want = pools[name]
        counts[name] = want
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mix several DocTag sources into one train/val pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", action="append", nargs=3, required=True,
                    metavar=("NAME", "SPEC", "PATHS"),
                    help="A source: NAME, then 'all' / a row count / a fraction, "
                         "then one or more comma-separated .jsonl paths. Repeatable.")
    ap.add_argument("--val-fraction", type=float, default=0.08,
                    help="Share of each source held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="richmix")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--rewrite-root", metavar="OLD=NEW", action="append", default=[],
                    help="Rewrite image path prefixes, e.g. '/old/images/=/new/images/'. "
                         "Repeatable. Applied before the existence check, and the way "
                         "to retarget a dataset at another machine's layout.")
    ap.add_argument("--tag-source", action="store_true",
                    help="Record which source each row came from in a \"source\" field")
    ap.add_argument("--check-images", action=argparse.BooleanOptionalAction, default=True,
                    help="Verify every referenced image exists. This script copies path "
                         "references, it does not create images, so a dataset whose "
                         "images are absent looks fine here and fails at training.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    rewrites = []
    for spec in args.rewrite_root:
        if "=" not in spec:
            raise SystemExit(f"--rewrite-root expects OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        rewrites.append((old, new))

    pools: dict[str, list[dict]] = {}
    specs: list[tuple[str, str]] = []
    for name, spec, paths in args.source:
        rows: list[dict] = []
        for p in paths.split(","):
            rows += read_jsonl(p.strip())
        for r in rows:
            for old, new in rewrites:
                if r["image"].startswith(old):
                    r["image"] = new + r["image"][len(old):]
        # A pool assembled from several files (or appended to over time) can
        # repeat an image; keep the first and say so rather than letting a
        # duplicate land in both train and val.
        seen, unique = set(), []
        for r in rows:
            if r["image"] in seen:
                continue
            seen.add(r["image"])
            unique.append(r)
        if len(unique) != len(rows):
            print(f"NOTE [{name}]: dropped {len(rows) - len(unique)} duplicate image(s)",
                  file=sys.stderr)
        pools[name] = unique
        specs.append((name, spec))

    print("source pools")
    for name, rows in pools.items():
        d = density(rows)
        print(f"  {name:10} {len(rows):6d} rows   " +
              "  ".join(f"{t[:9]}={d[t]:.2f}" for t in REPORT_TAGS[:4]))

    counts = resolve_counts(specs, {n: len(r) for n, r in pools.items()})
    total = sum(counts.values())

    train: list[dict] = []
    val: list[dict] = []
    for name, want in counts.items():
        rng = random.Random(f"{args.seed}:{name}")
        picked = rng.sample(pools[name], min(want, len(pools[name])))
        rng.shuffle(picked)
        n_val = round(len(picked) * args.val_fraction)
        part_val, part_train = picked[:n_val], picked[n_val:]
        if args.tag_source:
            for r in part_train + part_val:
                r["source"] = name
        train += [dict(r) for r in part_train]
        val += [dict(r) for r in part_val]

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)

    leaked = {r["image"] for r in train} & {r["image"] for r in val}
    if leaked:
        raise SystemExit(f"ABORT: {len(leaked)} image(s) in both train and val")

    print(f"\nmix ({total} rows -> {len(train)} train / {len(val)} val)")
    for name, want in counts.items():
        print(f"  {name:10} {want:6d}  {want / total:6.1%}")
    d = density(train + val)
    print("  per-row tag density: " +
          "  ".join(f"{t}={d[t]:.2f}" for t in REPORT_TAGS))

    if args.check_images:
        missing = [r["image"] for r in train + val if not Path(r["image"]).exists()]
        if not missing:
            print(f"  all {total} referenced images exist")
        else:
            by_root: dict[str, int] = {}
            for p in missing:
                root = p.rsplit("/", 1)[0]
                by_root[root] = by_root.get(root, 0) + 1
            print(f"\n{len(missing)} of {total} referenced images are MISSING:",
                  file=sys.stderr)
            for root, n in sorted(by_root.items(), key=lambda kv: -kv[1]):
                print(f"  {n:6d} under {root}", file=sys.stderr)
            print("\nThis script references images, it does not create them. Mount the "
                  "volume, or pass --rewrite-root OLD=NEW if they live elsewhere on the "
                  "machine that will train. Pass --no-check-images to write anyway.",
                  file=sys.stderr)
            raise SystemExit("ABORT: refusing to write a dataset whose images are absent")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rows, split in ((train, "train"), (val, "val")):
        path = out_dir / f"{args.out_prefix}_{split}.jsonl"
        if path.exists() and not args.force:
            raise SystemExit(f"{path} already exists -- pass --force or change --out-prefix")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {path}  {path.stat().st_size / 1e6:.1f} MB  {len(rows)} rows")


if __name__ == "__main__":
    main()
