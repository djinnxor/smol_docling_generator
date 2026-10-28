#!/usr/bin/env python3
"""
Combine many small JSON/JSONL doctag label files into one unified dataset
matching the schema used in the project examples:
    {"image": ".../images/mn_doc_00000.png", "doctag": "<doctag>...</doctag>"}

Usage:
    python json_fixer.py /path/to/dataset_root --output /path/to/combined.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, List, Optional


def _iter_candidate_files(root: Path) -> Iterator[Path]:
    possible_roots = [root, root / "doctag_labels", root / "labels", root / "annotations"]
    seen = set()

    for base in possible_roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"} and path not in seen:
                seen.add(path)
                yield path


def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_records_from_file(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    parsed = _safe_json_loads(text)
    if parsed is None:
        if path.suffix.lower() == ".jsonl":
            records: List[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = _safe_json_loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                elif isinstance(obj, str):
                    records.append({"doctag": obj})
            return records
        return [{"doctag": text}]

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    if isinstance(parsed, dict):
        return [parsed]

    return []


def _normalize_path(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    path = Path(value)
    if value.startswith("/") and path.exists():
        return str(path.resolve())
    return value


def _normalize_record(record: dict, file_path: Path, root_dir: Path) -> Optional[dict]:
    if not isinstance(record, dict):
        return None

    doctag_value = None
    for key in ("doctag", "label", "text", "annotation", "content", "doctag_text", "xml"):
        candidate = record.get(key)
        if isinstance(candidate, str):
            doctag_value = candidate
            break

    if doctag_value is None:
        for key in ("data", "result", "value"):
            nested = record.get(key)
            if isinstance(nested, dict):
                nested_record = _normalize_record(nested, file_path, root_dir)
                if nested_record is not None:
                    return nested_record

        for value in record.values():
            if isinstance(value, str) and "<doctag" in value.lower():
                doctag_value = value
                break

    if doctag_value is None:
        return None

    image_value = None
    for key in ("image", "image_path", "img", "filepath", "file", "path"):
        candidate = record.get(key)
        if isinstance(candidate, str) and candidate.strip():
            image_value = candidate
            break

    if image_value is None:
        stem = file_path.stem
        candidate_names = [
            f"{stem}.png",
            f"{stem}.jpg",
            f"{stem}.jpeg",
            f"{stem}.webp",
            f"{stem}.bmp",
        ]
        search_dirs = [
            root_dir / "images",
            root_dir / "imgs",
            root_dir,
            file_path.parent,
            file_path.parent.parent / "images",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for name in candidate_names:
                candidate = search_dir / name
                if candidate.exists():
                    image_value = str(candidate)
                    break
            if image_value:
                break

    if image_value is None:
        image_value = str(file_path)

    final = {
        "image": _normalize_path(image_value),
        "doctag": doctag_value,
    }

    if not final["doctag"]:
        return None

    return final


def combine_dataset(dataset_root: str, output_path: Optional[str] = None) -> List[dict]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    records: List[dict] = []
    seen = set()

    for file_path in _iter_candidate_files(root):
        try:
            raw_records = _extract_records_from_file(file_path)
        except Exception:
            continue

        for record in raw_records:
            normalized = _normalize_record(record, file_path, root)
            if normalized is None:
                continue

            key = (normalized.get("image"), normalized.get("doctag"))
            if key in seen:
                continue
            seen.add(key)
            records.append(normalized)

    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine split doctag label files into a single JSONL dataset.")
    parser.add_argument("dataset_path", help="Path to the dataset root containing label files and images")
    parser.add_argument("--output", "-o", default=None, help="Optional output JSONL path")
    args = parser.parse_args()

    combined = combine_dataset(args.dataset_path, args.output)
    print(f"Combined {len(combined)} records")
    if args.output:
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
