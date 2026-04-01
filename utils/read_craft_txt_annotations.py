"""Read CRAFT-style txt annotations with quadrilateral boxes and text.

Expected line format:
    x1,y1,x2,y2,x3,y3,x4,y4,transcription

The transcription field may contain commas. This parser keeps everything
after the first 8 numeric fields as transcription.

Usage:
    python utils/read_craft_txt_annotations.py \
        --input-dir /path/to/ch4_training_localization_transcription_gt

    python utils/read_craft_txt_annotations.py \
        --input-dir /path/to/ch4_training_localization_transcription_gt \
        --output-jsonl parsed_annotations.jsonl

    python utils/read_craft_txt_annotations.py \
        --input-dir /path/to/ch4_training_localization_transcription_gt \
        --output-doctag-jsonl craft_doctag.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def _axis_bbox(bbox: Dict[str, int]) -> Tuple[float, float, float, float]:
    xs = [bbox["x1"], bbox["x2"], bbox["x3"], bbox["x4"]]
    ys = [bbox["y1"], bbox["y2"], bbox["y3"], bbox["y4"]]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _bbox_center_height(bbox: Dict[str, int]) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = _axis_bbox(bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1)


def _bbox_union(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    ax1, ay1, ax2, ay2 = _axis_bbox(a)
    bx1, by1, bx2, by2 = _axis_bbox(b)
    x1 = int(round(min(ax1, bx1)))
    y1 = int(round(min(ay1, by1)))
    x2 = int(round(max(ax2, bx2)))
    y2 = int(round(max(ay2, by2)))
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y1,
        "x3": x2,
        "y3": y2,
        "x4": x1,
        "y4": y2,
    }


def _bbox_to_loc_tokens(
    bbox: Dict[str, int], image_width: int, image_height: int, loc_scale: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = _axis_bbox(bbox)

    def _clamp_loc(v: float, image_side: int) -> int:
        if image_side <= 0:
            return 0
        norm = max(0.0, min(1.0, v / float(image_side)))
        return int(round(norm * loc_scale))

    return (
        _clamp_loc(x1, image_width),
        _clamp_loc(y1, image_height),
        _clamp_loc(x2, image_width),
        _clamp_loc(y2, image_height),
    )


def parse_annotation_line(line: str, line_no: int, file_path: Path) -> Dict:
    """Parse one CRAFT annotation line into coordinates and text."""
    raw = line.rstrip("\n")
    if not raw.strip():
        raise ValueError(f"Empty annotation line at {file_path}:{line_no}")

    parts = raw.split(",", 8)
    if len(parts) < 9:
        raise ValueError(
            f"Invalid annotation at {file_path}:{line_no} - expected 9+ fields, got {len(parts)}"
        )

    try:
        coords = [int(v.strip()) for v in parts[:8]]
    except ValueError as exc:
        raise ValueError(
            f"Invalid coordinate value at {file_path}:{line_no}: {exc}"
        ) from exc

    transcription = parts[8].strip()
    return {
        "bbox": {
            "x1": coords[0],
            "y1": coords[1],
            "x2": coords[2],
            "y2": coords[3],
            "x3": coords[4],
            "y3": coords[5],
            "x4": coords[6],
            "y4": coords[7],
        },
        "text": transcription,
    }


def read_craft_txt_file(file_path: Path) -> List[Dict]:
    """Parse all annotations from a single .txt file."""
    items: List[Dict] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = parse_annotation_line(line, line_no, file_path)
            items.append(item)
    return items


def read_annotation_directory(input_dir: Path) -> List[Dict]:
    """Read all .txt annotation files in a directory."""
    txt_files = sorted(input_dir.glob("*.txt"))
    records: List[Dict] = []

    for txt_file in txt_files:
        items = read_craft_txt_file(txt_file)
        records.append(
            {
                "file": txt_file.name,
                "path": str(txt_file),
                "annotations": items,
            }
        )
    return records


def _group_into_lines(items: List[Dict], *, y_ratio: float) -> List[List[Dict]]:
    """Sort items by cy then x1 and split into lines on cy gaps.

    A new line is started whenever the gap between consecutive center-y values
    exceeds ``y_ratio * average_height`` of the two adjacent items.  This
    avoids the rolling-average drift that plagues greedy clustering.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda e: (e["cy"], e["x1"]))
    lines: List[List[Dict]] = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        prev = lines[-1][-1]
        gap = item["cy"] - prev["cy"]
        threshold = y_ratio * max(prev["h"], item["h"])
        if gap > threshold:
            lines.append([item])
        else:
            lines[-1].append(item)
    # Sort each line top-to-bottom by average cy (usually a no-op, but
    # guarantees stability when multiple lines have nearly equal average cy).
    lines.sort(key=lambda ln: sum(e["cy"] for e in ln) / len(ln))
    return lines


def merge_nearby_texts(
    annotations: List[Dict],
    *,
    y_ratio: float = 0.7,
    x_gap_ratio: float = 1.5,
) -> List[Dict]:
    """Group nearby boxes into lines, merge neighbors, and return reading order."""
    if not annotations:
        return []

    enriched = []
    for ann in annotations:
        text = ann["text"].strip()
        if not text:
            continue
        x1, y1, x2, y2 = _axis_bbox(ann["bbox"])
        cx, cy, h = _bbox_center_height(ann["bbox"])
        enriched.append(
            {
                "bbox": ann["bbox"],
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "h": h,
            }
        )

    # Group words into lines, then sort each line left-to-right.
    line_groups = _group_into_lines(enriched, y_ratio=y_ratio)

    merged: List[Dict] = []
    for line_group in line_groups:
        items = sorted(line_group, key=lambda a: a["x1"])
        current = None
        for it in items:
            if current is None:
                current = {"bbox": it["bbox"], "text": it["text"], "h": it["h"], "x2": it["x2"]}
                continue

            gap = it["x1"] - current["x2"]
            gap_tol = x_gap_ratio * max(current["h"], it["h"])

            if gap <= gap_tol:
                current["bbox"] = _bbox_union(current["bbox"], it["bbox"])
                current["text"] = f"{current['text']} {it['text']}".strip()
                current["h"] = max(current["h"], it["h"])
                current["x2"] = max(current["x2"], it["x2"])
            else:
                merged.append({"bbox": current["bbox"], "text": current["text"]})
                current = {"bbox": it["bbox"], "text": it["text"], "h": it["h"], "x2": it["x2"]}

        if current is not None:
            merged.append({"bbox": current["bbox"], "text": current["text"]})

    # Lines and within-line order are already correct from _group_into_lines +
    # per-line x1 sort above — no further sorting needed.
    return merged


def group_into_paragraphs(
    annotations: List[Dict],
    *,
    para_gap_ratio: float = 1.8,
) -> List[List[Dict]]:
    """Group ordered text lines into paragraphs based on vertical gaps.

    A new paragraph starts when the vertical gap between consecutive text
    blocks exceeds ``para_gap_ratio * average_height`` of the two adjacent
    blocks.  Returns a list of paragraphs, each paragraph being a list of
    annotation dicts (in reading order).
    """
    if not annotations:
        return []

    # Enrich each annotation with vertical metrics.
    enriched = []
    for ann in annotations:
        x1, y1, x2, y2 = _axis_bbox(ann["bbox"])
        cy = (y1 + y2) / 2.0
        h = max(1.0, y2 - y1)
        enriched.append({"ann": ann, "cy": cy, "h": h, "y2": y2, "y1": y1})

    paragraphs: List[List[Dict]] = [[enriched[0]["ann"]]]
    for i in range(1, len(enriched)):
        prev = enriched[i - 1]
        curr = enriched[i]
        # Gap = distance from bottom of previous block to top of current block.
        gap = curr["y1"] - prev["y2"]
        avg_h = (prev["h"] + curr["h"]) / 2.0
        if gap > para_gap_ratio * avg_h:
            paragraphs.append([])
        paragraphs[-1].append(curr["ann"])

    return paragraphs


def _order_texts_2d(annotations: List[Dict], y_ratio: float = 0.7) -> List[Dict]:
    """Order text blocks by rows (top-to-bottom) and columns (left-to-right)."""
    if not annotations:
        return []

    enriched = []
    for ann in annotations:
        x1, y1, x2, y2 = _axis_bbox(ann["bbox"])
        cy = (y1 + y2) / 2.0
        h = max(1.0, y2 - y1)
        enriched.append({"ann": ann, "x1": x1, "cy": cy, "h": h})

    rows = _group_into_lines(enriched, y_ratio=y_ratio)

    ordered: List[Dict] = []
    for row in rows:
        for e in sorted(row, key=lambda e: e["x1"]):
            ordered.append(e["ann"])
    return ordered


def build_doctag_text_elements(
    annotations: List[Dict],
    *,
    image_width: int,
    image_height: int,
    loc_scale: int,
    para_gap_ratio: float = 1.8,
) -> str:
    paragraphs = group_into_paragraphs(annotations, para_gap_ratio=para_gap_ratio)
    elements: List[str] = []
    for para in paragraphs:
        # Compute the bounding box that encloses the whole paragraph.
        para_bbox = para[0]["bbox"]
        for ann in para[1:]:
            para_bbox = _bbox_union(para_bbox, ann["bbox"])
        lx1, ly1, lx2, ly2 = _bbox_to_loc_tokens(
            para_bbox, image_width, image_height, loc_scale
        )
        # Join the text of all lines in the paragraph.
        para_text = " ".join(html.escape(ann["text"], quote=False) for ann in para)
        elements.append(f"<text><loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>{para_text}</text>")

    if elements:
        return "<doctag>\n" + "\n".join(elements) + "\n</doctag>"
    return "<doctag>\n</doctag>"


def export_doctag_jsonl(
    input_dir: Path,
    output_jsonl: Path,
    *,
    image_dir: Path,
    image_ext: str,
    y_ratio: float,
    x_gap_ratio: float,
    loc_scale: int,
    para_gap_ratio: float = 1.8,
) -> Tuple[int, int, int]:
    """Convert all txt annotations to DocTag JSONL records.

    Returns (written_count, skipped_no_image, skipped_empty).
    """
    txt_files = sorted(input_dir.glob("*.txt"))
    written = 0
    skipped_no_image = 0
    skipped_empty = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for txt_file in txt_files:
            image_path = _resolve_image_path(txt_file, image_dir, image_ext)
            anns = read_craft_txt_file(txt_file)
            merged = merge_nearby_texts(anns, y_ratio=y_ratio, x_gap_ratio=x_gap_ratio)
            if not merged:
                skipped_empty += 1
                continue

            if image_path.exists():
                with Image.open(image_path) as im:
                    width, height = im.size
            else:
                skipped_no_image += 1
                # Fallback: estimate canvas from observed coordinates
                max_x = max(_axis_bbox(a["bbox"])[2] for a in merged)
                max_y = max(_axis_bbox(a["bbox"])[3] for a in merged)
                width = int(max(1.0, max_x + 1.0))
                height = int(max(1.0, max_y + 1.0))

            doctag = build_doctag_text_elements(
                merged,
                image_width=width,
                image_height=height,
                loc_scale=loc_scale,
                para_gap_ratio=para_gap_ratio,
            )
            row = {
                "image": str(image_path),
                "doctag": doctag,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped_no_image, skipped_empty


def _resolve_image_path(txt_file: Path, image_dir: Path, image_ext: str) -> Path:
    """Resolve matching image path for a txt file.

    Image names are the same as txt names except image names do not start with
    'gt_'. This function tries both exact stem and gt_-stripped stem.
    """
    stem = txt_file.stem
    candidates = [stem]
    if stem.startswith("gt_"):
        candidates.append(stem[3:])

    exts = [image_ext, ".jpg", ".jpeg", ".png"]
    exts = list(dict.fromkeys(exts))

    for cand in candidates:
        for ext in exts:
            path = image_dir / f"{cand}{ext}"
            if path.exists():
                return path

    return image_dir / f"{candidates[-1]}{image_ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read CRAFT txt files with bbox corner coordinates and transcription text."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing CRAFT gt_*.txt files",
    )
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Optional JSONL output path. One parsed annotation row per line.",
    )
    parser.add_argument(
        "--output-doctag-jsonl",
        default="",
        help="Optional output path for DocTag dataset JSONL.",
    )
    parser.add_argument(
        "--preview-files",
        type=int,
        default=3,
        help="How many files to preview in stdout summary.",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Image extension for matching same-name image files (default: .jpg).",
    )
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional folder containing images. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--merge-y-ratio",
        type=float,
        default=2,
        help="Vertical tolerance ratio for grouping nearby text boxes.",
    )
    parser.add_argument(
        "--merge-x-gap-ratio",
        type=float,
        default=3,
        help="Horizontal gap ratio for merging neighboring text boxes in the same line.",
    )
    parser.add_argument(
        "--loc-scale",
        type=int,
        default=500,
        help="Scale for DocTag location tokens, usually 500.",
    )
    parser.add_argument(
        "--para-gap-ratio",
        type=float,
        default=1.8,
        help=(
            "Vertical gap ratio for paragraph breaks. Lines whose gap "
            "exceeds para_gap_ratio * avg_height start a new paragraph. "
            "Lower = more paragraphs, higher = fewer (default: 1.8)."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_dir = Path(args.image_dir) if args.image_dir else input_dir
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records = read_annotation_directory(input_dir)

    total_files = len(records)
    total_annotations = sum(len(r["annotations"]) for r in records)
    print(f"Parsed {total_files} txt file(s), total annotations: {total_annotations}")

    preview_n = max(0, min(args.preview_files, total_files))
    for record in records[:preview_n]:
        count = len(record["annotations"])
        print(f"- {record['file']}: {count} annotation(s)")
        if count:
            print(f"  first: {record['annotations'][0]}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        with output_path.open("w", encoding="utf-8") as out:
            for record in records:
                for ann in record["annotations"]:
                    row = {
                        "source_file": record["file"],
                        "bbox": ann["bbox"],
                        "text": ann["text"],
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL: {output_path}")

    if args.output_doctag_jsonl:
        output_path = Path(args.output_doctag_jsonl)
        written, skipped_no_image, skipped_empty = export_doctag_jsonl(
            input_dir,
            output_path,
            image_dir=image_dir,
            image_ext=args.image_ext,
            y_ratio=args.merge_y_ratio,
            x_gap_ratio=args.merge_x_gap_ratio,
            loc_scale=args.loc_scale,
            para_gap_ratio=args.para_gap_ratio,
        )
        print(
            "DocTag JSONL export done: "
            f"written={written}, skipped_no_image={skipped_no_image}, skipped_empty={skipped_empty}"
        )
        print(f"Wrote DocTag JSONL: {output_path}")


if __name__ == "__main__":
    main()
