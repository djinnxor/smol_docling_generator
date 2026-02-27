"""
Prepare training data from XML annotations for fine-tuning
ibm-granite/granite-docling-258M (or any SmolDocling-compatible model)
to output DocTag format.

Reads XML annotation files produced by the labeler tool, matches each page
annotation to its corresponding page image, converts bounding-box + label
pairs to the DocTag string format used by SmolDocling, and writes the result
as JSONL files suitable for train.py.

Output JSONL schema (one JSON object per line):
    {"image": "<absolute path to PNG>", "doctag": "<doctag string>"}

Usage:
    python prepare_dataset.py [options]
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Scale used for <loc_N> tokens  (SmolDocling default: 500)
LOC_SCALE = 500

# Map XML element tag names → DocTag tag names
LABEL_TO_DOCTAG = {
    "text": "text",
    "section_header": "section_header_level_1",
    "section_header_level_1": "section_header_level_1",
    "section_header_level_2": "section_header_level_2",
    "section_header_level_3": "section_header_level_3",
    "title": "title",
    "caption": "caption",
    "picture": "picture",
    "table": "otsl",
    "list_item": "list_item",
    "formula": "formula",
    "page_header": "page_header",
    "page_footer": "page_footer",
    "code": "code",
    "footnote": "footnote",
}

# Image extensions searched in order when locating a page image
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _bbox_to_loc(x1: float, y1: float, x2: float, y2: float, scale: int = LOC_SCALE):
    """Convert normalised (0-1) bounding-box coords to integer location tokens."""
    return (
        max(0, min(scale, round(x1 * scale))),
        max(0, min(scale, round(y1 * scale))),
        max(0, min(scale, round(x2 * scale))),
        max(0, min(scale, round(y2 * scale))),
    )


def _element_to_doctag(tag: str, lx1: int, ly1: int, lx2: int, ly2: int, content: str) -> str:
    """Render a single document element as a DocTag string."""
    loc = f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
    content = content.strip().replace("\n", " ") if content else ""
    if tag == "picture":
        # Pictures have no text content in DocTag format
        return f"<{tag}>{loc}</{tag}>"
    return f"<{tag}>{loc}{content}</{tag}>"


def xml_to_doctag_pairs(xml_path: str, images_dir: str):
    """
    Parse a single XML annotation file and yield (image_path, doctag_string) pairs,
    one per page that has a matching image file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for page in root.findall("page"):
        page_num = page.get("num", "1")
        image_stem = page.get("image", "")

        # Locate the page image
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = Path(images_dir) / f"{image_stem}{ext}"
            if candidate.exists():
                image_path = str(candidate)
                break

        if image_path is None:
            continue

        # Build the ordered list of DocTag element strings for this page
        element_strings = []
        for child in page:
            tag = child.tag
            doctag_tag = LABEL_TO_DOCTAG.get(tag, tag)
            content = (child.text or "").strip()

            try:
                x1 = float(child.get("bbox_x1", 0))
                y1 = float(child.get("bbox_y1", 0))
                x2 = float(child.get("bbox_x2", 1))
                y2 = float(child.get("bbox_y2", 1))
            except (TypeError, ValueError):
                continue

            lx1, ly1, lx2, ly2 = _bbox_to_loc(x1, y1, x2, y2)
            element_strings.append(_element_to_doctag(doctag_tag, lx1, ly1, lx2, ly2, content))

        if not element_strings:
            continue

        doctag_str = "<doctag>\n" + "\n".join(element_strings) + "\n</doctag>"
        yield image_path, doctag_str


def prepare_dataset(
    xml_dir: str,
    images_dir: str,
    output_path: str = "training_data.jsonl",
    split_ratio: float = 0.9,
):
    """
    Process all XML annotation files in xml_dir, build (image, doctag) pairs,
    and write train / validation JSONL files.

    Returns (train_path, val_path) or (train_path, None) when all data goes to train.
    """
    records = []
    xml_dir_path = Path(xml_dir)

    xml_files = sorted(xml_dir_path.glob("*.xml"))
    if not xml_files:
        print(f"No XML annotation files found in {xml_dir}")
        return None, None

    for xml_file in xml_files:
        print(f"Processing {xml_file.name} …")
        for image_path, doctag in xml_to_doctag_pairs(str(xml_file), images_dir):
            records.append({"image": image_path, "doctag": doctag})

    if not records:
        print("No records found. Check that XML annotation files and page images are present.")
        return None, None

    split_idx = max(1, int(len(records) * split_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    base = output_path.replace(".jsonl", "")
    train_path = f"{base}_train.jsonl"
    val_path = f"{base}_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_records)} training record(s) to {train_path}")

    if val_records:
        with open(val_path, "w", encoding="utf-8") as f:
            for r in val_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(val_records)} validation record(s) to {val_path}")
    else:
        val_path = None
        print("No validation records (dataset too small for the chosen split ratio).")

    return train_path, val_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare DocTag training dataset from XML annotations"
    )
    parser.add_argument(
        "--xml_dir",
        default="dataset/xml_annotations",
        help="Directory containing XML annotation files (default: dataset/xml_annotations)",
    )
    parser.add_argument(
        "--images_dir",
        default="dataset/uploads",
        help="Directory containing page PNG images (default: dataset/uploads)",
    )
    parser.add_argument(
        "--output",
        default="training_data.jsonl",
        help="Base path for output JSONL files (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Fraction of data used for training; remainder goes to validation (default: 0.9)",
    )
    args = parser.parse_args()

    prepare_dataset(args.xml_dir, args.images_dir, args.output, args.split_ratio)
