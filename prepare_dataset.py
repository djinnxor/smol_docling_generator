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
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET


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
    if tag == "otsl":
        # OTSL tables have their own special format and ignore content
        return table_xml_to_doctag_from_str(content)
    loc = f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
    content = content.strip().replace("\n", " ") if content else ""
    if tag == "picture":
        # Pictures have no text content in DocTag format
        return f"<{tag}>{loc}</{tag}>"
    return f"<{tag}>{loc}{content}</{tag}>"

def table_xml_to_doctag(table_elem, scale=500):
    """
    Convert an XML <table> element to DocTag format.
    Preserves all rows, columns, and cells (including empty).
    """
    # Get table bbox
    bbox_x1 = float(table_elem.attrib.get("bbox_x1", 0))
    bbox_y1 = float(table_elem.attrib.get("bbox_y1", 0))
    bbox_x2 = float(table_elem.attrib.get("bbox_x2", 1))
    bbox_y2 = float(table_elem.attrib.get("bbox_y2", 1))

    # Scale bbox to DocTag loc tokens
    def loc(val): return int(round(val * scale))
    loc_str = (
        f"<loc_{loc(bbox_x1)}><loc_{loc(bbox_y1)}>"
        f"<loc_{loc(bbox_x2)}><loc_{loc(bbox_y2)}>"
    )

    # Parse table rows/cells
    rows = []
    # Find the nested <table> inside the outer <table>
    inner_table = table_elem.find("table")
    if inner_table is None:
        inner_table = table_elem  # fallback: treat as flat table

    for tr in inner_table.findall("tr"):
        row_cells = []
        for cell in tr:
            tag = cell.tag  # td or th
            text = (cell.text or "").strip()
            row_cells.append(f"<{tag}>{text}</{tag}>")
        rows.append("<row>" + "".join(row_cells) + "</row>")

    table_content = "".join(rows)

    # Compose DocTag
    doctag = f"<otsl>{loc_str}{table_content}</otsl>"
    return doctag

def table_xml_to_doctag_from_str(table_str, scale=500):
    """Parse table XML string and convert to DocTag."""
    table_elem = ET.fromstring(table_str)
    return table_xml_to_template_doctag(table_elem, scale=scale)

def table_xml_to_template_doctag(table_elem, scale=500):
    """
    Convert an XML <table> element to DocTag template with cspan/rspan for colspan/rowspan.
    """
    bbox_x1 = float(table_elem.attrib.get("bbox_x1", 0))
    bbox_y1 = float(table_elem.attrib.get("bbox_y1", 0))
    bbox_x2 = float(table_elem.attrib.get("bbox_x2", 1))
    bbox_y2 = float(table_elem.attrib.get("bbox_y2", 1))

    def loc(val): return int(round(val * scale))
    loc_str = (
        f"<loc_{loc(bbox_x1)}><loc_{loc(bbox_y1)}>"
        f"<loc_{loc(bbox_x2)}><loc_{loc(bbox_y2)}>"
    )

    inner_table = table_elem.find("table")
    if inner_table is None:
        inner_table = table_elem

    rows = inner_table.findall("tr")
    doctag_rows = []

    # Header row
    header_cells = rows[0]
    header_tags = []
    for cell in header_cells:
        text = (cell.text or "").strip()
        cspan = cell.attrib.get("colspan")
        rspan = cell.attrib.get("rowspan")
        span_str = ""
        if cspan:
            span_str += f' cspan="{cspan}"'
        if rspan:
            span_str += f' rspan="{rspan}"'
        header_tags.append(f"<ched{span_str}>{text}")
    doctag_rows.append("".join(header_tags) + "<nl>")

    # Data rows
    for tr in rows[1:]:
        row_tags = []
        first_cell = True
        for cell in tr:
            text = (cell.text or "").strip()
            cspan = cell.attrib.get("colspan")
            rspan = cell.attrib.get("rowspan")
            span_str = ""
            if cspan:
                span_str += f' cspan="{cspan}"'
            if rspan:
                span_str += f' rspan="{rspan}"'
            if first_cell:
                row_tags.append(f"<rhed{span_str}>{text}")
                first_cell = False
            else:
                row_tags.append(f"<fcel{span_str}>{text}")
        doctag_rows.append("".join(row_tags) + "<nl>")

    doctag = f"<otsl>{loc_str}{''.join(doctag_rows)}</otsl>"
    return doctag

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

            try:
                x1 = float(child.get("bbox_x1", 0))
                y1 = float(child.get("bbox_y1", 0))
                x2 = float(child.get("bbox_x2", 1))
                y2 = float(child.get("bbox_y2", 1))
            except (TypeError, ValueError):
                continue

            lx1, ly1, lx2, ly2 = _bbox_to_loc(x1, y1, x2, y2)

            # For tables, pass the full XML string (including children)
            if doctag_tag == "otsl":
                content = ET.tostring(child, encoding="unicode")
            else:
                content = (child.text or "").strip()

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

    train_records, val_records = train_test_split(
        records, test_size=1 - split_ratio, random_state=42
    )
    
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
        default="labeler_data/xml_annotations",
        help="Directory containing XML annotation files (default: labeler_data/xml_annotations)",
    )
    parser.add_argument(
        "--images_dir",
        default="labeler_data/uploads",
        help="Directory containing page PNG images (default: labeler_data/uploads)",
    )
    parser.add_argument(
        "--output",
        default="training_data.jsonl",
        help="Base path for output JSONL files (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training; remainder goes to validation (default: 0.8)",
    )
    args = parser.parse_args()

    prepare_dataset(args.xml_dir, args.images_dir, args.output, args.split_ratio)
