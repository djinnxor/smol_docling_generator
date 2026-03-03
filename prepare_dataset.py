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
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split


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




# -----------------------------------------------------------------------
# COCO table-structure → DocTag OTSL converter
# -----------------------------------------------------------------------

# Class-ID mapping used by the COCO table-structure annotations
# (matches the categories in 60/notes.json and 60/classes.txt)
COCO_TABLE_CLASSES = {
    0: "table-row",
    1: "table-column",
    2: "table-spanning-cell",
    3: "table-column-header",
}


def _coco_cx_to_corners(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Convert COCO center-format bbox to (x1, y1, x2, y2) corner format."""
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _overlaps_1d(
    a1: float, a2: float, b1: float, b2: float, threshold: float = 0.3
) -> bool:
    """Return True if 1-D intervals [a1,a2] and [b1,b2] overlap by at least
    *threshold* of the shorter interval's length."""
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    shorter = min(a2 - a1, b2 - b1)
    return shorter > 0 and inter / shorter >= threshold


def _coco_annotations_to_otsl(
    annotations: list[tuple[int, float, float, float, float]],
    scale: int = LOC_SCALE,
) -> str | None:
    """Convert COCO-style table-structure annotations to a DocTag ``<otsl>…</otsl>`` string.

    Parameters
    ----------
    annotations:
        List of ``(class_id, cx, cy, w, h)`` tuples with normalised (0–1)
        coordinates in COCO center format.  ``class_id`` follows the mapping
        in :data:`COCO_TABLE_CLASSES`:
        0 = table-row, 1 = table-column, 2 = table-spanning-cell,
        3 = table-column-header.
    scale:
        ``<loc_N>`` token scale (default 500, matching SmolDocling convention).

    Returns
    -------
    OTSL DocTag string, or ``None`` when there are no rows or columns in the
    annotation set.
    """
    rows_raw: list[tuple[float, float, float, float]] = []
    cols_raw: list[tuple[float, float, float, float]] = []
    spans_raw: list[tuple[float, float, float, float]] = []
    header_regions: list[tuple[float, float, float, float]] = []

    for cls, cx, cy, w, h in annotations:
        bbox = _coco_cx_to_corners(cx, cy, w, h)
        if cls == 0:
            rows_raw.append(bbox)
        elif cls == 1:
            cols_raw.append(bbox)
        elif cls == 2:
            spans_raw.append(bbox)
        elif cls == 3:
            header_regions.append(bbox)

    if not rows_raw and not header_regions:
        return None
    if not cols_raw:
        return None

    # Build the unified row list: (x1, y1, x2, y2, is_header).
    # A class-0 row is a header if it overlaps a class-3 header region.
    # Class-3 header regions that don't overlap any class-0 row are inserted
    # as standalone header rows (some annotators label the header separately).
    matched_header_idx: set[int] = set()
    all_rows: list[tuple[float, float, float, float, bool]] = []
    for bbox in rows_raw:
        is_h = False
        for i, hr in enumerate(header_regions):
            if _overlaps_1d(bbox[1], bbox[3], hr[1], hr[3]):
                is_h = True
                matched_header_idx.add(i)
        all_rows.append((*bbox, is_h))
    for i, hr in enumerate(header_regions):
        if i not in matched_header_idx:
            all_rows.append((*hr, True))

    # Sort rows top-to-bottom (by y-center), columns left-to-right (by x-center)
    all_rows.sort(key=lambda r: (r[1] + r[3]) / 2)
    cols_raw.sort(key=lambda b: (b[0] + b[2]) / 2)

    # Table bounding box = envelope of all rows and columns
    all_bboxes = [(r[0], r[1], r[2], r[3]) for r in all_rows] + cols_raw
    tbl_x1 = min(b[0] for b in all_bboxes)
    tbl_y1 = min(b[1] for b in all_bboxes)
    tbl_x2 = max(b[2] for b in all_bboxes)
    tbl_y2 = max(b[3] for b in all_bboxes)
    lx1, ly1, lx2, ly2 = _bbox_to_loc(tbl_x1, tbl_y1, tbl_x2, tbl_y2, scale)
    loc_str = f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"

    n_cols = len(cols_raw)

    # Build spanning-cell extension map: (row_idx, col_idx) → continuation token
    span_token: dict[tuple[int, int], str] = {}
    for sx1, sy1, sx2, sy2 in spans_raw:
        covered: list[tuple[int, int]] = []
        for ri, row in enumerate(all_rows):
            if _overlaps_1d(row[1], row[3], sy1, sy2):
                for ci, col in enumerate(cols_raw):
                    if _overlaps_1d(col[0], col[2], sx1, sx2):
                        covered.append((ri, ci))

        if len(covered) <= 1:
            continue  # single-cell overlap — not a spanning cell

        min_r = min(c[0] for c in covered)
        min_c = min(c[1] for c in covered)
        for ri, ci in covered:
            if ri == min_r and ci == min_c:
                continue  # anchor cell keeps its normal token
            is_right = ci > min_c
            is_below = ri > min_r
            if is_right and is_below:
                span_token[(ri, ci)] = "xcel"
            elif is_right:
                span_token[(ri, ci)] = "lcel"
            else:
                span_token[(ri, ci)] = "ucel"

    # Generate OTSL tokens row by row
    otsl_parts: list[str] = []
    for ri, row in enumerate(all_rows):
        row_is_header = row[4]
        for ci in range(n_cols):
            if (ri, ci) in span_token:
                otsl_parts.append(f"<{span_token[(ri, ci)]}>")
            elif row_is_header:
                otsl_parts.append("<ched>")
            elif ci == 0:
                otsl_parts.append("<rhed>")
            else:
                otsl_parts.append("<fcel>")
        otsl_parts.append("<nl>")

    return f"<otsl>{loc_str}{''.join(otsl_parts)}</otsl>"


def coco_table_to_doctag_dataset(
    labels_dir: str,
    images_dir: str,
    output_path: str = "coco_table_training_data.jsonl",
    split_ratio: float = 0.9,
) -> tuple[str | None, str | None]:
    """Convert COCO-style table-structure annotations to DocTag OTSL format and
    write train / validation JSONL files ready for fine-tuning.

    The *labels_dir* directory must contain ``.txt`` files where each line has
    the form::

        <class_id> <cx> <cy> <w> <h>

    with normalised (0–1) coordinates in COCO center format and class IDs:

    * 0 = table-row
    * 1 = table-column
    * 2 = table-spanning-cell
    * 3 = table-column-header

    Each label file must share its stem with a matching image file (PNG/JPG/JPEG)
    in *images_dir*.  Label files whose matching image cannot be found are
    skipped with a warning.

    Output JSONL schema (one JSON object per line)::

        {"image": "<absolute path>", "doctag": "<doctag string>"}

    Returns ``(train_path, val_path)``.  *val_path* is ``None`` when there are
    too few records to split.
    """
    labels_dir_path = Path(labels_dir)
    txt_files = sorted(labels_dir_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt label files found in {labels_dir}")
        return None, None

    records: list[dict] = []
    for txt_file in txt_files:
        # Locate matching image (same stem, any supported extension)
        image_path: str | None = None
        for ext in IMAGE_EXTENSIONS:
            candidate = Path(images_dir) / f"{txt_file.stem}{ext}"
            if candidate.exists():
                image_path = str(candidate)
                break

        if image_path is None:
            print(f"  No image found for {txt_file.name}; skipping")
            continue

        # Parse COCO annotation lines
        annotations: list[tuple[int, float, float, float, float]] = []
        with open(txt_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append((cls, cx, cy, w, h))

        if not annotations:
            continue

        otsl = _coco_annotations_to_otsl(annotations, scale=LOC_SCALE)
        if otsl is None:
            print(f"  Could not build OTSL for {txt_file.name} (no rows/columns); skipping")
            continue

        doctag_str = f"<doctag>\n{otsl}\n</doctag>"
        print(f"Processing {txt_file.name} …")
        records.append({"image": image_path, "doctag": doctag_str})

    if not records:
        print("No records produced. Check that label files and images are present.")
        return None, None

    if len(records) >= 2:
        train_records, val_records = train_test_split(
            records, test_size=1 - split_ratio, random_state=42
        )
    else:
        train_records, val_records = records, []

    base = output_path.replace(".jsonl", "")
    train_path = f"{base}_train.jsonl"
    val_path = f"{base}_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as fh:
        for r in train_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(train_records)} training record(s) to {train_path}")

    if val_records:
        with open(val_path, "w", encoding="utf-8") as fh:
            for r in val_records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(val_records)} validation record(s) to {val_path}")
    else:
        val_path = None
        print("No validation records (dataset too small for the chosen split ratio).")

    return train_path, val_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare DocTag training dataset from XML or COCO-table annotations"
    )
    parser.add_argument(
        "--mode",
        choices=["xml", "coco_table"],
        default="xml",
        help=(
            "Annotation format to process: "
            "'xml' (default) uses XML files from --xml_dir; "
            "'coco_table' uses COCO .txt label files from --labels_dir"
        ),
    )
    parser.add_argument(
        "--xml_dir",
        default="labeler_data/xml_annotations",
        help="[xml mode] Directory containing XML annotation files (default: labeler_data/xml_annotations)",
    )
    parser.add_argument(
        "--labels_dir",
        default=None,
        help="[coco_table mode] Directory containing COCO .txt label files",
    )
    parser.add_argument(
        "--images_dir",
        default="labeler_data/uploads",
        help="Directory containing page images (default: labeler_data/uploads)",
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

    if args.mode == "xml":
        prepare_dataset(args.xml_dir, args.images_dir, args.output, args.split_ratio)
    else:
        if not args.labels_dir:
            parser.error("--labels_dir is required when --mode coco_table is used")
        coco_table_to_doctag_dataset(
            labels_dir=args.labels_dir,
            images_dir=args.images_dir,
            output_path=args.output,
            split_ratio=args.split_ratio,
        )
