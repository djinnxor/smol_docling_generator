#!/usr/bin/env python3
"""
Interactive Docling Tag Labeler
A web-based tool to review and correct docling tag detections for dataset creation.

Usage:
    python labeler.py [--port 5000] [--no-docling]

Options:
    --port      Port to run the Flask server on (default: 5000)
    --no-docling  Skip docling processing (show empty annotations)
"""

import json
import os
import re
import uuid
import argparse
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

# Default A4 page dimensions in points (used when docling cannot determine page size)
A4_WIDTH_POINTS = 595.0
A4_HEIGHT_POINTS = 842.0

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "labeler_data", "uploads")
ANNOTATIONS_FOLDER = os.path.join(BASE_DIR, "labeler_data", "annotations")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)

# All supported docling tag types with display colors
DOCLING_TAGS = [
    "text",
    "section_header_level_1",
    "section_header_level_2",
    "section_header_level_3",
    "title",
    "caption",
    "picture",
    "table",
    "list_item",
    "formula",
    "page_header",
    "page_footer",
    "code",
    "footnote",
]

TAG_COLORS = {
    "text": "#4CAF50",
    "section_header_level_1": "#2196F3",
    "section_header_level_2": "#03A9F4",
    "section_header_level_3": "#00BCD4",
    "title": "#E91E63",
    "caption": "#9C27B0",
    "picture": "#FF9800",
    "table": "#F44336",
    "list_item": "#8BC34A",
    "formula": "#FF5722",
    "page_header": "#607D8B",
    "page_footer": "#78909C",
    "code": "#795548",
    "footnote": "#9E9E9E",
}


def convert_pdf_to_images(pdf_path, dpi=150):
    """Convert PDF pages to PNG images, returns (image_paths, page_sizes)."""
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    page_sizes = []
    base = Path(pdf_path).stem

    for i, img in enumerate(images):
        img_path = os.path.join(UPLOAD_FOLDER, f"{base}_page_{i + 1}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)
        page_sizes.append({"width": img.width, "height": img.height})

    return image_paths, page_sizes


def _get_item_text(item):
    """Extract text content from a docling item."""
    if hasattr(item, "text") and item.text:
        return item.text
    if hasattr(item, "export_to_markdown"):
        try:
            return item.export_to_markdown() or ""
        except Exception:
            pass
    return str(item)


def _render_table_html(item):
    """
    Convert a docling TableItem's cell grid to an HTML table with proper
    colspan/rowspan so merged cells are visible in the labeler.
    Returns an HTML string, or empty string if unavailable.
    """
    try:
        grid = item.data.grid
        if not grid:
            return ""

        rendered_cells: set = set()
        rows_html = []

        for row_idx, row in enumerate(grid):
            cells_html = []
            for col_idx, cell in enumerate(row):
                if (row_idx, col_idx) in rendered_cells:
                    continue

                rs = max(1, int(getattr(cell, "row_span", 1) or 1))
                cs = max(1, int(getattr(cell, "col_span", 1) or 1))

                for r in range(row_idx, row_idx + rs):
                    for c in range(col_idx, col_idx + cs):
                        rendered_cells.add((r, c))

                is_header = bool(
                    getattr(cell, "column_header", False)
                    or getattr(cell, "row_header", False)
                )
                tag = "th" if is_header else "td"
                attrs = ""
                if rs > 1:
                    attrs += f' rowspan="{rs}"'
                if cs > 1:
                    attrs += f' colspan="{cs}"'

                text = str(getattr(cell, "text", "") or "")
                text = (
                    text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                cells_html.append(f"<{tag}{attrs}>{text}</{tag}>")

            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

        return "<table>" + "".join(rows_html) + "</table>"
    except Exception:
        return ""


def _bbox_to_norm(bbox, page_width, page_height):
    """
    Convert a docling BoundingBox to normalised (0-1) top-left-origin coords.
    Handles both BOTTOMLEFT (PDF) and TOPLEFT coordinate origins.
    Returns dict {x1, y1, x2, y2} or None.
    """
    l = getattr(bbox, "l", None)
    t = getattr(bbox, "t", None)
    r = getattr(bbox, "r", None)
    b = getattr(bbox, "b", None)

    if None in (l, t, r, b) or page_width == 0 or page_height == 0:
        return None

    # Detect coordinate origin from docling's CoordOrigin enum if available
    try:
        from docling_core.types.doc import CoordOrigin

        origin = getattr(bbox, "coord_origin", CoordOrigin.BOTTOMLEFT)
        if origin == CoordOrigin.BOTTOMLEFT:
            # Flip y: y_img = 1 - y_pdf / page_height
            x1 = l / page_width
            y1 = 1.0 - t / page_height
            x2 = r / page_width
            y2 = 1.0 - b / page_height
        else:
            x1 = l / page_width
            y1 = t / page_height
            x2 = r / page_width
            y2 = b / page_height
    except ImportError:
        # Fallback: assume BOTTOMLEFT if t > b, TOPLEFT otherwise
        if t > b:
            x1 = l / page_width
            y1 = 1.0 - t / page_height
            x2 = r / page_width
            y2 = 1.0 - b / page_height
        else:
            x1 = l / page_width
            y1 = t / page_height
            x2 = r / page_width
            y2 = b / page_height

    # Ensure correct ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Clamp to [0, 1]
    x1, y1, x2, y2 = (
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    )

    return {"x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)}


def process_with_docling(pdf_path):
    """
    Process PDF with docling and return elements grouped by page.
    Returns (elements_by_page, page_dims) where elements_by_page is
    {page_no: [element_dict, ...]} and page_dims is {page_no: {width, height}}.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter(allowed_formats=["pdf"])
    result = converter.convert(pdf_path)
    doc = result.document

    # Collect page physical dimensions from docling
    page_dims = {}
    pages_attr = doc.pages
    if hasattr(pages_attr, "items"):
        page_iter = pages_attr.items()
    elif hasattr(pages_attr, "__iter__"):
        page_iter = enumerate(pages_attr, 1)
    else:
        raise ValueError(f"Unexpected pages type: {type(pages_attr)}")
    for page_no, page in page_iter:
        w, h = A4_WIDTH_POINTS, A4_HEIGHT_POINTS
        if hasattr(page, "size") and page.size:
            w = getattr(page.size, "width", w)
            h = getattr(page.size, "height", h)
        page_dims[int(page_no)] = {"width": w, "height": h}

    # Extract all items
    elements_by_page = {}
    for item, _level in doc.iterate_items():
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        content = _get_item_text(item)

        if not hasattr(item, "prov") or not item.prov:
            continue

        prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
        for prov in prov_list:
            page_no = int(getattr(prov, "page_no", 1))
            raw_bbox = getattr(prov, "bbox", None)
            if raw_bbox is None:
                continue

            pd = page_dims.get(page_no, {"width": A4_WIDTH_POINTS, "height": A4_HEIGHT_POINTS})
            bbox_norm = _bbox_to_norm(raw_bbox, pd["width"], pd["height"])
            if bbox_norm is None:
                continue

            element = {
                "id": str(uuid.uuid4()),
                "label": label,
                "content": content,
                "bbox_norm": bbox_norm,
            }
            # For table items include the structured cell grid as HTML
            if label == "table":
                thtml = _render_table_html(item)
                if thtml:
                    element["table_html"] = thtml
            elements_by_page.setdefault(page_no, []).append(element)

    return elements_by_page, page_dims


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", tags=DOCLING_TAGS, tag_colors=json.dumps(TAG_COLORS))


@app.route("/process", methods=["POST"])
def process_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    pdf_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(pdf_path)

    # Convert PDF to page images
    try:
        image_paths, page_sizes = convert_pdf_to_images(pdf_path)
    except Exception as e:
        return jsonify({"error": f"PDF to image conversion failed: {e}"}), 500

    # Docling processing (optional – skip if not installed or fails)
    elements_by_page = {}
    docling_error = None
    try:
        elements_by_page, _ = process_with_docling(pdf_path)
    except Exception as e:
        docling_error = str(e)

    # Build response
    pages = []
    for i, (img_path, img_size) in enumerate(zip(image_paths, page_sizes), 1):
        pages.append({
            "page_num": i,
            "image_url": f"/page_image/{Path(img_path).name}",
            "width": img_size["width"],
            "height": img_size["height"],
            "elements": elements_by_page.get(i, []),
        })

    resp = {
        "filename": Path(file.filename).stem,
        "num_pages": len(pages),
        "pages": pages,
    }
    if docling_error:
        resp["docling_warning"] = f"Docling processing failed: {docling_error}"

    return jsonify(resp)


@app.route("/page_image/<filename>")
def page_image(filename):
    filename = secure_filename(Path(filename).name)
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/save_annotations", methods=["POST"])
def save_annotations():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    raw_name = data.get("filename", "annotations")
    filename = secure_filename(re.sub(r"[^\w\-.]", "_", raw_name)) or "annotations"
    pages_data = data.get("pages", {})

    output = {
        "filename": filename,
        "pages": [
            {"page": int(p), "elements": elems}
            for p, elems in sorted(pages_data.items(), key=lambda x: int(x[0]))
        ],
    }

    output_path = os.path.join(ANNOTATIONS_FOLDER, f"{filename}_annotations.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total = sum(len(e) for e in pages_data.values())
    # Provide both full path and a short relative path for display
    try:
        rel_path = os.path.relpath(output_path, BASE_DIR)
    except ValueError:
        rel_path = output_path
    return jsonify({
        "success": True,
        "saved_to": output_path,
        "saved_to_rel": rel_path,
        "num_elements": total,
    })


@app.route("/list_annotations")
def list_annotations():
    files = []
    for fn in os.listdir(ANNOTATIONS_FOLDER):
        if fn.endswith("_annotations.json"):
            files.append(fn)
    return jsonify({"files": sorted(files)})


@app.route("/load_annotations/<filename>")
def load_annotations(filename):
    filename = secure_filename(Path(filename).name)
    fpath = os.path.join(ANNOTATIONS_FOLDER, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "File not found"}), 404
    with open(fpath, encoding="utf-8") as f:
        return jsonify(json.load(f))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Docling Tag Labeler")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()
    print(f"\n  Docling Tag Labeler running at http://{args.host}:{args.port}/\n")
    app.run(debug=args.debug, host=args.host, port=args.port)
