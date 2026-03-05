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
import shutil
import uuid
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from google_ocr import GoogleOCR

# Default A4 page dimensions in points (used when docling cannot determine page size)
A4_WIDTH_POINTS = 595.0
A4_HEIGHT_POINTS = 842.0

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "labeler_data", "uploads")
ANNOTATIONS_FOLDER = os.path.join(BASE_DIR, "labeler_data", "annotations")
XML_ANNOTATIONS_FOLDER = os.path.join(BASE_DIR, "labeler_data", "xml_annotations")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(XML_ANNOTATIONS_FOLDER, exist_ok=True)

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


def _extract_table_cell_bboxes(item, page_width, page_height):
    """
    Extract per-cell normalised bounding boxes from a docling TableItem.
    Returns a list of cell dicts or None if unavailable.
    """
    try:
        grid = item.data.grid
        if not grid:
            return None
        cells = []
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                cell_bbox = getattr(cell, "bbox", None)
                if cell_bbox is None:
                    continue
                norm = _bbox_to_norm(cell_bbox, page_width, page_height)
                if norm:
                    cells.append({
                        "row": row_idx,
                        "col": col_idx,
                        "row_span": max(1, int(getattr(cell, "row_span", 1) or 1)),
                        "col_span": max(1, int(getattr(cell, "col_span", 1) or 1)),
                        "bbox_norm": norm,
                        "text": str(getattr(cell, "text", "") or ""),
                    })
        return cells if cells else None
    except Exception:
        return None


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
            # For table items include the structured cell grid as HTML and cell bboxes
            if label == "table":
                thtml = _render_table_html(item)
                if thtml:
                    element["table_html"] = thtml
                cell_bboxes = _extract_table_cell_bboxes(item, pd["width"], pd["height"])
                # if cell_bboxes:
                #     element["table_cells"] = cell_bboxes
            elements_by_page.setdefault(page_no, []).append(element)

    return elements_by_page, page_dims


def apply_google_ocr_to_elements(elements_by_page, image_paths):
    """
    Replace each element's text content with Google Vision OCR output.
    Crops the element's bounding box from the page image and OCRs the region.
    Requires the GOOGLE_OCR_API_KEY environment variable to be set.
    Labels that typically contain no text (e.g. 'picture') are skipped.
    """
    api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
    ocr = GoogleOCR(api_key=api_key)
    skip_labels = {"picture"}

    for page_no, img_path in enumerate(image_paths, 1):
        page_elements = elements_by_page.get(page_no, [])
        if not page_elements:
            continue
        img = Image.open(img_path)
        img_w, img_h = img.size

        for el in page_elements:
            if el.get("label") in skip_labels:
                continue
            bbox = el.get("bbox_norm")
            if not bbox:
                continue
            x1 = int(bbox["x1"] * img_w)
            y1 = int(bbox["y1"] * img_h)
            x2 = int(bbox["x2"] * img_w)
            y2 = int(bbox["y2"] * img_h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cropped = img.crop((x1, y1, x2, y2))
            tmp_path = os.path.join(UPLOAD_FOLDER, f"_crop_{uuid.uuid4().hex}.png")
            cropped.save(tmp_path, "PNG")
            try:
                text = ocr.google_vision_ocr(tmp_path)
                if text.strip():
                    el["content"] = text.strip()
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    return elements_by_page


def apply_rapidocr_to_elements(elements_by_page, image_paths):
    """
    Replace each element's text content using RapidOCR (docling's built-in OCR engine).
    Crops each element's bounding box from the page image and OCRs the region.
    Labels that typically contain no text (e.g. 'picture') are skipped.
    """
    engine = _get_rapidocr_engine()
    skip_labels = {"picture"}

    for page_no, img_path in enumerate(image_paths, 1):
        page_elements = elements_by_page.get(page_no, [])
        if not page_elements:
            continue
        img = Image.open(img_path)
        img_w, img_h = img.size

        for el in page_elements:
            if el.get("label") in skip_labels:
                continue
            bbox = el.get("bbox_norm")
            if not bbox:
                continue
            x1 = int(bbox["x1"] * img_w)
            y1 = int(bbox["y1"] * img_h)
            x2 = int(bbox["x2"] * img_w)
            y2 = int(bbox["y2"] * img_h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cropped = img.crop((x1, y1, x2, y2))
            img_array = np.array(cropped)
            try:
                result, _ = engine(img_array)
                text = _rapidocr_text_from_result(result)
                if text:
                    el["content"] = text
            except Exception:
                pass

    return elements_by_page


# ---------------------------------------------------------------------------
# Shared OCR helpers used by the route handlers below
# ---------------------------------------------------------------------------

def _get_rapidocr_engine():
    """Return a RapidOCR engine instance (raises ImportError if not installed)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        try:
            from rapidocr_openvino import RapidOCR
        except ImportError:
            raise ImportError(
                "RapidOCR is not installed. Run: pip install rapidocr-onnxruntime"
            )
    return RapidOCR()


def _rapidocr_text_from_result(result) -> str:
    """Extract a plain text string from a RapidOCR result tuple.

    Each detected line is a tuple ``(bbox, text, confidence)``.  Lines are
    joined with a space so the output matches single-line text regions.
    """
    if not result:
        return ""
    lines = [r[1] for r in result if r and len(r) > 1 and r[1]]
    return " ".join(lines).strip()


def _ocr_crop_pil(
    cropped_pil,
    ocr_model: str,
    google_ocr: "GoogleOCR | None" = None,
) -> str:
    """Run OCR on a PIL Image crop and return the extracted text string.

    Parameters
    ----------
    cropped_pil:
        A :class:`PIL.Image.Image` crop of the region to recognise.
    ocr_model:
        ``"google"`` for Google Vision OCR, ``"rapidocr"`` for RapidOCR.
    google_ocr:
        An already-constructed :class:`GoogleOCR` instance (used when
        *ocr_model* is ``"google"``).  If *None* a new instance is created
        using the ``GOOGLE_OCR_API_KEY`` env var.
    """
    if ocr_model == "rapidocr":
        engine = _get_rapidocr_engine()
        img_array = np.array(cropped_pil)
        result, _ = engine(img_array)
        return _rapidocr_text_from_result(result)

    # Default: Google Vision OCR
    if google_ocr is None:
        api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
        google_ocr = GoogleOCR(api_key=api_key)
    tmp_path = os.path.join(UPLOAD_FOLDER, f"_ocr_{uuid.uuid4().hex}.png")
    cropped_pil.save(tmp_path, "PNG")
    try:
        return google_ocr.google_vision_ocr(tmp_path).strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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

    # Apply OCR based on selected model.
    # When a specific OCR model is chosen, first clear docling's own text so
    # only the selected OCR model's output is shown (not docling fallback text).
    ocr_model = request.form.get("ocr_model", "none")
    ocr_error = None
    if ocr_model != "none":
        for page_elements in elements_by_page.values():
            for el in page_elements:
                el["content"] = ""
    if ocr_model == "google":
        try:
            elements_by_page = apply_google_ocr_to_elements(elements_by_page, image_paths)
        except Exception as e:
            ocr_error = f"Google OCR failed: {e}"
    elif ocr_model == "rapidocr":
        try:
            elements_by_page = apply_rapidocr_to_elements(elements_by_page, image_paths)
        except Exception as e:
            ocr_error = f"RapidOCR failed: {e}"

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
        "pdf_filename": safe_name,
        "num_pages": len(pages),
        "pages": pages,
    }
    if docling_error:
        resp["docling_warning"] = f"Docling processing failed: {docling_error}"
    if ocr_error:
        resp["ocr_warning"] = ocr_error

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

    pages_data = data.get("pages", {})
    saved_files = []
    total_elements = 0

    for p, page_info in sorted(pages_data.items(), key=lambda x: int(x[0])):
        # Support both new format {image_name, width, height, elements} and legacy list format
        if isinstance(page_info, list):
            elements = page_info
            image_name = f"page_{p}"
            width = 0
            height = 0
        else:
            elements = page_info.get("elements", [])
            image_name = page_info.get("image_name", f"page_{p}")
            width = page_info.get("width", 0)
            height = page_info.get("height", 0)

        safe_image_name = secure_filename(re.sub(r"[^\w\-.]", "_", image_name)) or f"page_{p}"

        page_output = {
            "image_name": safe_image_name,
            "width": width,
            "height": height,
            "elements": elements,
        }

        output_path = os.path.join(ANNOTATIONS_FOLDER, f"{safe_image_name}_annotations.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(page_output, f, ensure_ascii=False, indent=2)

        saved_files.append(output_path)
        total_elements += len(elements)

    if not saved_files:
        return jsonify({"error": "No pages to save"}), 400

    try:
        rel_dir = os.path.relpath(ANNOTATIONS_FOLDER, BASE_DIR)
    except ValueError:
        rel_dir = ANNOTATIONS_FOLDER

    return jsonify({
        "success": True,
        "saved_to": saved_files[0],
        "saved_to_rel": rel_dir,
        "saved_files": [os.path.basename(f) for f in saved_files],
        "num_elements": total_elements,
        "num_pages": len(saved_files),
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


@app.route("/ocr_region", methods=["POST"])
def ocr_region():
    """
    Run OCR on a specific bounding box region of a page image.
    Expects JSON: { image_filename: str, bbox_norm: {x1, y1, x2, y2},
                    ocr_model: "google" | "rapidocr"  (optional, default "google") }
    Returns: { text: str }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    image_filename = data.get("image_filename", "")
    bbox_norm = data.get("bbox_norm", {})
    ocr_model = data.get("ocr_model", "google")

    if not image_filename:
        return jsonify({"error": "image_filename is required"}), 400

    img_path = os.path.join(UPLOAD_FOLDER, secure_filename(Path(image_filename).name))
    if not os.path.exists(img_path):
        return jsonify({"error": f"Image not found: {image_filename}"}), 404

    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
        x1 = int(bbox_norm.get("x1", 0) * img_w)
        y1 = int(bbox_norm.get("y1", 0) * img_h)
        x2 = int(bbox_norm.get("x2", 1) * img_w)
        y2 = int(bbox_norm.get("y2", 1) * img_h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return jsonify({"error": "Bounding box too small"}), 400

        cropped = img.crop((x1, y1, x2, y2))
        text = _ocr_crop_pil(cropped, ocr_model)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500


@app.route("/export_xml", methods=["POST"])
def export_xml():
    """
    Save each page's annotations as a separate XML file in xml_annotations/,
    named after the page image. Does NOT return a file download.
    Accepts the same JSON payload as /save_annotations.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pages_data = data.get("pages", {})
    saved_files = []

    for p, page_info in sorted(pages_data.items(), key=lambda x: int(x[0])):
        if isinstance(page_info, list):
            elements = page_info
            image_name = f"page_{p}"
            width = 0
            height = 0
        else:
            elements = page_info.get("elements", [])
            image_name = page_info.get("image_name", f"page_{p}")
            width = page_info.get("width", 0)
            height = page_info.get("height", 0)

        safe_image_name = secure_filename(re.sub(r"[^\w\-.]", "_", image_name)) or f"page_{p}"

        page_el = ET.Element("page")
        page_el.set("num", str(p))
        page_el.set("image", safe_image_name)
        page_el.set("width", str(width))
        page_el.set("height", str(height))

        for el in elements:
            label = re.sub(r"[^a-zA-Z0-9_]", "_", el.get("label", "text"))
            content = el.get("content", "")
            bbox = el.get("bbox_norm", {})

            item_el = ET.SubElement(page_el, label)
            item_el.set("bbox_x1", str(bbox.get("x1", 0)))
            item_el.set("bbox_y1", str(bbox.get("y1", 0)))
            item_el.set("bbox_x2", str(bbox.get("x2", 1)))
            item_el.set("bbox_y2", str(bbox.get("y2", 1)))

            if label == "table" and el.get("table_html"):
                try:
                    table_xml = ET.fromstring(el["table_html"])
                    item_el.append(table_xml)
                except Exception:
                    item_el.text = content
            else:
                item_el.text = content

        # Pretty-print
        raw_xml = ET.tostring(page_el, encoding="unicode")
        try:
            pretty = minidom.parseString(raw_xml).toprettyxml(indent="  ")
            lines = pretty.split("\n")
            if lines and lines[0].startswith("<?xml"):
                lines = lines[1:]
            xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines)
        except Exception:
            xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n' + raw_xml

        # Save to xml_annotations folder, one file per page image
        xml_save_path = os.path.join(XML_ANNOTATIONS_FOLDER, f"{safe_image_name}_annotations.xml")
        with open(xml_save_path, "w", encoding="utf-8") as f:
            f.write(xml_output)
        saved_files.append(xml_save_path)

    if not saved_files:
        return jsonify({"error": "No pages to save"}), 400

    try:
        rel_dir = os.path.relpath(XML_ANNOTATIONS_FOLDER, BASE_DIR)
    except ValueError:
        rel_dir = XML_ANNOTATIONS_FOLDER

    return jsonify({
        "success": True,
        "saved_to": saved_files[0],
        "saved_to_rel": rel_dir,
        "saved_files": [os.path.basename(f) for f in saved_files],
        "num_pages": len(saved_files),
    })


@app.route("/redetect_tables", methods=["POST"])
def redetect_tables():
    """
    Re-detect tables on a single page using the CV-based TableDetector
    (table_detection.py) and populate cell text with the selected OCR engine.
    Accepts optional field ``ocr_model``: ``"google"`` (default) or ``"rapidocr"``.
    Returns table elements in the same format as /process.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pdf_filename = data.get("pdf_filename", "")
    page_no = int(data.get("page_no", 1))
    ocr_model = data.get("ocr_model", "google")

    if not pdf_filename:
        return jsonify({"error": "pdf_filename is required"}), 400

    # Locate the page image saved during /process
    pdf_stem = Path(secure_filename(pdf_filename)).stem
    img_filename = f"{pdf_stem}_page_{page_no}.png"
    img_path = os.path.join(UPLOAD_FOLDER, img_filename)
    if not os.path.exists(img_path):
        return jsonify({"error": f"Page image not found ({img_filename}). Re-process the PDF first."}), 404

    try:
        import cv2
        import numpy as np
        from table_detection import TableDetector, HTMLTableBuilder

        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        detector = TableDetector()
        detections = detector.detect(cv_img, page_no)

        # Prepare OCR engine once for all cells
        if ocr_model == "rapidocr":
            rapid_engine = _get_rapidocr_engine()
            _google_ocr_inst = None
        else:
            rapid_engine = None
            api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
            _google_ocr_inst = GoogleOCR(api_key=api_key)

        for det in detections:
            for cell in det.cells:
                x1 = max(0, int(cell.x))
                y1 = max(0, int(cell.y))
                x2 = min(img_w, int(cell.x + cell.width))
                y2 = min(img_h, int(cell.y + cell.height))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                cropped = pil_img.crop((x1, y1, x2, y2))
                try:
                    if ocr_model == "rapidocr":
                        img_array = np.array(cropped)
                        result, _ = rapid_engine(img_array)
                        text = _rapidocr_text_from_result(result)
                    else:
                        text = _ocr_crop_pil(cropped, "google", _google_ocr_inst)
                    if text:
                        cell.text = text
                except Exception:
                    pass

            # Rebuild HTML with OCR text
            num_rows = max((c.row for c in det.cells), default=0) + 1
            num_cols = max((c.col for c in det.cells), default=0) + 1
            builder = HTMLTableBuilder()
            det.html = builder.build(det.cells, num_rows, num_cols)

        # Convert to element format expected by the frontend
        def _clamp_norm(v):
            return round(max(0.0, min(1.0, v)), 4)

        table_elements = []
        for det in detections:
            bbox_norm = {
                "x1": _clamp_norm(det.x / img_w),
                "y1": _clamp_norm(det.y / img_h),
                "x2": _clamp_norm((det.x + det.width) / img_w),
                "y2": _clamp_norm((det.y + det.height) / img_h),
            }
            table_cells = [
                {
                    "row": cell.row,
                    "col": cell.col,
                    "row_span": cell.rowspan,
                    "col_span": cell.colspan,
                    "bbox_norm": {
                        "x1": _clamp_norm(cell.x / img_w),
                        "y1": _clamp_norm(cell.y / img_h),
                        "x2": _clamp_norm((cell.x + cell.width) / img_w),
                        "y2": _clamp_norm((cell.y + cell.height) / img_h),
                    },
                    "text": cell.text,
                }
                for cell in det.cells
            ]
            # Build content field row-by-row to preserve table structure
            rows_text = {}
            for cell in det.cells:
                if cell.text:
                    rows_text.setdefault(cell.row, []).append(cell.text)
            content_str = "\n".join(
                " | ".join(row_texts)
                for row_index, row_texts in sorted(rows_text.items())
                if row_texts
            ) if rows_text else ""
            table_elements.append({
                "id": str(uuid.uuid4()),
                "label": "table",
                "content": content_str,
                "bbox_norm": bbox_norm,
                "table_html": det.html,
                #"table_cells": table_cells,
            })

        return jsonify({
            "tables": table_elements,
            "all_elements": table_elements,
            "page_no": page_no,
        })

    except Exception as e:
        return jsonify({"error": f"Re-detection failed: {e}"}), 500


@app.route("/ocr_table_cells", methods=["POST"])
def ocr_table_cells():
    """
    Run OCR on every individual cell of a manually structured table.

    The frontend sends the page image filename, the table's normalised
    bounding box, and arrays of row/column separator positions.  Each cell
    is cropped, OCR-ed with the selected engine, and the resulting text is
    written into a rebuilt HTML ``<table>``.

    Expects JSON:
    {
        "image_filename": str,
        "bbox_norm":  { x1, y1, x2, y2 },
        "table_rows": [ 0.0, 0.3, ... ],   // normalised y positions of row dividers
        "table_cols": [ 0.0, 0.5, ... ],   // normalised x positions of col dividers
        "table_html": str,                  // existing HTML (for header detection)
        "ocr_model":  "google" | "rapidocr"  (optional, default "google")
    }
    Returns: { "table_html": str }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    image_filename = data.get("image_filename", "")
    bbox_norm = data.get("bbox_norm", {})
    table_rows = data.get("table_rows", [])
    table_cols = data.get("table_cols", [])
    existing_html = data.get("table_html", "")
    ocr_model = data.get("ocr_model", "google")

    if not image_filename:
        return jsonify({"error": "image_filename is required"}), 400

    img_path = os.path.join(UPLOAD_FOLDER, secure_filename(Path(image_filename).name))
    if not os.path.exists(img_path):
        return jsonify({"error": f"Image not found: {image_filename}"}), 404

    try:
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Absolute pixel coordinates of the table bounding box
        tx1 = int(bbox_norm.get("x1", 0) * img_w)
        ty1 = int(bbox_norm.get("y1", 0) * img_h)
        tx2 = int(bbox_norm.get("x2", 1) * img_w)
        ty2 = int(bbox_norm.get("y2", 1) * img_h)
        tbl_w = max(1, tx2 - tx1)
        tbl_h = max(1, ty2 - ty1)

        # Build cell grid from divider positions.
        # table_rows / table_cols are normalised positions *within* the table box.
        def _row_boundaries():
            ys = sorted({0.0} | {r for r in table_rows if 0.0 < r < 1.0} | {1.0})
            return list(zip(ys[:-1], ys[1:]))

        def _col_boundaries():
            xs = sorted({0.0} | {c for c in table_cols if 0.0 < c < 1.0} | {1.0})
            return list(zip(xs[:-1], xs[1:]))

        row_bounds = _row_boundaries()
        col_bounds = _col_boundaries()

        if not row_bounds or not col_bounds:
            return jsonify({"error": "No valid row/col dividers provided"}), 400

        # Pre-parse existing HTML to preserve header (th) tags
        existing_header_flags: dict[tuple, bool] = {}
        try:
            import html.parser as _hp

            class _HeaderParser(_hp.HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._row = -1
                    self._col = -1
                    self._in_cell = False
                    self._is_header = False
                    self.flags: dict[tuple, bool] = {}

                def handle_starttag(self, tag, attrs):
                    if tag == "tr":
                        self._row += 1
                        self._col = -1
                    elif tag in ("td", "th"):
                        self._col += 1
                        self._in_cell = True
                        self._is_header = (tag == "th")

                def handle_endtag(self, tag):
                    if tag in ("td", "th"):
                        self.flags[(self._row, self._col)] = self._is_header
                        self._in_cell = False

            p = _HeaderParser()
            p.feed(existing_html)
            existing_header_flags = p.flags
        except Exception:
            pass

        # Prepare OCR engine once for all cells
        if ocr_model == "rapidocr":
            rapid_engine = _get_rapidocr_engine()
            _google_ocr_inst = None
        else:
            rapid_engine = None
            api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
            _google_ocr_inst = GoogleOCR(api_key=api_key)

        # Build new HTML table
        rows_html = []
        for ri, (r0, r1) in enumerate(row_bounds):
            cells_html = []
            for ci, (c0, c1) in enumerate(col_bounds):
                # Cell pixel coords (absolute in page image)
                cx1 = tx1 + int(c0 * tbl_w)
                cy1 = ty1 + int(r0 * tbl_h)
                cx2 = tx1 + int(c1 * tbl_w)
                cy2 = ty1 + int(r1 * tbl_h)
                cell_text = ""
                if cx2 - cx1 >= 4 and cy2 - cy1 >= 4:
                    cropped = img.crop((cx1, cy1, cx2, cy2))
                    try:
                        if ocr_model == "rapidocr":
                            img_arr = np.array(cropped)
                            result, _ = rapid_engine(img_arr)
                            cell_text = _rapidocr_text_from_result(result)
                        else:
                            cell_text = _ocr_crop_pil(cropped, "google", _google_ocr_inst)
                    except Exception:
                        pass
                is_header = existing_header_flags.get((ri, ci), False)
                tag = "th" if is_header else "td"
                safe_text = (
                    cell_text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                cells_html.append(f"<{tag}>{safe_text}</{tag}>")
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

        new_html = "<table>" + "".join(rows_html) + "</table>"
        return jsonify({"table_html": new_html})

    except Exception as e:
        return jsonify({"error": f"OCR table cells failed: {e}"}), 500


# ---------------------------------------------------------------------------
# Import image folder
# ---------------------------------------------------------------------------

_YOLO_ROW_CLASSES = {"table row"}
_YOLO_COL_CLASSES = {"table column"}
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _parse_yolo_annotations(label_file: Path, class_names: list) -> list:
    """Parse a YOLO-format label file and return a list of element dicts.

    Each line: ``class_id  cx  cy  w  h``  (all values normalised 0-1).

    For table-structure datasets (rows, columns, spanning cells, headers) the
    function groups all annotations on the page into a single ``"table"``
    element whose bounding box is the union of all annotations.  Row separator
    positions (normalised within the table box) are stored in ``table_rows``
    and column separator positions in ``table_cols`` so the labeler can render
    the grid immediately.
    """
    try:
        lines = label_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except Exception:
        return []

    rows_bboxes: list[tuple] = []
    cols_bboxes: list[tuple] = []
    all_bboxes: list[tuple] = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue

        x1 = max(0.0, cx - bw / 2)
        y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2)
        y2 = min(1.0, cy + bh / 2)

        cls_name = class_names[cls_id] if cls_id < len(class_names) else ""
        all_bboxes.append((x1, y1, x2, y2))

        if cls_name in _YOLO_ROW_CLASSES:
            rows_bboxes.append((x1, y1, x2, y2))
        elif cls_name in _YOLO_COL_CLASSES:
            cols_bboxes.append((x1, y1, x2, y2))

    if not all_bboxes:
        return []

    # Union bounding box for the whole table (values already clamped during parsing)
    tbl_x1 = min(b[0] for b in all_bboxes)
    tbl_y1 = min(b[1] for b in all_bboxes)
    tbl_x2 = max(b[2] for b in all_bboxes)
    tbl_y2 = max(b[3] for b in all_bboxes)
    tbl_w = max(1e-6, tbl_x2 - tbl_x1)
    tbl_h = max(1e-6, tbl_y2 - tbl_y1)

    # Row dividers: bottom edge of each row (except the last), normalised within table
    seen_rows: set[float] = set()
    table_rows: list[float] = []
    if rows_bboxes:
        rows_bboxes.sort(key=lambda b: b[1])
        for rx1, ry1, rx2, ry2 in rows_bboxes[:-1]:
            div_y = round(max(0.0, min(1.0, (ry2 - tbl_y1) / tbl_h)), 4)
            if div_y not in seen_rows:
                seen_rows.add(div_y)
                table_rows.append(div_y)

    # Column dividers: right edge of each column (except the last), normalised within table
    seen_cols: set[float] = set()
    table_cols: list[float] = []
    if cols_bboxes:
        cols_bboxes.sort(key=lambda b: b[0])
        for cx1, cy1, cx2, cy2 in cols_bboxes[:-1]:
            div_x = round(max(0.0, min(1.0, (cx2 - tbl_x1) / tbl_w)), 4)
            if div_x not in seen_cols:
                seen_cols.add(div_x)
                table_cols.append(div_x)

    element: dict = {
        "id": str(uuid.uuid4()),
        "label": "table",
        "content": "",
        "bbox_norm": {
            "x1": round(tbl_x1, 4),
            "y1": round(tbl_y1, 4),
            "x2": round(tbl_x2, 4),
            "y2": round(tbl_y2, 4),
        },
    }
    if table_rows:
        element["table_rows"] = sorted(table_rows)
    if table_cols:
        element["table_cols"] = sorted(table_cols)
    return [element]


@app.route("/import_folder", methods=["POST"])
def import_folder():
    """
    Import all images from a local folder into the labeler, optionally reading
    YOLO-format table annotations from a sibling ``labels/`` subdirectory.

    Accepts JSON:
    {
        "folder_path": str,   // absolute path or path relative to the project root
        "run_docling": bool   // optional, default false – run Docling on each
                              // image to detect non-table elements
    }
    Returns the same response format as ``/process``.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    folder_path_raw = str(data.get("folder_path", "")).strip()
    run_docling = bool(data.get("run_docling", False))

    if not folder_path_raw:
        return jsonify({"error": "folder_path is required"}), 400

    # Resolve path (relative paths are resolved against the project root)
    folder_path = Path(folder_path_raw)
    if not folder_path.is_absolute():
        folder_path = Path(BASE_DIR) / folder_path

    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({"error": f"Folder not found: {folder_path_raw}"}), 404

    # Locate images (prefer images/ subdirectory, fall back to folder root)
    img_dir = folder_path / "images"
    if img_dir.is_dir():
        image_files = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in _IMG_EXTENSIONS
        )
    else:
        image_files = sorted(
            f for f in folder_path.iterdir() if f.suffix.lower() in _IMG_EXTENSIONS
        )

    if not image_files:
        return jsonify({"error": f"No image files found in {folder_path_raw}"}), 404

    # Load YOLO class names if available
    classes_file = folder_path / "classes.txt"
    if classes_file.exists():
        class_names = [
            ln.strip()
            for ln in classes_file.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    else:
        class_names = []

    labels_dir = folder_path / "labels"

    # Build page list; also keep a map of page_no → dest_path for the Docling pass
    pages = []
    elements_by_page: dict[int, list] = {}
    dest_paths: dict[int, str] = {}

    for page_no, img_file in enumerate(image_files, 1):
        # Sanitize folder and image names to avoid path-traversal issues
        safe_folder = re.sub(r"[^\w\-]", "_", folder_path.name)
        safe_imgname = re.sub(r"[^\w\-.]", "_", img_file.name)
        dest_name = f"import_{safe_folder}_{safe_imgname}"
        dest_path = os.path.join(UPLOAD_FOLDER, dest_name)
        shutil.copy2(str(img_file), dest_path)
        dest_paths[page_no] = dest_path

        try:
            with Image.open(dest_path) as pil_img:
                img_w, img_h = pil_img.size
        except Exception:
            img_w, img_h = 0, 0

        # Read YOLO annotations if present
        elements: list = []
        label_file = labels_dir / (img_file.stem + ".txt") if labels_dir.is_dir() else None
        # Also check a flat .txt alongside the image (e.g. for flat-layout folders)
        if label_file is None or not label_file.exists():
            flat_label = img_file.parent / (img_file.stem + ".txt")
            if flat_label.exists():
                label_file = flat_label
        if label_file and label_file.exists():
            elements = _parse_yolo_annotations(label_file, class_names)

        elements_by_page[page_no] = elements
        pages.append({
            "page_num": page_no,
            "image_url": f"/page_image/{dest_name}",
            "width": img_w,
            "height": img_h,
            "elements": elements,
        })

    # Optionally run Docling on each image to detect non-table elements.
    # Docling's DocumentConverter accepts images directly.
    docling_warning = None
    if run_docling:
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter(allowed_formats=["image"])
            for page_no, img_file in enumerate(image_files, 1):
                dest_path = dest_paths[page_no]
                try:
                    result = converter.convert(dest_path)
                    doc = result.document

                    pd = {"width": pages[page_no - 1]["width"], "height": pages[page_no - 1]["height"]}
                    if pd["width"] == 0 or pd["height"] == 0:
                        continue

                    for item, _level in doc.iterate_items():
                        label = item.label.value if hasattr(item.label, "value") else str(item.label)
                        # Skip table labels – already sourced from YOLO annotations
                        if label == "table":
                            continue
                        content = _get_item_text(item)
                        if not hasattr(item, "prov") or not item.prov:
                            continue
                        prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                        for prov in prov_list:
                            raw_bbox = getattr(prov, "bbox", None)
                            if raw_bbox is None:
                                continue
                            bbox_norm = _bbox_to_norm(raw_bbox, pd["width"], pd["height"])
                            if bbox_norm is None:
                                continue
                            elements_by_page[page_no].append({
                                "id": str(uuid.uuid4()),
                                "label": label,
                                "content": content,
                                "bbox_norm": bbox_norm,
                            })
                    # Update pages list
                    pages[page_no - 1]["elements"] = elements_by_page[page_no]
                except Exception as img_err:
                    docling_warning = (docling_warning or "") + f" [{img_file.name}: {img_err}]"
        except Exception as e:
            docling_warning = f"Docling processing failed: {e}"

    resp: dict = {
        "filename": folder_path.name,
        "pdf_filename": "",
        "num_pages": len(pages),
        "pages": pages,
    }
    if docling_warning:
        resp["docling_warning"] = docling_warning.strip()
    return jsonify(resp)


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
