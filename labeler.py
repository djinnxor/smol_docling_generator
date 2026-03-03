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
    skip_labels = {"picture", "table"}

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
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        try:
            from rapidocr_openvino import RapidOCR
        except ImportError:
            raise ImportError(
                "RapidOCR is not installed. Run: pip install rapidocr-onnxruntime"
            )

    import numpy as np

    engine = RapidOCR()
    skip_labels = {"picture", "table"}

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
                if result:
                    lines = [r[1] for r in result if r and len(r) > 1 and r[1]]
                    text = "\n".join(lines)
                    if text.strip():
                        el["content"] = text.strip()
            except Exception:
                pass

    return elements_by_page

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
    Run Google OCR on a specific bounding box region of a page image.
    Expects JSON: { image_filename: str, bbox_norm: {x1, y1, x2, y2} }
    Returns: { text: str }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    image_filename = data.get("image_filename", "")
    bbox_norm = data.get("bbox_norm", {})

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
        tmp_path = os.path.join(UPLOAD_FOLDER, f"_ocr_{uuid.uuid4().hex}.png")
        cropped.save(tmp_path, "PNG")
        try:
            api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
            ocr = GoogleOCR(api_key=api_key)
            text = ocr.google_vision_ocr(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return jsonify({"text": text.strip()})
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
    (table_detection.py) and populate cell text with Google OCR.
    Returns table elements in the same format as /process.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pdf_filename = data.get("pdf_filename", "")
    page_no = int(data.get("page_no", 1))

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

        # OCR each cell using Google Vision
        api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
        ocr = GoogleOCR(api_key=api_key)

        for det in detections:
            for cell in det.cells:
                x1 = max(0, int(cell.x))
                y1 = max(0, int(cell.y))
                x2 = min(img_w, int(cell.x + cell.width))
                y2 = min(img_h, int(cell.y + cell.height))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                cropped = pil_img.crop((x1, y1, x2, y2))
                tmp_path = os.path.join(UPLOAD_FOLDER, f"_cell_{uuid.uuid4().hex}.png")
                cropped.save(tmp_path, "PNG")
                try:
                    text = ocr.google_vision_ocr(tmp_path)
                    if text.strip():
                        cell.text = text.strip()
                except Exception:
                    pass
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

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


# ---------------------------------------------------------------------------
# COCO-table helper
# ---------------------------------------------------------------------------

def _coco_to_table_element(annotations):
    """Convert a list of (class_id, cx, cy, w, h) COCO table-structure annotations
    to a labeler element dict containing ``table_html``, ``table_rows``, and
    ``table_cols``.

    Class IDs:
      0 = table-row
      1 = table-column
      2 = table-spanning-cell  (layout only; cells stay empty)
      3 = table-column-header
    """
    rows_raw = []
    cols_raw = []
    header_regions = []

    for cls, cx, cy, w, h in annotations:
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if cls == 0:
            rows_raw.append((x1, y1, x2, y2))
        elif cls == 1:
            cols_raw.append((x1, y1, x2, y2))
        elif cls == 3:
            header_regions.append((x1, y1, x2, y2))

    if not rows_raw and not header_regions:
        return None
    if not cols_raw:
        return None

    rows_raw.sort(key=lambda b: (b[1] + b[3]) / 2)
    cols_raw.sort(key=lambda b: (b[0] + b[2]) / 2)

    def _overlaps_y(a, b, threshold=0.3):
        inter = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        shorter = min(a[3] - a[1], b[3] - b[1])
        return shorter > 0 and inter / shorter >= threshold

    # Merge class-0 rows with class-3 header regions; insert unmatched header regions
    matched_header_idx: set = set()
    all_rows = []
    for row_bbox in rows_raw:
        is_h = False
        for i, hr in enumerate(header_regions):
            if _overlaps_y(row_bbox, hr):
                is_h = True
                matched_header_idx.add(i)
        all_rows.append((*row_bbox, is_h))
    for i, hr in enumerate(header_regions):
        if i not in matched_header_idx:
            all_rows.append((*hr, True))

    all_rows.sort(key=lambda r: (r[1] + r[3]) / 2)

    # Table bounding box
    all_bboxes = [(r[0], r[1], r[2], r[3]) for r in all_rows] + cols_raw
    tbl_x1 = max(0.0, min(1.0, min(b[0] for b in all_bboxes)))
    tbl_y1 = max(0.0, min(1.0, min(b[1] for b in all_bboxes)))
    tbl_x2 = max(0.0, min(1.0, max(b[2] for b in all_bboxes)))
    tbl_y2 = max(0.0, min(1.0, max(b[3] for b in all_bboxes)))

    n_cols = len(cols_raw)

    # Build empty-cell HTML
    html_rows = []
    for row in all_rows:
        cells = "".join(f"<{'th' if row[4] else 'td'}></{'th' if row[4] else 'td'}>" for _ in range(n_cols))
        html_rows.append(f"<tr>{cells}</tr>")
    table_html = "<table>" + "".join(html_rows) + "</table>"

    # Separator positions (image-normalised coords)
    table_rows = []
    for i in range(len(all_rows) - 1):
        mid_y = (all_rows[i][3] + all_rows[i + 1][1]) / 2
        table_rows.append(round(max(0.0, min(1.0, mid_y)), 4))

    table_cols = []
    for i in range(len(cols_raw) - 1):
        mid_x = (cols_raw[i][2] + cols_raw[i + 1][0]) / 2
        table_cols.append(round(max(0.0, min(1.0, mid_x)), 4))

    return {
        "id": str(uuid.uuid4()),
        "label": "table",
        "content": "",
        "bbox_norm": {
            "x1": round(tbl_x1, 4),
            "y1": round(tbl_y1, 4),
            "x2": round(tbl_x2, 4),
            "y2": round(tbl_y2, 4),
        },
        "table_html": table_html,
        "table_rows": table_rows,
        "table_cols": table_cols,
    }


# ---------------------------------------------------------------------------
# Import folder of images + COCO table labels
# ---------------------------------------------------------------------------

@app.route("/import_folder", methods=["POST"])
def import_folder():
    """Import all images from a local folder, matching each to a COCO-format
    .txt label file (same stem) to build table elements.

    Accepts JSON: ``{ "folder_path": "/abs/or/relative/path" }``

    Returns the same JSON structure as ``/process``.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    folder_path = data.get("folder_path", "").strip()
    if not folder_path:
        return jsonify({"error": "folder_path is required"}), 400

    # Resolve relative to BASE_DIR
    folder = (
        Path(folder_path) if Path(folder_path).is_absolute()
        else Path(BASE_DIR) / folder_path
    )
    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": f"Folder not found: {folder_path}"}), 400

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    img_files = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in image_extensions
    )
    if not img_files:
        return jsonify({"error": "No image files found in folder"}), 400

    pages = []
    for page_no, img_file in enumerate(img_files, 1):
        dest_name = secure_filename(img_file.name)
        dest_path = os.path.join(UPLOAD_FOLDER, dest_name)
        # Copy only if missing or size differs
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) != img_file.stat().st_size:
            shutil.copy2(str(img_file), dest_path)

        try:
            with Image.open(dest_path) as pil_img:
                img_w, img_h = pil_img.size
        except Exception:
            img_w, img_h = 0, 0

        # Parse optional COCO label file
        txt_file = folder / (img_file.stem + ".txt")
        elements = []
        if txt_file.exists():
            annotations = []
            with open(txt_file, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls = int(parts[0])
                        cx, cy, w, h = (
                            float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]),
                        )
                        annotations.append((cls, cx, cy, w, h))
                    except ValueError:
                        continue
            if annotations:
                el = _coco_to_table_element(annotations)
                if el:
                    elements.append(el)

        pages.append({
            "page_num": page_no,
            "image_url": f"/page_image/{dest_name}",
            "width": img_w,
            "height": img_h,
            "elements": elements,
        })

    return jsonify({
        "filename": folder.name,
        "pdf_filename": "",
        "num_pages": len(pages),
        "pages": pages,
    })


# ---------------------------------------------------------------------------
# OCR all cells in a table element
# ---------------------------------------------------------------------------

@app.route("/ocr_table_cells", methods=["POST"])
def ocr_table_cells():
    """Run Google OCR on every cell of a table element and return updated HTML.

    Accepts JSON::

        {
            "image_filename": "image.jpg",
            "bbox_norm": {"x1": ..., "y1": ..., "x2": ..., "y2": ...},
            "table_rows": [y1, y2, ...],   # separator y-positions (normalised)
            "table_cols": [x1, x2, ...],   # separator x-positions (normalised)
            "table_html": "<table>...</table>"  # current HTML (preserves header tags)
        }

    Returns::

        { "table_html": "<table>...</table>" }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    image_filename = data.get("image_filename", "")
    bbox_norm = data.get("bbox_norm", {})
    table_rows_sep = data.get("table_rows", [])
    table_cols_sep = data.get("table_cols", [])
    table_html_in = data.get("table_html", "")

    if not image_filename:
        return jsonify({"error": "image_filename is required"}), 400

    img_path = os.path.join(UPLOAD_FOLDER, secure_filename(Path(image_filename).name))
    if not os.path.exists(img_path):
        return jsonify({"error": f"Image not found: {image_filename}"}), 404

    try:
        img = Image.open(img_path)
        img_w, img_h = img.size

        tbl_x1 = bbox_norm.get("x1", 0)
        tbl_y1 = bbox_norm.get("y1", 0)
        tbl_x2 = bbox_norm.get("x2", 1)
        tbl_y2 = bbox_norm.get("y2", 1)

        row_bounds = [tbl_y1] + list(table_rows_sep) + [tbl_y2]
        col_bounds = [tbl_x1] + list(table_cols_sep) + [tbl_x2]
        n_rows = len(row_bounds) - 1
        n_cols = len(col_bounds) - 1

        # Determine which rows are headers from the existing table_html
        header_row_indices: set = set()
        if table_html_in:
            try:
                tbl_elem = ET.fromstring(table_html_in)
                for ri, tr in enumerate(tbl_elem.findall("tr")):
                    if any(cell.tag == "th" for cell in tr):
                        header_row_indices.add(ri)
            except Exception:
                pass

        api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")
        ocr = GoogleOCR(api_key=api_key)

        grid_texts = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for ri in range(n_rows):
            y1_px = int(row_bounds[ri] * img_h)
            y2_px = int(row_bounds[ri + 1] * img_h)
            for ci in range(n_cols):
                x1_px = int(col_bounds[ci] * img_w)
                x2_px = int(col_bounds[ci + 1] * img_w)
                if x2_px - x1_px < 4 or y2_px - y1_px < 4:
                    continue
                # Small padding to avoid clipping text at cell edges
                pad = 2
                x1c = max(0, x1_px - pad)
                y1c = max(0, y1_px - pad)
                x2c = min(img_w, x2_px + pad)
                y2c = min(img_h, y2_px + pad)
                cropped = img.crop((x1c, y1c, x2c, y2c))
                tmp_path = os.path.join(UPLOAD_FOLDER, f"_cell_{uuid.uuid4().hex}.png")
                cropped.save(tmp_path, "PNG")
                try:
                    text = ocr.google_vision_ocr(tmp_path)
                    grid_texts[ri][ci] = text.strip()
                except Exception:
                    pass
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # Build updated table HTML, preserving header designation
        def _esc(s):
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        html_rows = []
        for ri in range(n_rows):
            cells = ""
            for ci in range(n_cols):
                tag = "th" if ri in header_row_indices else "td"
                cells += f"<{tag}>{_esc(grid_texts[ri][ci])}</{tag}>"
            html_rows.append(f"<tr>{cells}</tr>")
        new_table_html = "<table>" + "".join(html_rows) + "</table>"

        return jsonify({"table_html": new_table_html})

    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500


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
