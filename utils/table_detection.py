import os
import json
import time
import cv2
import numpy as np
import requests
import base64
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import argparse

# ─────────────────────────────────────────────
# Google Vision OCR
# ─────────────────────────────────────────────
class GoogleVisionOCR:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    def _image_to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")

    def recognize(self, image: np.ndarray) -> str:
        """Run OCR on a single image/cell crop and return text."""
        if image is None or image.size == 0:
            return ""
        # Skip tiny crops
        h, w = image.shape[:2]
        if w < 5 or h < 5:
            return ""

        b64 = self._image_to_base64(image)
        payload = {
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "TEXT_DETECTION"}],
                "imageContext": {"languageHints": []}
            }]
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            annotations = data.get("responses", [{}])[0].get("textAnnotations", [])
            if annotations:
                return annotations[0].get("description", "").strip().replace("\n", " ")
        except Exception as e:
            print(f"[GoogleVisionOCR] Error: {e}")
        return ""

    def recognize_cells(self, page_image: np.ndarray, cells: List) -> List:
        """Crop each cell from page image, run OCR, fill cell.text."""
        for cell in cells:
            x, y, w, h = int(cell.x), int(cell.y), int(cell.width), int(cell.height)
            # Add small padding
            pad = 2
            x1 = max(0, x + pad)
            y1 = max(0, y + pad)
            x2 = min(page_image.shape[1], x + w - pad)
            y2 = min(page_image.shape[0], y + h - pad)
            crop = page_image[y1:y2, x1:x2]
            cell.text = self.recognize(crop)
            print(f"  Cell R{cell.row}C{cell.col}: '{cell.text}'")
        return cells


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────
@dataclass
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str
    x: float
    y: float
    width: float
    height: float

@dataclass
class TableDetection:
    page: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    model: str
    cells: List[Cell]
    html: str = ""


# ─────────────────────────────────────────────
# Cell Extractor
# ─────────────────────────────────────────────
class CellExtractor:
    def extract_cells(self, table_img: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> List[Cell]:
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        grid = cv2.add(h_lines, v_lines)
        dilated = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = table_img.shape[:2]
        cells_raw = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 500 or (w > img_w * 0.95 and h > img_h * 0.95):
                continue
            cells_raw.append((x, y, w, h))

        if not cells_raw:
            return []

        cells_raw.sort(key=lambda c: (c[1], c[0]))
        row_threshold = 10
        rows = []
        current_row = [cells_raw[0]]
        for cell in cells_raw[1:]:
            if abs(cell[1] - current_row[0][1]) <= row_threshold:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]
        rows.append(sorted(current_row, key=lambda c: c[0]))

        cells = []
        avg_w = np.mean([c[2] for c in cells_raw])
        avg_h = np.mean([c[3] for c in cells_raw])
        for row_idx, row in enumerate(rows):
            for col_idx, (x, y, w, h) in enumerate(row):
                colspan = max(1, round(w / avg_w))
                rowspan = max(1, round(h / avg_h))
                cells.append(Cell(
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    text="",
                    x=float(x + offset_x),
                    y=float(y + offset_y),
                    width=float(w),
                    height=float(h)
                ))
        return cells


# ─────────────────────────────────────────────
# HTML Table Builder
# ─────────────────────────────────────────────
class HTMLTableBuilder:
    def build(self, cells: List[Cell], num_rows: int, num_cols: int) -> str:
        if not cells:
            return "<table><tr><td>No cells detected</td></tr></table>"

        grid = [[None] * num_cols for _ in range(num_rows)]
        for cell in cells:
            r, c = cell.row, cell.col
            if r < num_rows and c < num_cols:
                grid[r][c] = cell

        html = ['<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;">']
        occupied = set()

        for r in range(num_rows):
            html.append("  <tr>")
            for c in range(num_cols):
                if (r, c) in occupied:
                    continue
                cell = grid[r][c]
                if cell is None:
                    html.append("    <td></td>")
                    continue
                attrs = []
                if cell.rowspan > 1:
                    attrs.append(f'rowspan="{cell.rowspan}"')
                    for dr in range(cell.rowspan):
                        for dc in range(cell.colspan):
                            if dr != 0 or dc != 0:
                                occupied.add((r + dr, c + dc))
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                    for dc in range(1, cell.colspan):
                        occupied.add((r, c + dc))
                attr_str = " " + " ".join(attrs) if attrs else ""
                text = cell.text or ""
                html.append(f"    <td{attr_str}>{text}</td>")
            html.append("  </tr>")
        html.append("</table>")
        return "\n".join(html)


# ─────────────────────────────────────────────
# Table Detector
# ─────────────────────────────────────────────
class TableDetector:
    def __init__(self, google_api_key: Optional[str] = None):
        self.cell_extractor = CellExtractor()
        self.html_builder = HTMLTableBuilder()
        self.ocr = GoogleVisionOCR(google_api_key) if google_api_key else None

    def detect(self, image: np.ndarray, page_num: int) -> List[TableDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        grid = cv2.add(h_lines, v_lines)
        dilated = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 100 or h < 50:
                continue

            table_img = image[y:y+h, x:x+w]
            # Pass offset so cell coords are in full image space
            cells = self.cell_extractor.extract_cells(table_img, offset_x=x, offset_y=y)

            # Run Google Vision OCR on each cell
            if self.ocr and cells:
                print(f"  Running Google Vision OCR on {len(cells)} cells...")
                cells = self.ocr.recognize_cells(image, cells)

            num_rows = max((cell.row for cell in cells), default=0) + 1
            num_cols = max((cell.col for cell in cells), default=0) + 1
            html = self.html_builder.build(cells, num_rows, num_cols)

            detections.append(TableDetection(
                page=page_num,
                x=float(x), y=float(y),
                width=float(w), height=float(h),
                confidence=0.75,
                model="DeepDeSRT",
                cells=cells,
                html=html
            ))
        return detections

    def detect_source(self, source: str, dpi: int = 200) -> dict:
        ext = Path(source).suffix.lower()
        if ext == ".pdf":
            pil_images = convert_from_path(source, dpi=dpi)
            images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]
        else:
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Cannot load: {source}")
            images = [img]

        results = {"source": source, "pages": []}
        all_html = []
        out_dir = "table_detections"
        os.makedirs(out_dir, exist_ok=True)

        for idx, image in enumerate(images):
            page_num = idx + 1
            print(f"\nProcessing page {page_num}...")
            detections = self.detect(image, page_num)
            page_tables = []

            for t_idx, det in enumerate(detections):
                num_rows = max((c.row for c in det.cells), default=0) + 1
                num_cols = max((c.col for c in det.cells), default=0) + 1
                page_tables.append({
                    "table_index": t_idx + 1,
                    "page": det.page,
                    "bbox": {"x": det.x, "y": det.y, "width": det.width, "height": det.height},
                    "confidence": det.confidence,
                    "model": det.model,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "cells": [asdict(c) for c in det.cells],
                    "html": det.html
                })
                all_html.append(f"<h3>Page {page_num} - Table {t_idx + 1}</h3>\n{det.html}")

            results["pages"].append({"page": page_num, "tables": page_tables})

            # Draw visualizations
            vis = self._draw(image, detections)
            cv2.imwrite(os.path.join(out_dir, f"page_{page_num:03d}.png"), vis)
            print(f"Page {page_num}: {len(detections)} table(s) detected")

        with open("tables_output.html", "w", encoding="utf-8") as f:
            f.write("<html><body>\n")
            f.write("\n".join(all_html))
            f.write("\n</body></html>")
        print("\nSaved tables_output.html")
        return results

    def _draw(self, image: np.ndarray, detections: List[TableDetection]) -> np.ndarray:
        vis = image.copy()
        for det in detections:
            tx, ty, tw, th = int(det.x), int(det.y), int(det.width), int(det.height)
            cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2)
            cv2.putText(vis, f"{det.model} ({det.confidence:.2f})",
                        (tx, max(0, ty - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for cell in det.cells:
                cx, cy = int(cell.x), int(cell.y)
                cw, ch = int(cell.width), int(cell.height)
                color = (255, 140, 0) if (cell.rowspan > 1 or cell.colspan > 1) else (0, 180, 0)
                cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), color, 1)
                label = f"R{cell.row}C{cell.col}"
                cv2.putText(vis, label, (cx + 4, cy + ch - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 200), 1)
                if cell.text:
                    cv2.putText(vis, cell.text[:25], (cx + 4, cy + ch // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        return vis


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Detector with Google Vision OCR")
    parser.add_argument("--source", type=str, default="tables.pdf", help="PDF or image path")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering")
    parser.add_argument("--api-key", type=str, default=None, help="Google Vision API key")
    parser.add_argument("--out", type=str, default="table_detections.json", help="Output JSON path")
    args = parser.parse_args()

    detector = TableDetector(google_api_key=args.api_key)
    start = time.time()
    results = detector.detect_source(args.source, dpi=args.dpi)
    print(f"\nDetection completed in {time.time() - start:.2f}s")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {args.out}")