import os
import json
import time
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

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
# Cell Extractor (from detected table region)
# ─────────────────────────────────────────────
class CellExtractor:
    def extract_cells(self, table_img: np.ndarray) -> List[Cell]:
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        # Detect horizontal lines
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
            # Skip full image contour and tiny boxes
            if area < 500 or (w > img_w * 0.95 and h > img_h * 0.95):
                continue
            cells_raw.append((x, y, w, h))

        if not cells_raw:
            return []

        # Sort by row then col
        cells_raw.sort(key=lambda c: (c[1], c[0]))

        # Cluster into rows/cols using y-threshold
        row_threshold = 10
        col_threshold = 10
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
        for row_idx, row in enumerate(rows):
            for col_idx, (x, y, w, h) in enumerate(row):
                # Detect merges by comparing size to average cell size
                avg_w = np.mean([c[2] for c in cells_raw])
                avg_h = np.mean([c[3] for c in cells_raw])
                colspan = max(1, round(w / avg_w))
                rowspan = max(1, round(h / avg_h))
                cells.append(Cell(
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    text="",
                    x=float(x),
                    y=float(y),
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

        # Build grid map to track occupied cells
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
    

class TableVisualizer:
    def draw(self, image: np.ndarray, detections: List[TableDetection]) -> np.ndarray:
        vis = image.copy()

        for det in detections:
            # Draw table bounding box
            tx, ty, tw, th = int(det.x), int(det.y), int(det.width), int(det.height)
            cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2)
            cv2.putText(vis, f"{det.model} ({det.confidence:.2f})",
                        (tx, max(0, ty - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for cell in det.cells:
                cx, cy = int(cell.x), int(cell.y)
                cw, ch = int(cell.width), int(cell.height)

                # Draw cell border
                cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (0, 180, 0), 1)

                # Draw merge indicator
                if cell.rowspan > 1 or cell.colspan > 1:
                    cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (255, 140, 0), 2)
                    merge_label = f"r{cell.rowspan}c{cell.colspan}"
                    cv2.putText(vis, merge_label,
                                (cx + 4, cy + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 140, 0), 1)
                # Draw row/col index
                label = f"R{cell.row}C{cell.col}"
                cv2.putText(vis, label,
                            (cx + 4, cy + ch - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 200), 1)

                # Draw cell content if available
                if cell.text:
                    cv2.putText(vis, cell.text[:20],
                                (cx + 4, cy + ch // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        return vis
    def draw_pdf(
        self,
        pdf_path: str,
        results: dict,
        detections_by_page: dict,
        out_dir: str = "table_detections",
        dpi: int = 200
    ):
        os.makedirs(out_dir, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=dpi)

        for page_data in results["pages"]:
            page_num = page_data["page"]
            image = cv2.cvtColor(np.array(images[page_num - 1]), cv2.COLOR_RGB2BGR)
            detections = detections_by_page.get(page_num, [])
            vis = self.draw(image, detections)
            out_path = os.path.join(out_dir, f"page_{page_num:03d}.png")
            cv2.imwrite(out_path, vis)
            print(f"Saved: {out_path}")

        print(f"All visualizations saved to: {out_dir}")



# ─────────────────────────────────────────────
# Table Detector (DeepDeSRT style)
# ─────────────────────────────────────────────
class TableDetector:
    def __init__(self):
        self.cell_extractor = CellExtractor()
        self.html_builder = HTMLTableBuilder()
        self.visualizer = TableVisualizer()
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

            # Crop table region
            table_img = image[y:y+h, x:x+w]
            cells = self.cell_extractor.extract_cells(table_img)

            # Offset cell coordinates to full image space
            for cell in cells:
                cell.x += x
                cell.y += y

            num_rows = max((cell.row for cell in cells), default=0) + 1
            num_cols = max((cell.col for cell in cells), default=0) + 1
            html = self.html_builder.build(cells, num_rows, num_cols)

            detections.append(TableDetection(
                page=page_num,
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h),
                confidence=0.75,
                model="DeepDeSRT",
                cells=cells,
                html=html
            ))
        return detections

    def detect_pdf(self, pdf_path: str, dpi: int = 200) -> dict:
        images = convert_from_path(pdf_path, dpi=dpi)
        results = {"source": pdf_path, "pages": []}
        all_html = []
        detections_by_page = {}

        for idx, pil_img in enumerate(images):
            page_num = idx + 1
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            detections = self.detect(image, page_num)
            detections_by_page[page_num] = detections

            page_tables = []
            for t_idx, det in enumerate(detections):
                page_tables.append({
                    "table_index": t_idx + 1,
                    "page": det.page,
                    "bbox": {"x": det.x, "y": det.y, "width": det.width, "height": det.height},
                    "confidence": det.confidence,
                    "model": det.model,
                    "num_rows": max((c.row for c in det.cells), default=0) + 1,
                    "num_cols": max((c.col for c in det.cells), default=0) + 1,
                    "cells": [asdict(c) for c in det.cells],
                    "html": det.html
                })
                all_html.append(f"<h3>Page {page_num} - Table {t_idx + 1}</h3>\n{det.html}")

            results["pages"].append({"page": page_num, "tables": page_tables})
            print(f"Page {page_num}: {len(detections)} table(s) detected")

        # Save combined HTML
        with open("tables_output.html", "w", encoding="utf-8") as f:
            f.write("<html><body>\n")
            f.write("\n".join(all_html))
            f.write("\n</body></html>")

        # Draw detections on images
        self.visualizer.draw_pdf(pdf_path, results, detections_by_page, out_dir="table_detections", dpi=dpi)

        return results

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = "page_001.pdf"
    detector = TableDetector()

    start = time.time()
    results = detector.detect_pdf(pdf_path, dpi=200)
    print(f"Detection completed in {time.time() - start:.2f}s")


    print("Saved table_detections.json")