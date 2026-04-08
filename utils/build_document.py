"""
Build a visual document (PNG or PDF) from a DocTag string.

Each DocTag element is rendered at its exact location on the page,
reconstructing the spatial layout of the original document.

DocTag location tokens use a 0-500 normalised coordinate system.
This script maps them back to absolute pixel positions and draws
text, headers, tables, and placeholders for images.

Supported element types:
    text, section_header_level_1/2/3, title, caption, picture,
    otsl (table), list_item, formula, page_header, page_footer,
    code, footnote

Usage:
    # From a JSON file containing a "doctag" field
    python build_document.py --input output_doctags.json --output page.png

    # From a JSONL training file (renders each line as a separate page)
    python build_document.py --input training_data_val.jsonl --output pages/ --format png

    # Render a raw DocTag string directly
    python build_document.py --doctag "<doctag><text><loc_50>..." --output page.png

    # Output as PDF instead of PNG
    python build_document.py --input output_doctags.json --output page.pdf --format pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOC_SCALE = 500  # DocTag normalised coordinate range

# Default page dimensions (A4 at 150 DPI ≈ 1240 × 1754)
DEFAULT_PAGE_W = 1240
DEFAULT_PAGE_H = 1754

# Colours
BG_COLOUR = (255, 255, 255)
TEXT_COLOUR = (0, 0, 0)
HEADER_COLOUR = (20, 20, 100)
TABLE_LINE_COLOUR = (100, 100, 100)
TABLE_HEADER_BG = (230, 235, 245)
PICTURE_BG = (220, 220, 220)
CAPTION_COLOUR = (80, 80, 80)

# Font sizes (will be scaled relative to page height)
FONT_SIZES = {
    "text": 14,
    "title": 22,
    "section_header_level_1": 18,
    "section_header_level_2": 16,
    "section_header_level_3": 15,
    "caption": 12,
    "page_header": 11,
    "page_footer": 11,
    "footnote": 11,
    "list_item": 14,
    "formula": 14,
    "code": 13,
    "table_cell": 11,
}


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def _find_cyrillic_font() -> str | None:
    """Try to locate a Cyrillic-capable TrueType font on the system."""
    # Well-known paths
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # Ask fontconfig
    try:
        result = subprocess.run(
            ["fc-match", "DejaVu Sans", "--format=%{file}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and os.path.isfile(result.stdout.strip()):
            return result.stdout.strip()
    except Exception:
        pass

    return None


_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_FONT_PATH: str | None = _find_cyrillic_font()


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a PIL font object at the requested size, with caching."""
    if size not in _FONT_CACHE:
        if _FONT_PATH:
            _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def get_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a bold variant if available, otherwise fall back to regular."""
    bold_paths = [
        p.replace("Sans.ttf", "Sans-Bold.ttf")
        .replace("Regular", "Bold")
        .replace("-R.ttf", "-B.ttf")
        for p in [_FONT_PATH or ""]
        if p
    ]
    for bp in bold_paths:
        if os.path.isfile(bp):
            return ImageFont.truetype(bp, size)
    return get_font(size)


# ---------------------------------------------------------------------------
# DocTag parser
# ---------------------------------------------------------------------------

class DocElement(NamedTuple):
    tag: str
    x1: int  # 0-500 normalised
    y1: int
    x2: int
    y2: int
    content: str  # raw inner text (may contain cell tokens for otsl)



# Regex that captures each element: <tag><loc_N><loc_N><loc_N><loc_N>...content...</tag>
_ELEMENT_RE = re.compile(
    r"<(?P<tag>text|title|section_header_level_[123]|caption|picture|otsl|"
    r"list_item|formula|page_header|page_footer|code|footnote)>"
    r"<loc_(?P<x1>\d+)><loc_(?P<y1>\d+)><loc_(?P<x2>\d+)><loc_(?P<y2>\d+)>"
    r"(?P<content>.*?)"
    r"</(?P=tag)>",
    re.DOTALL,
)


def parse_doctag(doctag: str) -> list[DocElement]:
    """Parse a DocTag string into a list of positioned document elements."""
    elements: list[DocElement] = []
    for m in _ELEMENT_RE.finditer(doctag):
        elements.append(DocElement(
            tag=m.group("tag"),
            x1=int(m.group("x1")),
            y1=int(m.group("y1")),
            x2=int(m.group("x2")),
            y2=int(m.group("y2")),
            content=m.group("content"),
        ))
    return elements


# ---------------------------------------------------------------------------
# OTSL table parser
# ---------------------------------------------------------------------------

_CELL_TOKEN_RE = re.compile(
    r"<(ecel|ched|fcel|lcel|rhed|ucel)(\s[^>]*)?>" 
)



# New: parse per-cell <loc_*> tokens for OTSL tables
_CELL_LOC_RE = re.compile(r"<loc_(\d+)>" * 4)

def parse_otsl(content: str):
    """
    Parse OTSL cell tokens into a 2-D grid of dicts with cell_type, text, colspan, rowspan, and location.
    Each cell is expected to start with 4 <loc_*> tokens.
    """
    rows = []
    current_row = []
    parts = re.split(r"(<(?:ecel|ched|fcel|lcel|rhed|ucel|nl)(?:\s[^>]*)?>)", content)
    current_type = None
    current_colspan = 1
    current_rowspan = 1
    text_accum = ""
    cell_loc = None

    def flush_cell():
        nonlocal text_accum, current_type, current_colspan, current_rowspan, cell_loc
        if current_type is not None:
            # Extract <loc_*> tokens from text_accum
            loc_match = _CELL_LOC_RE.match(text_accum)
            if loc_match:
                x1, y1, x2, y2 = map(int, loc_match.groups())
                cell_text = text_accum[loc_match.end():].strip()
                loc = (x1, y1, x2, y2)
            else:
                cell_text = text_accum.strip()
                loc = None
            current_row.append({
                "type": current_type,
                "text": cell_text,
                "colspan": current_colspan,
                "rowspan": current_rowspan,
                "loc": loc,
            })
            current_type = None
            text_accum = ""
            current_colspan = 1
            current_rowspan = 1
            cell_loc = None

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == "<nl>":
            flush_cell()
            if current_row:
                rows.append(current_row)
                current_row = []
        else:
            m = _CELL_TOKEN_RE.fullmatch(part)
            if m:
                flush_cell()
                current_type = m.group(1)
                attrs_str = m.group(2) or ""
                cspan_m = re.search(r'cspan\s*=\s*["\']?(\d+)["\']?', attrs_str)
                rspan_m = re.search(r'rspan\s*=\s*["\']?(\d+)["\']?', attrs_str)
                current_colspan = int(cspan_m.group(1)) if cspan_m else 1
                current_rowspan = int(rspan_m.group(1)) if rspan_m else 1
            else:
                text_accum += part
    flush_cell()
    if current_row:
        rows.append(current_row)
    return rows


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _loc_to_px(
    loc: int, page_dim: int, margin: int = 0, scale: int = LOC_SCALE
) -> int:
    """Convert a normalised location token to a pixel coordinate."""
    return margin + int(loc / scale * (page_dim - 2 * margin))


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_text_element(
    draw: ImageDraw.ImageDraw,
    elem: DocElement,
    page_w: int,
    page_h: int,
    margin: int,
) -> None:
    """Render a text-type element at its DocTag location."""
    px1 = _loc_to_px(elem.x1, page_w, margin)
    py1 = _loc_to_px(elem.y1, page_h, margin)
    px2 = _loc_to_px(elem.x2, page_w, margin)
    py2 = _loc_to_px(elem.y2, page_h, margin)

    box_w = max(px2 - px1, 20)
    box_h = max(py2 - py1, 12)

    tag = elem.tag
    font_size = FONT_SIZES.get(tag, 14)
    # Scale font size relative to page height (base: 1754px)
    font_size = max(8, int(font_size * (page_h / 1754)))

    is_header = tag in ("title", "section_header_level_1",
                        "section_header_level_2", "section_header_level_3")
    colour = HEADER_COLOUR if is_header else TEXT_COLOUR
    if tag == "caption":
        colour = CAPTION_COLOUR

    font = get_bold_font(font_size) if is_header else get_font(font_size)

    text = elem.content.strip()
    if tag == "list_item":
        text = "• " + text

    lines = _wrap_text(text, font, box_w)
    line_height = font_size + 3
    y = py1
    for line in lines:
        if y + line_height > py2 + line_height:
            break  # clip to bounding box
        draw.text((px1, y), line, fill=colour, font=font)
        y += line_height


def _draw_picture(
    draw: ImageDraw.ImageDraw,
    elem: DocElement,
    page_w: int,
    page_h: int,
    margin: int,
) -> None:
    """Render a picture placeholder at its DocTag location."""
    px1 = _loc_to_px(elem.x1, page_w, margin)
    py1 = _loc_to_px(elem.y1, page_h, margin)
    px2 = _loc_to_px(elem.x2, page_w, margin)
    py2 = _loc_to_px(elem.y2, page_h, margin)

    draw.rectangle([px1, py1, px2, py2], fill=PICTURE_BG, outline=TABLE_LINE_COLOUR)
    # Draw an X and label
    draw.line([px1, py1, px2, py2], fill=TABLE_LINE_COLOUR, width=1)
    draw.line([px1, py2, px2, py1], fill=TABLE_LINE_COLOUR, width=1)
    font = get_font(max(8, int(12 * page_h / 1754)))
    draw.text(
        (px1 + 4, py1 + 4), "[picture]",
        fill=TABLE_LINE_COLOUR, font=font,
    )


def _draw_table(
    draw: ImageDraw.ImageDraw,
    elem: DocElement,
    page_w: int,
    page_h: int,
    margin: int,
) -> None:
    """Render an OTSL table at its DocTag location, with grid lines and cell text.

    Handles merged cells:
        lcel = left-merged (extends the cell to its left horizontally)
        ucel = up-merged   (extends the cell above it vertically)
    """
    px1 = _loc_to_px(elem.x1, page_w, margin)
    py1 = _loc_to_px(elem.y1, page_h, margin)
    px2 = _loc_to_px(elem.x2, page_w, margin)
    py2 = _loc_to_px(elem.y2, page_h, margin)

    rows = parse_otsl(elem.content)
    if not rows:
        draw.rectangle([px1, py1, px2, py2], outline=TABLE_LINE_COLOUR, width=1)
        return

    font_size = max(8, int(FONT_SIZES["table_cell"] * page_h / 1754))
    font = get_font(font_size)
    bold_font = get_bold_font(font_size)

    # Build a grid to track merged cells
    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0
    cell_drawn = [[False for _ in range(num_cols)] for _ in range(num_rows)]

    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            if cell_drawn[ri][ci]:
                continue  # already covered by a merge
            cell_type = cell["type"]
            cell_text = cell["text"]
            loc = cell["loc"]
            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)
            # Skip merged cells (lcel, ucel)
            if cell_type in ("lcel", "ucel"):
                continue
            # Find the merged area if colspan/rowspan > 1
            if loc:
                # Expand to cover all merged cells
                end_ri = ri + rowspan - 1
                end_ci = ci + colspan - 1
                # Use the loc of the top-left and bottom-right cells
                start_loc = loc
                end_loc = loc
                if rowspan > 1 or colspan > 1:
                    # Find the bottom-right cell's loc
                    for rj in range(ri, ri + rowspan):
                        for cj in range(ci, ci + colspan):
                            if rj < num_rows and cj < len(rows[rj]):
                                cell_drawn[rj][cj] = True
                                candidate = rows[rj][cj].get("loc")
                                if candidate:
                                    end_loc = candidate
                else:
                    cell_drawn[ri][ci] = True
                cx1 = _loc_to_px(start_loc[0], page_w, margin)
                cy1 = _loc_to_px(start_loc[1], page_h, margin)
                cx2 = _loc_to_px(end_loc[2], page_w, margin)
                cy2 = _loc_to_px(end_loc[3], page_h, margin)
            else:
                continue

            # Background for header cells
            if cell_type in ("ched", "rhed"):
                draw.rectangle([cx1, cy1, cx2, cy2], fill=TABLE_HEADER_BG)

            # Cell border
            draw.rectangle([cx1, cy1, cx2, cy2], outline=TABLE_LINE_COLOUR, width=1)

            # Cell text
            if cell_text:
                cell_content_w = cx2 - cx1 - 6
                cfont = bold_font if cell_type in ("ched", "rhed") else font
                lines = _wrap_text(cell_text, cfont, max(cell_content_w, 10))
                ty = cy1 + 3
                for line in lines:
                    if ty + font_size + 2 > cy2:
                        draw.text((cx1 + 3, ty - font_size), "…", fill=TEXT_COLOUR, font=cfont)
                        break
                    draw.text((cx1 + 3, ty), line, fill=TEXT_COLOUR, font=cfont)
                    ty += font_size + 2

    # Outer border (thicker)
    draw.rectangle([px1, py1, px2, py2], outline=TABLE_LINE_COLOUR, width=2)


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

def render_doctag(
    doctag: str,
    page_w: int = DEFAULT_PAGE_W,
    page_h: int = DEFAULT_PAGE_H,
    margin: int = 20,
) -> Image.Image:
    """
    Parse a DocTag string and render all elements onto a PIL Image,
    placing each element at its encoded location.
    """
    elements = parse_doctag(doctag)
    img = Image.new("RGB", (page_w, page_h), BG_COLOUR)
    draw = ImageDraw.Draw(img)

    for elem in elements:
        if elem.tag == "picture":
            _draw_picture(draw, elem, page_w, page_h, margin)
        elif elem.tag == "otsl":
            _draw_table(draw, elem, page_w, page_h, margin)
        else:
            _draw_text_element(draw, elem, page_w, page_h, margin)

    return img


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_doctags_from_file(path: str) -> list[str]:
    """
    Load DocTag strings from a file. Supports:
      - .json  → expects a string or {"doctag": "..."} or list thereof
      - .jsonl → one {"doctag": "..."} per line
      - .txt   → raw DocTag string
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()

    if p.suffix == ".jsonl":
        tags = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "doctag" in obj:
                tags.append(obj["doctag"])
            elif isinstance(obj, str):
                tags.append(obj)
        return tags

    if p.suffix == ".json":
        data = json.loads(raw)
        # Could be a bare string
        if isinstance(data, str):
            return [data]
        # Or a dict with "doctag" key
        if isinstance(data, dict) and "doctag" in data:
            return [data["doctag"]]
        # Or a list
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict) and "doctag" in item:
                    result.append(item["doctag"])
            return result

    # Fallback: treat as raw DocTag text
    return [raw]


def save_output(
    images: list[Image.Image],
    output_path: str,
    fmt: str = "png",
) -> list[str]:
    """
    Save rendered pages to files.
    Returns list of written file paths.

    If output_path is a directory or multiple images exist, pages are saved
    individually.  A single image with a .pdf extension triggers PDF output.
    """
    out = Path(output_path)
    written: list[str] = []

    if fmt == "pdf":
        _save_as_pdf(images, out)
        written.append(str(out))
    elif len(images) == 1 and out.suffix.lower() in (".png", ".jpg", ".jpeg"):
        out.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(str(out))
        written.append(str(out))
    else:
        # Multiple pages → directory
        out.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            fp = out / f"page_{i + 1}.{fmt}"
            img.save(str(fp))
            written.append(str(fp))

    return written


def _save_as_pdf(images: list[Image.Image], out: Path) -> None:
    """Save a list of PIL images as a multi-page PDF."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(str(out), "PDF", resolution=150.0)
    else:
        images[0].save(
            str(out), "PDF", resolution=150.0,
            save_all=True, append_images=images[1:],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a document (PNG/PDF) from a DocTag string, "
                    "placing every element at its encoded location."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i",
        help="Path to a .json / .jsonl / .txt file containing DocTag(s)",
    )
    group.add_argument(
        "--doctag", "-d",
        help="Raw DocTag string to render",
    )
    parser.add_argument(
        "--output", "-o", default="doctag_render.png",
        help="Output file or directory (default: doctag_render.png)",
    )
    parser.add_argument(
        "--format", "-f", choices=["png", "pdf"], default=None,
        help="Output format. Inferred from --output extension if omitted.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_PAGE_W, help="Page width in px")
    parser.add_argument("--height", type=int, default=DEFAULT_PAGE_H, help="Page height in px")
    parser.add_argument("--margin", type=int, default=20, help="Page margin in px")
    args = parser.parse_args()

    # Determine format
    fmt = args.format
    if fmt is None:
        ext = Path(args.output).suffix.lower()
        fmt = "pdf" if ext == ".pdf" else "png"

    # Load DocTag strings
    if args.doctag:
        doctags = [args.doctag]
    else:
        doctags = load_doctags_from_file(args.input)

    if not doctags:
        print("No DocTag strings found in input.", file=sys.stderr)
        sys.exit(1)

    print(f"Rendering {len(doctags)} page(s) …")

    images = []
    for i, dt in enumerate(doctags):
        img = render_doctag(dt, args.width, args.height, args.margin)
        images.append(img)
        print(f"  Page {i + 1}: {len(parse_doctag(dt))} elements")

    written = save_output(images, args.output, fmt)
    for fp in written:
        print(f"  → {fp}")
    print("Done.")


if __name__ == "__main__":
    main()
