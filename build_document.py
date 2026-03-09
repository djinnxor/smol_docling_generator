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

_DOCTAG_INNER_RE = re.compile(r"<doctag>(.*?)</doctag>", re.DOTALL)
_PAGEBREAK_SPLIT_RE = re.compile(r"\s*<page_break>\s*", re.IGNORECASE)


def split_doctag_on_pagebreaks(doctag: str) -> list[str]:
    """Split a combined multi-page doctag on ``<page_break>`` tokens.

    Returns a list of single-page ``<doctag>…</doctag>`` strings.  If the
    input contains no ``<page_break>``, a one-element list with the original
    doctag is returned unchanged.
    """
    if "<page_break>" not in doctag.lower():
        return [doctag]
    m = _DOCTAG_INNER_RE.search(doctag)
    if not m:
        return [doctag]
    inner = m.group(1)
    parts = _PAGEBREAK_SPLIT_RE.split(inner)
    result = [f"<doctag>{p.strip()}</doctag>" for p in parts if p.strip()]
    return result if result else [doctag]


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
    r"<(ecel|ched|fcel|lcel|rhed|ucel)>"
)


def parse_otsl(content: str) -> list[list[tuple[str, str]]]:
    """
    Parse OTSL cell tokens into a 2-D grid of (cell_type, text) tuples.

    Tokens:
        <nl>   = new row
        <fcel> = data cell          <ecel> = empty cell
        <ched> = column header      <rhed> = row header
        <lcel> = left-merged cell   <ucel> = up-merged cell
    """
    rows: list[list[tuple[str, str]]] = []
    current_row: list[tuple[str, str]] = []

    # Split on cell tokens, capturing the token itself
    parts = re.split(r"(<(?:ecel|ched|fcel|lcel|rhed|ucel|nl)>)", content)

    current_type: str | None = None
    text_accum = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == "<nl>":
            # Flush any pending cell
            if current_type is not None:
                current_row.append((current_type, text_accum.strip()))
                current_type = None
                text_accum = ""
            if current_row:
                rows.append(current_row)
                current_row = []
        elif _CELL_TOKEN_RE.fullmatch(part):
            # Flush previous cell
            if current_type is not None:
                current_row.append((current_type, text_accum.strip()))
                text_accum = ""
            current_type = part[1:-1]  # strip < >
        else:
            text_accum += part

    # Flush remaining
    if current_type is not None:
        current_row.append((current_type, text_accum.strip()))
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
        # Fallback: just draw the bounding rectangle
        draw.rectangle([px1, py1, px2, py2], outline=TABLE_LINE_COLOUR, width=1)
        return

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 1
    table_w = px2 - px1
    table_h = py2 - py1
    col_w = table_w / num_cols if num_cols else table_w
    row_h = table_h / num_rows if num_rows else table_h

    font_size = max(8, int(FONT_SIZES["table_cell"] * page_h / 1754))
    font = get_font(font_size)
    bold_font = get_bold_font(font_size)

    # --- Build a normalised grid (pad short rows with ecel) ---
    grid: list[list[tuple[str, str]]] = []
    for row in rows:
        padded = list(row) + [("ecel", "")] * (num_cols - len(row))
        grid.append(padded[:num_cols])

    # --- Compute span (colspan, rowspan) for each content cell ---
    # span_map[ri][ci] = (colspan, rowspan) for content cells, None for merged cells
    span_map: list[list[tuple[int, int] | None]] = [
        [None] * num_cols for _ in range(num_rows)
    ]

    for ri in range(num_rows):
        for ci in range(num_cols):
            cell_type, _ = grid[ri][ci]
            if cell_type in ("lcel", "ucel"):
                # This cell is part of a merge — mark as None
                span_map[ri][ci] = None
                continue

            # Count colspan (consecutive lcel to the right)
            colspan = 1
            while ci + colspan < num_cols and grid[ri][ci + colspan][0] == "lcel":
                colspan += 1

            # Count rowspan (consecutive ucel below, across full colspan width)
            rowspan = 1
            while ri + rowspan < num_rows:
                all_ucel = True
                for dc in range(colspan):
                    col_idx = ci + dc
                    if col_idx < num_cols and grid[ri + rowspan][col_idx][0] == "ucel":
                        continue
                    all_ucel = False
                    break
                if all_ucel:
                    rowspan += 1
                else:
                    break

            span_map[ri][ci] = (colspan, rowspan)

    # --- Draw cells ---
    for ri in range(num_rows):
        for ci in range(num_cols):
            span = span_map[ri][ci]
            if span is None:
                # Merged cell — skip (drawn as part of the origin cell)
                continue

            colspan, rowspan = span
            cell_type, cell_text = grid[ri][ci]

            cx1 = px1 + int(ci * col_w)
            cy1 = py1 + int(ri * row_h)
            cx2 = px1 + int((ci + colspan) * col_w)
            cy2 = py1 + int((ri + rowspan) * row_h)

            # Background for header cells
            if cell_type in ("ched", "rhed"):
                draw.rectangle([cx1, cy1, cx2, cy2], fill=TABLE_HEADER_BG)

            # Cell border
            draw.rectangle([cx1, cy1, cx2, cy2], outline=TABLE_LINE_COLOUR, width=1)

            # Cell text
            if cell_text:
                cell_content_w = cx2 - cx1 - 6
                cell_content_h = cy2 - cy1
                cfont = bold_font if cell_type in ("ched", "rhed") else font
                lines = _wrap_text(cell_text, cfont, max(cell_content_w, 10))
                ty = cy1 + 3
                for line in lines:
                    if ty + font_size + 2 > cy2:
                        # Draw truncation indicator
                        draw.text((cx1 + 3, ty - font_size), "…", fill=TEXT_COLOUR, font=cfont)
                        break
                    draw.text((cx1 + 3, ty), line, fill=TEXT_COLOUR, font=cfont)
                    ty += font_size + 2

    # Outer border (thicker)
    draw.rectangle([px1, py1, px2, py2], outline=TABLE_LINE_COLOUR, width=2)

    # --- Draw grid lines explicitly for visual clarity ---
    # Horizontal lines
    for ri in range(1, num_rows):
        y = py1 + int(ri * row_h)
        draw.line([(px1, y), (px2, y)], fill=TABLE_LINE_COLOUR, width=1)
    # Vertical lines
    for ci in range(1, num_cols):
        x = px1 + int(ci * col_w)
        draw.line([(x, py1), (x, py2)], fill=TABLE_LINE_COLOUR, width=1)


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


def load_records_from_file(path: str) -> list[dict]:
    """Load records from a file, preserving ``image`` paths alongside doctags.

    Supports the same formats as :func:`load_doctags_from_file`:

    * ``.jsonl`` – one JSON object per line; each object must have a
      ``"doctag"`` key and may have an ``"image"`` key.
    * ``.json``  – a single object, list of objects, or bare string.
    * ``.txt``   – raw DocTag text (no image information).

    Returns a list of dicts, each with at least ``{"doctag": str}`` and
    optionally ``{"image": str | list[str]}``.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()

    if p.suffix == ".jsonl":
        records: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "doctag" in obj:
                records.append({"doctag": obj["doctag"], "image": obj.get("image")})
            elif isinstance(obj, str):
                records.append({"doctag": obj})
        return records

    if p.suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, str):
            return [{"doctag": data}]
        if isinstance(data, dict) and "doctag" in data:
            return [{"doctag": data["doctag"], "image": data.get("image")}]
        if isinstance(data, list):
            result: list[dict] = []
            for item in data:
                if isinstance(item, str):
                    result.append({"doctag": item})
                elif isinstance(item, dict) and "doctag" in item:
                    result.append({"doctag": item["doctag"], "image": item.get("image")})
            return result

    # Fallback: raw doctag text
    return [{"doctag": raw}]


def _resolve_output_path(
    base_out: Path,
    img_field,
    page_rel_idx: int,
    num_pages: int,
    global_page_num: int,
    fmt: str,
) -> Path:
    """Return the output file path for a single rendered page.

    Priority order:

    1. Use the ``image`` field from the JSONL record (ensures annotation and
       rendered image names match exactly).
    2. Fall back to a numbered scheme inside *base_out*.

    Parameters
    ----------
    base_out:
        The ``--output`` path supplied on the CLI (file or directory).
    img_field:
        Value of the ``"image"`` key from the JSONL record; may be a string,
        a list of strings (multi-page combined entry), or ``None``.
    page_rel_idx:
        0-based index of this page within the current doctag's pages.
    num_pages:
        Total number of pages produced by the current doctag (after
        splitting on ``<page_break>``).
    global_page_num:
        1-based sequential page counter across all records.
    fmt:
        File format extension (``"png"`` or ``"pdf"``).
    """
    if img_field is not None:
        if isinstance(img_field, list):
            # Multi-page combined entry: each page has its own image name
            if page_rel_idx < len(img_field):
                return Path(img_field[page_rel_idx])
            # More pages than image entries: fall through to fallback
        elif isinstance(img_field, str):
            if num_pages == 1:
                # Single-page record with a known image name
                return Path(img_field)
            else:
                # Single image name but doctag was split into multiple pages:
                # append a page suffix so filenames are unique
                p = Path(img_field)
                sfx = p.suffix or f".{fmt}"
                return p.parent / f"{p.stem}_p{page_rel_idx + 1:02d}{sfx}"

    # No image field: use base_out as directory (or exact file for single pages)
    if base_out.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"):
        if num_pages == 1 and global_page_num == 1:
            return base_out
        return base_out.parent / f"page_{global_page_num}.{fmt}"
    return base_out / f"page_{global_page_num}.{fmt}"


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

    # Load records (doctag + optional image path from JSONL)
    if args.doctag:
        records = [{"doctag": args.doctag, "image": None}]
    else:
        records = load_records_from_file(args.input)

    if not records:
        print("No DocTag strings found in input.", file=sys.stderr)
        sys.exit(1)

    print(f"Rendering {len(records)} record(s) …")

    out = Path(args.output)
    written: list[str] = []

    if fmt == "pdf":
        # For PDF output: collect all pages then save as a single multi-page PDF
        all_images: list[Image.Image] = []
        for rec in records:
            for page_dt in split_doctag_on_pagebreaks(rec.get("doctag", "")):
                all_images.append(render_doctag(page_dt, args.width, args.height, args.margin))
        written = save_output(all_images, args.output, fmt)
    else:
        # For image output: render and save each page immediately with the
        # filename from the record's "image" field (when available) so that
        # rendered filenames always match the annotation file.
        global_page_num = 0
        for rec_idx, rec in enumerate(records):
            dt = rec.get("doctag", "")
            img_field = rec.get("image")  # str | list[str] | None
            pages = split_doctag_on_pagebreaks(dt)
            for page_rel_idx, page_dt in enumerate(pages):
                global_page_num += 1
                rendered = render_doctag(page_dt, args.width, args.height, args.margin)
                n_elems = len(parse_doctag(page_dt))
                out_path = _resolve_output_path(
                    out, img_field, page_rel_idx, len(pages),
                    global_page_num, fmt,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(str(out_path))
                written.append(str(out_path))
                print(f"  Record {rec_idx + 1} page {page_rel_idx + 1}: {n_elems} elements")

    for fp in written:
        print(f"  → {fp}")
    print("Done.")


if __name__ == "__main__":
    main()
