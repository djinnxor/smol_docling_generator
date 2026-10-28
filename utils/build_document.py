"""
Build a visual document (PNG or PDF) from a DocTag string.

Each DocTag element is rendered at its exact location on the page,
reconstructing the spatial layout of the original document.

DocTag location tokens use a 0-500 normalised coordinate system.
This script maps them back to absolute pixel positions and draws
text, headers, tables, and placeholders for images.

<formula> elements are typeset as equations rather than printed as source:
LaTeX goes through matplotlib's mathtext, and a formula written with a run
of dashes as its fraction rule is drawn as a real stacked fraction.  Pass
--no-math to see the raw generated source instead.

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


# Permissive element regex: ANY tag name immediately followed by 4 <loc_*>
# tokens.  This matches the standard document tags (text, otsl, …) as well as
# the custom ID-card field tags emitted by the ID model (register_number,
# gender, familyname, surname, name, birthday, …).  The <doctag> wrapper is not
# matched because it is not followed by location tokens.
_ANY_ELEMENT_RE = re.compile(
    r"<(?P<tag>[A-Za-z_][\w-]*)>"
    r"<loc_(?P<x1>\d+)><loc_(?P<y1>\d+)><loc_(?P<x2>\d+)><loc_(?P<y2>\d+)>"
    r"(?P<content>.*?)"
    r"</(?P=tag)>",
    re.DOTALL,
)


def parse_doctag_elements(doctag: str) -> list[DocElement]:
    """Parse any positioned ``<tag><loc>…<loc>…</tag>`` element.

    Unlike :func:`parse_doctag`, this is not limited to the standard document
    element types, so it also captures custom ID-card field tags.
    """
    elements: list[DocElement] = []
    for m in _ANY_ELEMENT_RE.finditer(doctag):
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
            # Remove all <loc_*> tokens from text_accum (cell location info)
            cell_text = re.sub(r"<loc_\d+>", "", text_accum).strip()
            current_row.append({
                "type": current_type,
                "text": cell_text,
                "colspan": current_colspan,
                "rowspan": current_rowspan,
                # No cell location
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


# ---------------------------------------------------------------------------
# Formula rendering
# ---------------------------------------------------------------------------

# A run of 3+ dashes is a fraction rule: the numerator sits above it and the
# denominator below.  Scanned Mongolian accounting documents write fractions
# that way and the OCR labels carry the dashes through literally, so drawing
# them as a flat line of text turns a fraction into nonsense.
_DASH_RULE_RE = re.compile(r"[-\u2013\u2014_]{3,}")

# The labels carry LaTeX with doubled backslashes (\\frac, \\sum) because the
# doctag string was JSON-escaped twice, and the model emits them the same way.
# Collapse only a pair that introduces a command -- a bare \\ is a row break.
_DOUBLE_ESC_RE = re.compile(r"\\\\(?=[A-Za-z,;:!])")

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

# Markers that make a string worth handing to the math typesetter.
_MATH_HINT_RE = re.compile(r"\\[A-Za-z]+|[\^_=]")

MATH_RENDERING = True    # False (or --no-math) renders formulas as plain text

_MATH_DPI = 200          # typeset large, then downscale into the element box
_MATH_FONTSIZE = 28
# Pixels per em in the typeset bitmap -- what a point of font size is worth
# once mathtext has rendered at _MATH_DPI.  Used to match the page's text size.
_MATH_EM_PX = _MATH_FONTSIZE / 72 * _MATH_DPI
_MATH_CACHE: dict[tuple[str, tuple[int, int, int]], Image.Image | None] = {}


def normalise_latex(text: str) -> str:
    """Undo the label double-escaping and drop any $ delimiters."""
    return _DOUBLE_ESC_RE.sub(r"\\", text).replace("$", "").strip()


def looks_like_math(text: str) -> bool:
    """True when *text* is LaTeX-ish enough to be worth typesetting."""
    if not text or _CYRILLIC_RE.search(text):
        return False         # mathtext has no Cyrillic glyphs -- would be tofu
    return bool(_MATH_HINT_RE.search(text))


def render_math(
    tex: str, colour: tuple[int, int, int] = TEXT_COLOUR
) -> Image.Image | None:
    """Typeset *tex* as a transparent RGBA bitmap, or None if it will not parse.

    matplotlib's mathtext is used rather than a real TeX install: it covers the
    \\frac, \\sum, \\sqrt, sub/superscript and Greek-letter vocabulary these
    documents use, and matplotlib is already a dependency of the chart code.
    """
    key = (tex, colour)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]

    image: Image.Image | None = None
    try:
        import io

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(0.01, 0.01), dpi=_MATH_DPI)
        FigureCanvasAgg(fig)
        fig.text(0, 0, f"${tex}$", fontsize=_MATH_FONTSIZE,
                 color=tuple(c / 255 for c in colour))
        buf = io.BytesIO()
        # bbox_inches="tight" shrinks the canvas to the equation's own ink, so
        # the caller gets a bitmap it can scale straight into the element box.
        fig.savefig(buf, format="png", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        buf.seek(0)
        image = Image.open(buf).convert("RGBA")
        image.load()
    except Exception:
        # A formula that will not parse -- a truncated generation, an
        # unbalanced brace -- must not take the whole page down.  Returning
        # None tells the caller to fall back to drawing the source text.
        image = None

    _MATH_CACHE[key] = image
    return image


def _draw_centred_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font,
    centre_x: int,
    y: int,
    line_h: int,
    colour=TEXT_COLOUR,
    limit_y: int | None = None,
) -> int:
    """Draw *lines* centred on *centre_x*, returning the y below the last one."""
    for line in lines:
        if limit_y is not None and y + line_h > limit_y:
            break
        bbox = font.getbbox(line)
        draw.text((centre_x - (bbox[2] - bbox[0]) // 2, y), line,
                  fill=colour, font=font)
        y += line_h
    return y


def _draw_stacked_fraction(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font,
    line_h: int,
) -> None:
    """Draw a dash-rule formula as a real fraction: num / rule / denom.

    The numerator may be absent -- the labels sometimes open an element with
    the rule because the numerator was recognised as the line above -- in
    which case the rule is drawn at the top and everything else below it.
    """
    px1, py1, px2, py2 = box
    m = _DASH_RULE_RE.search(text)
    num = text[:m.start()].strip()
    den = text[m.end():].strip()

    box_w = max(px2 - px1, 20)
    centre_x = px1 + box_w // 2
    num_lines = _wrap_text(num, font, box_w) if num else []
    den_lines = _wrap_text(den, font, box_w) if den else []

    # Put the rule where the two halves meet, biased to the measured box so a
    # fraction keeps the vertical position the model predicted for it.
    needed = (len(num_lines) + len(den_lines)) * line_h + 6
    top = py1 if needed <= (py2 - py1) else py1
    rule_y = top + len(num_lines) * line_h + 2

    _draw_centred_lines(draw, num_lines, font, centre_x, top, line_h)
    draw.line([px1, rule_y, px2, rule_y], fill=TEXT_COLOUR, width=1)
    _draw_centred_lines(draw, den_lines, font, centre_x, rule_y + 3, line_h)


def _draw_formula(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    elem: DocElement,
    page_w: int,
    page_h: int,
    margin: int,
) -> None:
    """Render a <formula> element as an equation rather than as source text.

    Three cases, in order: a dash-rule fraction is stacked, a LaTeX formula is
    typeset and pasted in, and anything else falls back to the plain text path.
    """
    px1 = _loc_to_px(elem.x1, page_w, margin)
    py1 = _loc_to_px(elem.y1, page_h, margin)
    px2 = _loc_to_px(elem.x2, page_w, margin)
    py2 = _loc_to_px(elem.y2, page_h, margin)
    box_w, box_h = max(px2 - px1, 20), max(py2 - py1, 12)

    font_size = max(8, int(FONT_SIZES["formula"] * (page_h / 1754)))
    font = get_font(font_size)

    text = re.sub(r"<loc_\d+>", "", elem.content).strip()
    if not text:
        return

    if _DASH_RULE_RE.search(text):
        _draw_stacked_fraction(draw, text, (px1, py1, px2, py2),
                               font, font_size + 3)
        return

    tex = normalise_latex(text)
    if MATH_RENDERING and looks_like_math(tex):
        math_img = render_math(tex)
        if math_img is not None:
            # Scale so the equation's type size matches the page's body text
            # rather than so the bitmap fits the box.  Predicted formula boxes
            # hug the baseline, and a stacked fraction is two lines plus a rule
            # tall -- squeezing that into a one-line box renders it at half the
            # surrounding text size and unreadable.  The width is still capped
            # to the box, because horizontal overflow would run into the
            # neighbouring column; the extra height is centred on the box.
            scale = min(font_size / _MATH_EM_PX, box_w / math_img.width)
            w = max(1, int(math_img.width * scale))
            h = max(1, int(math_img.height * scale))
            resized = math_img.resize((w, h), Image.LANCZOS)
            img.paste(resized,
                      (px1 + (box_w - w) // 2, py1 + (box_h - h) // 2),
                      resized)
            return

    _draw_text_element(draw, elem, page_w, page_h, margin)


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


    # Compute grid cell sizes
    if num_rows == 0 or num_cols == 0:
        return
    cell_width = (px2 - px1) // num_cols
    cell_height = (py2 - py1) // num_rows

    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            cell_type = cell["type"]
            cell_text = cell["text"]
            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)
            # Skip merged cells (lcel, ucel)
            if cell_type in ("lcel", "ucel"):
                continue
            # Compute cell bounding box
            cx1 = px1 + ci * cell_width
            cy1 = py1 + ri * cell_height
            cx2 = cx1 + cell_width * colspan
            cy2 = cy1 + cell_height * rowspan

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
        elif elem.tag == "formula":
            _draw_formula(img, draw, elem, page_w, page_h, margin)
        else:
            _draw_text_element(draw, elem, page_w, page_h, margin)

    return img


# ---------------------------------------------------------------------------
# Overlay rendering (draw boxes + text on top of a real image)
# ---------------------------------------------------------------------------

# Overlay colours
ID_BOX_COLOUR = (0, 180, 0)        # YOLO ID detection box (green)
ELEM_BOX_COLOUR = (220, 30, 30)    # DocTag element box (red)
OVERLAY_TEXT_COLOUR = (10, 40, 200)  # rendered content text (blue)
LABEL_BG_COLOUR = (255, 255, 255)  # background strip behind text for legibility


def _clean_content(tag: str, content: str) -> str:
    """Return human-readable text for an element, stripping markup tokens."""
    if tag == "otsl":
        # Flatten table cells into a single space-joined string.
        rows = parse_otsl(content)
        cells = [c["text"] for r in rows for c in r if c.get("text")]
        return " | ".join(cells)
    # Drop any stray location tokens, collapse whitespace.
    text = re.sub(r"<loc_\d+>", "", content)
    text = re.sub(r"<[^>]+>", "", text)
    if tag == "formula":
        # The overlay labels a box with its text, so the equation cannot be
        # typeset here -- but it can at least lose the doubled backslashes.
        text = normalise_latex(text)
    return " ".join(text.split())


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    max_width: int,
    text_colour=OVERLAY_TEXT_COLOUR,
    anchor: str = "lt",
) -> None:
    """Draw wrapped *text* at *xy* on top of a white strip for readability.

    *anchor* controls how *xy* is interpreted:
        "lt" – *xy* is the top-left corner, text grows to the right (default)
        "rt" – *xy* is the top-right corner, each line is right-aligned to it
    """
    if not text:
        return
    lines = _wrap_text(text, font, max(max_width, 40))
    x, y = xy
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 2
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        if anchor == "rt":
            x0 = x - w - 4
            draw.rectangle([x0, y, x, y + line_h], fill=LABEL_BG_COLOUR)
            draw.text((x0 + 2, y), line, fill=text_colour, font=font)
        else:
            draw.rectangle([x, y, x + w + 4, y + line_h], fill=LABEL_BG_COLOUR)
            draw.text((x + 2, y), line, fill=text_colour, font=font)
        y += line_h


def draw_doctag_overlay(
    base_image: Image.Image,
    doctag: str,
    region: tuple[int, int, int, int] | None = None,
    id_box: tuple[int, int, int, int] | None = None,
    show_text: bool = True,
) -> Image.Image:
    """Draw DocTag element boxes (+ their text) and the ID box on *base_image*.

    DocTag ``<loc_*>`` tokens use a 0-500 coordinate range relative to the
    image the model actually saw — here that is the cropped ID region. They are
    mapped into *region* ``(x1, y1, x2, y2)`` (the crop's location inside
    *base_image*); if *region* is omitted the whole image is used.

    *id_box* is the YOLO detection rectangle to outline in green.
    """
    img = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    iw, ih = img.size
    rx1, ry1, rx2, ry2 = region if region is not None else (0, 0, iw, ih)
    rw = max(rx2 - rx1, 1)
    rh = max(ry2 - ry1, 1)

    # Fonts scaled to the region size so labels stay readable. The content
    # (Mongolian text) is drawn noticeably larger than the small tag-name label.
    text_size = max(22, int(30 * rh / 1754))
    label_size = max(13, int(16 * rh / 1754))
    font = get_font(text_size)
    label_font = get_font(label_size)

    # ID detection box (green).
    if id_box is not None:
        draw.rectangle(list(id_box), outline=ID_BOX_COLOUR, width=3)
        _draw_label(draw, (id_box[0] + 3, max(0, id_box[1] - label_size - 4)),
                    "ID", label_font, rw, text_colour=ID_BOX_COLOUR)

    # DocTag element boxes (red) + Mongolian text (blue). Use the permissive
    # parser so custom ID-card field tags are captured, not just the standard
    # document element types.
    for elem in parse_doctag_elements(doctag):
        ex1 = rx1 + int(elem.x1 / LOC_SCALE * rw)
        ey1 = ry1 + int(elem.y1 / LOC_SCALE * rh)
        ex2 = rx1 + int(elem.x2 / LOC_SCALE * rw)
        ey2 = ry1 + int(elem.y2 / LOC_SCALE * rh)
        draw.rectangle([ex1, ey1, ex2, ey2], outline=ELEM_BOX_COLOUR, width=2)

        # Tag name just outside the top-left corner.
        _draw_label(draw, (ex1, max(0, ey1 - label_size - 3)),
                    elem.tag, label_font, ex2 - ex1, text_colour=ELEM_BOX_COLOUR)

        # Mongolian text at the upper-right corner of the box, right-aligned to
        # the box's right edge so it reads outward from the corner.
        if show_text:
            content = _clean_content(elem.tag, elem.content)
            _draw_label(draw, (ex2, ey1 + 2), content, font, ex2 - ex1,
                        anchor="rt")

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
    parser.add_argument(
        "--no-math", action="store_true",
        help="Draw <formula> elements as plain source text instead of "
             "typesetting them (matplotlib is needed for typesetting)",
    )
    args = parser.parse_args()

    if args.no_math:
        globals()["MATH_RENDERING"] = False

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
