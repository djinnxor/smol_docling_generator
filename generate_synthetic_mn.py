"""
Generate Mongolian synthetic training data from the HuggingFaceM4/DoclingMatix
dataset.

Pipeline
--------
1.  **Stream** DoclingMatix (text labels only — images are NOT downloaded,
    avoiding the slow data-file resolution).
2.  **Extract** ``<doctag>`` strings from the ``texts`` conversation column.
3.  **Translate** the human-readable text fragments inside each doctag from
    English to Mongolian, preserving the structural tags and ``<loc_N>``
    location tokens.
4.  **Render** the translated doctags as document-page images via
    ``build_document.render_doctag()``.
5.  **Save** ``{"image": "<path>", "doctag": "<doctag string>"}`` pairs as a
    JSONL training file compatible with ``train.py``.

Output JSONL schema (one JSON object per line):
    {"image": "labeler_data/uploads/mn_page_000001.png", "doctag": "<doctag>...</doctag>"}

Usage
-----
    # Basic — stream 100 samples, render & save
    python generate_synthetic_mn.py --num_samples 100

    # Larger run with custom output directory
    python generate_synthetic_mn.py --num_samples 1000 \\
        --output_dir mn_synthetic_data --batch_size 20

    # Skip rendering (useful for quick doctag-only export)
    python generate_synthetic_mn.py --num_samples 500 --skip_render
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# build_document imports (local)
# ---------------------------------------------------------------------------
from build_document import render_doctag, parse_doctag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "HuggingFaceM4/DoclingMatix"

# Regex: match the text content *after* the four <loc_N> tokens and before
# the closing tag.  Works for every tag type that carries text content.
_TEXT_IN_TAG_RE = re.compile(
    r"(?P<open><(?:text|title|section_header_level_[123]|caption|list_item|"
    r"formula|page_header|page_footer|code|footnote)>"
    r"<loc_\d+><loc_\d+><loc_\d+><loc_\d+>)"
    r"(?P<content>[^<]+)"
    r"(?P<close></(?:text|title|section_header_level_[123]|caption|list_item|"
    r"formula|page_header|page_footer|code|footnote)>)"
)

# OTSL table cell tokens that can carry text:  <ched>Text , <rhed>Text , <fcel>Text
# The optional ``(?:\s[^>]*)?`` handles OTSL span attributes like cspan="2" rspan="3"
# that appear in prepare_dataset.py's table_xml_to_template_doctag output.
_CELL_TEXT_RE = re.compile(
    r"(?P<open><(?:ched|rhed|fcel)(?:\s[^>]*)?>)"
    r"(?P<content>[^<]+)"
)


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def _get_translator(source: str = "en", target: str = "mn"):
    """Return a ``GoogleTranslator`` instance (lazy import)."""
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source=source, target=target)


def translate_texts(
    texts: list[str],
    translator=None,
    batch_size: int = 30,
) -> list[str]:
    """Translate a list of strings from English to Mongolian.

    Uses ``deep-translator`` GoogleTranslator under the hood with batching
    for efficiency.  Empty strings and very short tokens are passed through
    unchanged.

    Parameters
    ----------
    texts : list[str]
        Strings to translate.
    translator : GoogleTranslator, optional
        Pre-created translator instance (avoids re-creating per call).
    batch_size : int
        Maximum number of strings to send per batch request.

    Returns
    -------
    list[str]
        Translated strings in the same order.
    """
    if translator is None:
        translator = _get_translator()

    results: list[str] = [""] * len(texts)

    # Separate indices that need translation vs pass-through
    to_translate: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        stripped = t.strip()
        # Skip empty or purely numeric strings (numbers don't need translation)
        if not stripped or stripped.isnumeric():
            results[i] = t
        else:
            to_translate.append((i, stripped))

    if not to_translate:
        return results

    # Batch translate
    for start in range(0, len(to_translate), batch_size):
        batch = to_translate[start : start + batch_size]
        batch_texts = [t for _, t in batch]
        try:
            translated = translator.translate_batch(batch_texts)
        except Exception as exc:
            print(f"  ⚠ Translation batch {start}-{start + len(batch)} failed ({type(exc).__name__}: {exc}); keeping originals")
            translated = batch_texts

        for (idx, _), tr in zip(batch, translated):
            results[idx] = tr if tr else texts[idx]

    return results


# ---------------------------------------------------------------------------
# DocTag text extraction & replacement
# ---------------------------------------------------------------------------

def extract_texts_from_doctag(doctag: str) -> list[str]:
    """Return all human-readable text fragments from a doctag string.

    This includes text within element tags (``<text>``, ``<title>``, etc.)
    and within OTSL table cell tokens (``<ched>``, ``<rhed>``, ``<fcel>``).
    """
    texts: list[str] = []

    for m in _TEXT_IN_TAG_RE.finditer(doctag):
        texts.append(m.group("content"))

    for m in _CELL_TEXT_RE.finditer(doctag):
        texts.append(m.group("content"))

    return texts


def translate_doctag(
    doctag: str,
    translator=None,
) -> str:
    """Translate all text fragments inside a doctag string from EN→MN.

    Structural tags and ``<loc_N>`` tokens are preserved verbatim.
    """
    # Collect all text spans (element body + table cell texts)
    element_texts: list[str] = []
    cell_texts: list[str] = []

    for m in _TEXT_IN_TAG_RE.finditer(doctag):
        element_texts.append(m.group("content"))
    for m in _CELL_TEXT_RE.finditer(doctag):
        cell_texts.append(m.group("content"))

    all_texts = element_texts + cell_texts
    if not all_texts:
        return doctag  # nothing to translate

    translated = translate_texts(all_texts, translator=translator)
    element_translated = translated[: len(element_texts)]
    cell_translated = translated[len(element_texts) :]

    # Replace element-level texts
    elem_iter = iter(element_translated)

    def _replace_element(m: re.Match) -> str:
        return m.group("open") + next(elem_iter) + m.group("close")

    result = _TEXT_IN_TAG_RE.sub(_replace_element, doctag)

    # Replace cell-level texts
    cell_iter = iter(cell_translated)

    def _replace_cell(m: re.Match) -> str:
        return m.group("open") + next(cell_iter)

    result = _CELL_TEXT_RE.sub(_replace_cell, result)

    return result


# ---------------------------------------------------------------------------
# Dataset loading (streaming, text-only)
# ---------------------------------------------------------------------------

def stream_doctags(
    dataset_id: str = DATASET_ID,
    split: str = "train",
    num_samples: int | None = None,
):
    """Yield raw doctag strings from the DoclingMatix dataset.

    Uses **streaming** mode so that images and large data files are never
    downloaded — only the text/label column is fetched.

    Yields
    ------
    str
        A single ``<doctag>…</doctag>`` string per sample.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)

    count = 0
    for row in ds:
        # The "texts" column is a list of conversation turns (dicts or strings).
        texts_col = row.get("texts")
        if texts_col is None:
            texts_col = row.get("text")
        if texts_col is None:
            texts_col = []

        # Extract doctag string from conversation turns
        doctag = _extract_doctag_from_texts(texts_col)
        if doctag:
            yield doctag
            count += 1
            if num_samples is not None and count >= num_samples:
                return


def _extract_doctag_from_texts(texts_col) -> str | None:
    """Pull the ``<doctag>…</doctag>`` string out of the texts column.

    The column may be:
    *   A plain string containing the doctag directly.
    *   A list of dicts ``[{"user": "…", "assistant": "…"}, …]`` where the
        assistant turn holds the doctag.
    *   A list of strings where one contains ``<doctag>``.
    """
    if isinstance(texts_col, str):
        if "<doctag>" in texts_col:
            return _trim_doctag(texts_col)
        return None

    if isinstance(texts_col, list):
        for item in texts_col:
            if isinstance(item, dict):
                # Check common keys: assistant, content, text, doctag
                for key in ("assistant", "content", "text", "doctag"):
                    val = item.get(key, "")
                    if val and "<doctag>" in str(val):
                        return _trim_doctag(str(val))
            elif isinstance(item, str) and "<doctag>" in item:
                return _trim_doctag(item)

    return None


def _trim_doctag(s: str) -> str | None:
    """Extract the ``<doctag>…</doctag>`` substring from a larger string."""
    start = s.find("<doctag>")
    end = s.find("</doctag>")
    if start == -1 or end == -1:
        return None
    return s[start : end + len("</doctag>")]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    num_samples: int = 100,
    output_dir: str = "mn_synthetic_data",
    images_subdir: str = "labeler_data/uploads",
    batch_size: int = 10,
    skip_render: bool = False,
    split: str = "train",
    page_w: int = 1240,
    page_h: int = 1754,
    margin: int = 20,
) -> str:
    """Run the full pipeline and return the path to the output JSONL file.

    Parameters
    ----------
    num_samples : int
        Number of DoclingMatix samples to process.
    output_dir : str
        Directory for the JSONL output file.
    images_subdir : str
        Directory where rendered PNG images are saved.  Image paths in the
        JSONL use this as a prefix so the output matches the expected
        ``{"image": "labeler_data/uploads/…", "doctag": "…"}`` schema.
    batch_size : int
        How many doctags to accumulate before flushing a translation batch.
    skip_render : bool
        If ``True``, skip image rendering (useful for doctag-only export).
    split : str
        Dataset split to stream from (default ``"train"``).
    page_w, page_h, margin : int
        Page dimensions passed to ``render_doctag()``.

    Returns
    -------
    str
        Path to the output JSONL file.
    """
    out = Path(output_dir)
    img_dir = Path(images_subdir)
    if not skip_render:
        img_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "mn_synthetic_train.jsonl"
    translator = _get_translator()

    records_written = 0
    t0 = time.time()

    print(f"▶ Streaming up to {num_samples} samples from {DATASET_ID} …")

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for idx, doctag in enumerate(stream_doctags(
            dataset_id=DATASET_ID, split=split, num_samples=num_samples,
        )):
            try:
                translated_doctag = translate_doctag(doctag, translator=translator)
            except Exception as exc:
                print(f"  ⚠ Sample {idx}: translation failed ({type(exc).__name__}: {exc}); skipping")
                continue

            # Validate that the translated doctag has elements
            elems = parse_doctag(translated_doctag)
            if not elems:
                print(f"  ⚠ Sample {idx}: no elements after translation; skipping")
                continue

            # Build image filename
            img_name = f"mn_page_{idx:06d}.png"
            img_rel_path = str(img_dir / img_name)

            if not skip_render:
                img = render_doctag(translated_doctag, page_w, page_h, margin)
                img.save(str(img_dir / img_name))

            # Write record in the required format:
            # {"image": "labeler_data/uploads/mn_page_000001.png", "doctag": "<doctag>...</doctag>"}
            record = {"image": img_rel_path, "doctag": translated_doctag}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed else 0
                print(
                    f"  Processed {idx + 1}/{num_samples} "
                    f"({records_written} written) — {rate:.1f} samples/s"
                )

    elapsed = time.time() - t0
    print(
        f"\n✓ Done. {records_written} records → {jsonl_path}  "
        f"({elapsed:.1f}s total)"
    )
    return str(jsonl_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Mongolian synthetic training data from the "
            "HuggingFaceM4/DoclingMatix dataset. Streams text labels only "
            "(no images), translates EN→MN, renders documents, and saves JSONL."
        ),
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=100,
        help="Number of DoclingMatix samples to process (default: 100)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="mn_synthetic_data",
        help="Output directory for the JSONL file (default: mn_synthetic_data)",
    )
    parser.add_argument(
        "--images_dir",
        default="labeler_data/uploads",
        help=(
            "Directory where rendered PNG images are saved. Image paths in "
            "JSONL use this prefix (default: labeler_data/uploads)"
        ),
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=10,
        help="Translation batch size (default: 10)",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Skip image rendering (export doctags only)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (default: train)",
    )
    parser.add_argument(
        "--page_width",
        type=int,
        default=1240,
        help="Rendered page width in px (default: 1240)",
    )
    parser.add_argument(
        "--page_height",
        type=int,
        default=1754,
        help="Rendered page height in px (default: 1754)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Page margin in px (default: 20)",
    )
    args = parser.parse_args()

    generate_synthetic_data(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        images_subdir=args.images_dir,
        batch_size=args.batch_size,
        skip_render=args.skip_render,
        split=args.split,
        page_w=args.page_width,
        page_h=args.page_height,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
