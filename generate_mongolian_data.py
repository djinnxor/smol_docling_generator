"""
Generate Mongolian synthetic training data from HuggingFaceM4/DoclingMatix.

Efficient pipeline that avoids downloading images:
  1. Reads ONLY text columns from DoclingMatix parquet files (skips images)
  2. Extracts DocTag strings from the dataset text labels
  3. Translates English text content to Mongolian while preserving
     DocTag structure, coordinates, and layout tokens
  4. Outputs JSONL files compatible with build_document.py and train.py
  5. Optionally renders translated DocTags to PNG/PDF documents

Key efficiency features:
  - Column-projection on parquet files: downloads only the 'texts' column,
    completely skipping heavy image data
  - Configurable shard selection: process only the shards you need
  - Instance range selection: stream only a specific range of rows
  - Batch translation to minimise API round-trips
  - Supports offline translation via argostranslate

Multi-page document support:
  - Dataset rows with multiple pages are handled as a group
  - Pages are rendered as separate images (mn_page_00001_01.png, _02.png …)
  - Per-page annotations AND a combined annotation (using <page_break>
    to distinguish pages) are both written to the output JSONL

Usage:
    # Generate 100 translated samples (default)
    python generate_mongolian_data.py --output mongolian_data.jsonl

    # Process instances 50–150 only
    python generate_mongolian_data.py --output mongolian_data.jsonl \\
        --start-instance 50 --end-instance 150

    # Process more samples from more parquet shards
    python generate_mongolian_data.py --output mongolian_data.jsonl \\
        --limit 1000 --max-shards 5

    # Use offline argostranslate backend
    python generate_mongolian_data.py --translator argos \\
        --output mongolian_data.jsonl

    # Also render translated DocTags to PNG images for training
    python generate_mongolian_data.py --output mongolian_data.jsonl \\
        --render --render-dir mongolian_renders/

    # Append to an existing JSONL instead of overwriting
    python generate_mongolian_data.py --output mongolian_data.jsonl --append

    # Train/val split
    python generate_mongolian_data.py --output mongolian_data.jsonl \\
        --split-ratio 0.9
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
# DocTag text extraction / reconstruction
# ---------------------------------------------------------------------------

# Matches a DocTag element: <tag><loc_N><loc_N><loc_N><loc_N>content</tag>
_ELEMENT_RE = re.compile(
    r"<(?P<tag>text|title|section_header_level_[123]|caption|picture|otsl|"
    r"list_item|formula|page_header|page_footer|code|footnote)>"
    r"<loc_(?P<x1>\d+)><loc_(?P<y1>\d+)><loc_(?P<x2>\d+)><loc_(?P<y2>\d+)>"
    r"(?P<content>.*?)"
    r"</(?P=tag)>",
    re.DOTALL,
)

# Matches OTSL cell tokens
_CELL_TOKEN_RE = re.compile(
    r"(<(?:ecel|ched|fcel|lcel|rhed|ucel|nl)>)"
)

# Tags whose content is plain translatable text
_TEXT_TAGS = frozenset({
    "text", "title", "section_header_level_1", "section_header_level_2",
    "section_header_level_3", "caption", "list_item", "formula",
    "page_header", "page_footer", "code", "footnote",
})


def extract_texts_from_doctag(doctag: str) -> list[str]:
    """Return all translatable text strings found in a DocTag string.

    This collects text from both regular elements and OTSL table cells.
    """
    texts: list[str] = []
    for m in _ELEMENT_RE.finditer(doctag):
        tag = m.group("tag")
        content = m.group("content")
        if tag == "picture":
            continue
        if tag == "otsl":
            for cell_text in _extract_otsl_texts(content):
                if cell_text.strip():
                    texts.append(cell_text.strip())
        else:
            if content.strip():
                texts.append(content.strip())
    return texts


def _extract_otsl_texts(content: str) -> list[str]:
    """Extract translatable text strings from OTSL cell content."""
    parts = _CELL_TOKEN_RE.split(content)
    return [p.strip() for p in parts if p.strip() and not _CELL_TOKEN_RE.fullmatch(p)]


def translate_doctag(doctag: str, translate_fn) -> str:
    """Translate text content in a DocTag string, preserving structure.

    Parameters
    ----------
    doctag:
        Full DocTag string (``<doctag>…</doctag>``).
    translate_fn:
        Callable that accepts a single English string and returns the
        Mongolian translation.
    """

    def _replace_element(match: re.Match) -> str:
        tag = match.group("tag")
        x1, y1 = match.group("x1"), match.group("y1")
        x2, y2 = match.group("x2"), match.group("y2")
        content = match.group("content")
        loc = f"<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"

        if tag == "picture":
            return f"<{tag}>{loc}</{tag}>"

        if tag == "otsl":
            translated_content = _translate_otsl(content, translate_fn)
            return f"<{tag}>{loc}{translated_content}</{tag}>"

        # Regular text element
        stripped = content.strip()
        if stripped:
            translated = translate_fn(stripped)
            return f"<{tag}>{loc}{translated}</{tag}>"

        return match.group(0)

    return _ELEMENT_RE.sub(_replace_element, doctag)


def _translate_otsl(content: str, translate_fn) -> str:
    """Translate text within OTSL table cells."""
    parts = _CELL_TOKEN_RE.split(content)
    result: list[str] = []
    for part in parts:
        if _CELL_TOKEN_RE.fullmatch(part):
            result.append(part)  # Keep cell/row tokens as-is
        elif part.strip():
            result.append(translate_fn(part.strip()))
        else:
            result.append(part)
    return "".join(result)


# ---------------------------------------------------------------------------
# Translation backends
# ---------------------------------------------------------------------------

def _make_google_translator(
    source: str = "en", target: str = "mn",
):
    """Return a translate function using deep-translator's GoogleTranslator."""
    try:
        from deep_translator import GoogleTranslator  # type: ignore[import-untyped]
    except ImportError:
        print(
            "deep-translator is required for the 'google' backend.\n"
            "Install it with:  pip install deep-translator",
            file=sys.stderr,
        )
        sys.exit(1)

    translator = GoogleTranslator(source=source, target=target)

    def _translate(text: str) -> str:
        if not text:
            return text
        try:
            return translator.translate(text) or text
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Translation failed for '{text[:60]}…': {exc}",
                  file=sys.stderr)
            return text

    return _translate


def _make_argos_translator(
    source: str = "en", target: str = "mn",
):
    """Return a translate function using argostranslate (offline)."""
    try:
        import argostranslate.package  # type: ignore[import-untyped]
        import argostranslate.translate  # type: ignore[import-untyped]
    except ImportError:
        print(
            "argostranslate is required for the 'argos' backend.\n"
            "Install it with:  pip install argostranslate",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure the language package is installed
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available
         if p.from_code == source and p.to_code == target),
        None,
    )
    if pkg is None:
        # Try to see if routing through an intermediate language works
        print(
            f"No direct argostranslate package for {source}→{target}.\n"
            "Checking if translation is possible via intermediate languages …",
            file=sys.stderr,
        )
        # argostranslate can auto-chain; just make sure at least some
        # packages are installed.
        installed = argostranslate.translate.get_installed_languages()
        src_lang = next((l for l in installed if l.code == source), None)
        if src_lang is None:
            print(
                f"Source language '{source}' not available in argostranslate.\n"
                "Consider using --translator google instead.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)

    def _translate(text: str) -> str:
        if not text:
            return text
        try:
            return argostranslate.translate.translate(text, source, target)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Translation failed: {exc}", file=sys.stderr)
            return text

    return _translate


TRANSLATOR_FACTORIES = {
    "google": _make_google_translator,
    "argos": _make_argos_translator,
}


# ---------------------------------------------------------------------------
# Batch translation helper
# ---------------------------------------------------------------------------

def batch_translate_doctags(
    doctags: list[str],
    translate_fn,
    *,
    progress: bool = True,
) -> list[str]:
    """Translate a list of DocTag strings, showing progress."""
    translated: list[str] = []
    total = len(doctags)
    for i, dt in enumerate(doctags):
        translated.append(translate_doctag(dt, translate_fn))
        if progress and (i + 1) % 10 == 0:
            print(f"  Translated {i + 1}/{total} doctags …")
    return translated


# ---------------------------------------------------------------------------
# Dataset loading  (text-only, no images)
# ---------------------------------------------------------------------------

def _extract_doctags_from_texts(texts) -> list[str]:
    """Extract DocTag strings from the 'texts' column of DoclingMatix.

    The ``texts`` field can be:
      - A list of dicts with ``user`` / ``assistant`` keys (conversation)
      - A single string
      - A list of strings

    We look for any string that contains ``<doctag>`` and extract the
    full DocTag block from it.
    """
    doctags: list[str] = []

    if isinstance(texts, str):
        doctags.extend(_find_doctag_blocks(texts))
        return doctags

    if isinstance(texts, list):
        for item in texts:
            if isinstance(item, str):
                doctags.extend(_find_doctag_blocks(item))
            elif isinstance(item, dict):
                # Conversation format: check assistant response
                for key in ("assistant", "answer", "response", "output"):
                    val = item.get(key, "")
                    if val:
                        doctags.extend(_find_doctag_blocks(str(val)))
                # Also check user field in case the full doctag is there
                user_val = item.get("user", "")
                if user_val:
                    doctags.extend(_find_doctag_blocks(str(user_val)))

    return doctags


_DOCTAG_BLOCK_RE = re.compile(r"<doctag>.*?</doctag>", re.DOTALL)
_DOCTAG_INNER_RE = re.compile(r"<doctag>(.*?)</doctag>", re.DOTALL)


def _find_doctag_blocks(text: str) -> list[str]:
    """Find all <doctag>…</doctag> blocks in a string."""
    return _DOCTAG_BLOCK_RE.findall(text)


def combine_doctag_pages(page_doctags: list[str]) -> str:
    """Combine multiple page DocTag strings into one using ``<page_break>``.

    The inner content of each ``<doctag>…</doctag>`` block is extracted and
    joined with ``<page_break>`` so that the resulting string is a single
    multi-page doctag.  A single-element list is returned unchanged.
    """
    if len(page_doctags) == 1:
        return page_doctags[0]
    parts: list[str] = []
    for dt in page_doctags:
        m = _DOCTAG_INNER_RE.search(dt)
        parts.append(m.group(1) if m else dt)
    return "<doctag>" + "\n<page_break>\n".join(parts) + "</doctag>"


def load_doctags_from_doclingmatix(
    *,
    max_shards: int = 1,
    limit: int = 100,
    hf_token: str | None = None,
    start_instance: int = 0,
    end_instance: int | None = None,
) -> list[list[str]]:
    """Load DocTag strings from HuggingFaceM4/DoclingMatix.

    Uses direct parquet column-projection to read ONLY the ``texts``
    column, completely bypassing image data for maximum speed.

    Falls back to ``datasets`` streaming if direct parquet reading fails.

    Returns a list of *groups*.  Each group is the list of page DocTag
    strings extracted from a single dataset row, so a multi-page document
    becomes a group with more than one element.

    Parameters
    ----------
    start_instance:
        First dataset row index to process (0-based, inclusive).
    end_instance:
        Last dataset row index to process (exclusive).  ``None`` means
        process until *limit* groups are collected.
    """
    range_info = f"instances [{start_instance}, {end_instance or 'end'})"
    print(f"Loading DoclingMatix (max_shards={max_shards}, limit={limit}, "
          f"{range_info}) …")

    groups = _load_via_parquet(
        max_shards=max_shards, limit=limit, hf_token=hf_token,
        start_instance=start_instance, end_instance=end_instance,
    )
    if groups is not None:
        return groups

    print("Direct parquet loading failed; falling back to datasets streaming …")
    return _load_via_streaming(
        limit=limit, hf_token=hf_token,
        start_instance=start_instance, end_instance=end_instance,
    )


def _load_via_parquet(
    *,
    max_shards: int,
    limit: int,
    hf_token: str | None,
    start_instance: int = 0,
    end_instance: int | None = None,
) -> list[list[str]] | None:
    """Read DocTag strings directly from parquet files on the Hub.

    Returns None if the required libraries are unavailable or the
    parquet files cannot be listed.

    Returns a list of groups (one per dataset row that contained at least
    one DocTag block).  Rows are skipped until *start_instance* is reached
    and processing stops when *end_instance* is reached or *limit* groups
    have been collected.
    """
    try:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
    except ImportError:
        return None

    try:
        fs = HfFileSystem(token=hf_token)
        all_files = fs.ls(
            "datasets/HuggingFaceM4/DoclingMatix/data", detail=False,
        )
        parquet_files = sorted(
            f for f in all_files if f.endswith(".parquet")
        )
    except Exception as exc:
        print(f"  Could not list parquet files: {exc}", file=sys.stderr)
        return None

    if not parquet_files:
        return None

    shards = parquet_files[:max_shards]
    print(f"  Found {len(parquet_files)} parquet shards; "
          f"processing {len(shards)} …")

    doctag_groups: list[list[str]] = []
    row_counter = 0
    for shard_path in shards:
        if len(doctag_groups) >= limit:
            break
        print(f"  Reading {shard_path.split('/')[-1]} …")
        try:
            with fs.open(shard_path) as fh:
                table = pq.read_table(fh, columns=["texts"])
            for i in range(len(table)):
                if end_instance is not None and row_counter >= end_instance:
                    break
                if row_counter < start_instance:
                    row_counter += 1
                    continue
                if len(doctag_groups) >= limit:
                    break
                texts = table.column("texts")[i].as_py()
                found = _extract_doctags_from_texts(texts)
                if found:
                    doctag_groups.append(found)
                row_counter += 1
        except Exception as exc:
            print(f"  [warn] Failed to read shard: {exc}", file=sys.stderr)
            continue

    total = sum(len(g) for g in doctag_groups)
    print(f"  Extracted {total} DocTag strings in {len(doctag_groups)} groups")
    return doctag_groups[:limit]


def _load_via_streaming(
    *,
    limit: int,
    hf_token: str | None,
    start_instance: int = 0,
    end_instance: int | None = None,
) -> list[list[str]]:
    """Fallback: use ``datasets`` streaming mode.

    Rows before *start_instance* are skipped; processing stops when
    *end_instance* is reached or *limit* groups have been collected.
    Returns a list of groups (same structure as :func:`_load_via_parquet`).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print(
            "datasets library is required for streaming fallback.\n"
            "Install with:  pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    ds = load_dataset(
        "HuggingFaceM4/DoclingMatix",
        split="train",
        streaming=True,
        **kwargs,
    )

    doctag_groups: list[list[str]] = []
    for row_idx, row in enumerate(ds):
        if end_instance is not None and row_idx >= end_instance:
            break
        if row_idx < start_instance:
            continue
        if len(doctag_groups) >= limit:
            break
        texts = row.get("texts", [])
        found = _extract_doctags_from_texts(texts)
        if found:
            doctag_groups.append(found)

    total = sum(len(g) for g in doctag_groups)
    print(f"  Extracted {total} DocTag strings in "
          f"{len(doctag_groups)} groups (streaming)")
    return doctag_groups[:limit]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_jsonl(
    doctags: list[str],
    output_path: str,
    *,
    source_doctags: list[str] | None = None,
    image_paths: list[str | list[str]] | None = None,
    append: bool = False,
) -> str:
    """Write translated DocTag strings to a JSONL file.

    Each line is a JSON object.  The ``image`` key is written *before* the
    ``doctag`` key so that image paths are easy to locate at the start of
    each record.

    When *append* is ``True`` the file is opened in append mode, preserving
    any existing content.  The default (``False``) overwrites the file.

    Parameters
    ----------
    image_paths:
        Optional list of image paths.  Each element may be either a
        single path string (single-page) or a list of path strings
        (multi-page combined annotation).
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(p, mode, encoding="utf-8") as fh:
        for i, dt in enumerate(doctags):
            record: dict = {}
            if image_paths is not None and i < len(image_paths):
                record["image"] = image_paths[i]
            record["doctag"] = dt
            if source_doctags is not None and i < len(source_doctags):
                record["source_doctag"] = source_doctags[i]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(p)


def split_and_save(
    doctags: list[str],
    output_base: str,
    split_ratio: float = 0.9,
    *,
    source_doctags: list[str] | None = None,
    image_paths: list[str | list[str]] | None = None,
    append: bool = False,
) -> tuple[str, str | None]:
    """Split translated data into train/val and save as JSONL files."""
    n = len(doctags)
    split_idx = max(1, int(n * split_ratio))

    base = output_base.replace(".jsonl", "")
    train_path = f"{base}_train.jsonl"
    val_path = f"{base}_val.jsonl"

    train_sources = source_doctags[:split_idx] if source_doctags else None
    val_sources = source_doctags[split_idx:] if source_doctags else None
    train_images = image_paths[:split_idx] if image_paths else None
    val_images = image_paths[split_idx:] if image_paths else None

    save_jsonl(
        doctags[:split_idx], train_path,
        source_doctags=train_sources, image_paths=train_images, append=append,
    )
    print(f"Saved {split_idx} training records to {train_path}")

    val_written = None
    if split_idx < n:
        save_jsonl(
            doctags[split_idx:], val_path,
            source_doctags=val_sources, image_paths=val_images, append=append,
        )
        val_written = val_path
        print(f"Saved {n - split_idx} validation records to {val_path}")

    return train_path, val_written


def render_doctags(
    doctags: list[str],
    render_dir: str,
    *,
    page_w: int = 1240,
    page_h: int = 1754,
    fmt: str = "png",
) -> list[str]:
    """Render translated DocTag strings to image files using build_document.

    Returns the list of written file paths.
    """
    from build_document import render_doctag as _render  # type: ignore[import-untyped]

    out_dir = Path(render_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for i, dt in enumerate(doctags):
        img = _render(dt, page_w, page_h)
        fp = out_dir / f"mn_page_{i + 1:05d}.{fmt}"
        img.save(str(fp))
        paths.append(str(fp))
        if (i + 1) % 50 == 0:
            print(f"  Rendered {i + 1}/{len(doctags)} pages …")

    print(f"Rendered {len(paths)} pages to {render_dir}")
    return paths


def render_doctag_groups(
    doctag_groups: list[list[str]],
    render_dir: str,
    *,
    page_w: int = 1240,
    page_h: int = 1754,
    fmt: str = "png",
) -> list[str | list[str]]:
    """Render grouped DocTag strings to image files.

    For single-page groups the image is saved as ``mn_page_YYYYY.<fmt>``.
    For multi-page groups each page is saved as ``mn_page_YYYYY_PP.<fmt>``
    (PP = 01, 02, …) and the returned entry is a list of those paths.

    Returns a list parallel to *doctag_groups*; each element is either a
    single path string (single-page) or a list of path strings (multi-page).
    """
    from build_document import render_doctag as _render  # type: ignore[import-untyped]

    out_dir = Path(render_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths: list[str | list[str]] = []
    total_rendered = 0

    for row_idx, group in enumerate(doctag_groups):
        row_num = row_idx + 1
        if len(group) == 1:
            img = _render(group[0], page_w, page_h)
            fp = out_dir / f"mn_page_{row_num:05d}.{fmt}"
            img.save(str(fp))
            all_paths.append(str(fp))
            total_rendered += 1
        else:
            page_paths: list[str] = []
            for page_idx, dt in enumerate(group):
                img = _render(dt, page_w, page_h)
                fp = out_dir / f"mn_page_{row_num:05d}_{page_idx + 1:02d}.{fmt}"
                img.save(str(fp))
                page_paths.append(str(fp))
                total_rendered += 1
            all_paths.append(page_paths)

        if total_rendered % 50 == 0 and total_rendered > 0:
            print(f"  Rendered {total_rendered} pages …")

    print(f"Rendered {total_rendered} pages to {render_dir}")
    return all_paths


# ---------------------------------------------------------------------------
# Group-aware helpers
# ---------------------------------------------------------------------------

def batch_translate_doctag_groups(
    doctag_groups: list[list[str]],
    translate_fn,
    *,
    progress: bool = True,
) -> list[list[str]]:
    """Translate groups of DocTag strings, preserving the group structure.

    Flattens *doctag_groups* into a single list, translates it via
    :func:`batch_translate_doctags`, then re-packs the results into the
    original group structure.
    """
    flat = [dt for group in doctag_groups for dt in group]
    translated_flat = batch_translate_doctags(flat, translate_fn, progress=progress)
    result: list[list[str]] = []
    idx = 0
    for group in doctag_groups:
        result.append(translated_flat[idx: idx + len(group)])
        idx += len(group)
    return result


def _build_save_entries(
    translated_groups: list[list[str]],
    image_path_groups: list[str | list[str]] | None,
    source_groups: list[list[str]] | None,
) -> tuple[list[str], list[str | list[str]] | None, list[str] | None]:
    """Expand group-structured data into flat lists for :func:`save_jsonl`.

    For single-page groups one entry is emitted.  For multi-page groups
    N per-page entries are emitted (separate annotations) followed by one
    combined entry whose doctag uses ``<page_break>`` to separate pages and
    whose image field is the full list of page image paths.
    """
    save_doctags: list[str] = []
    save_images: list[str | list[str]] = []
    save_sources: list[str] = []
    has_images = image_path_groups is not None
    has_sources = source_groups is not None

    for row_idx, group in enumerate(translated_groups):
        img_entry: str | list[str] = (
            image_path_groups[row_idx] if has_images and image_path_groups else ""
        )
        src_group: list[str] = (
            source_groups[row_idx] if has_sources and source_groups else []
        )

        if len(group) == 1:
            save_doctags.append(group[0])
            if has_images:
                save_images.append(img_entry)
            if has_sources and src_group:
                save_sources.append(src_group[0])
        else:
            # --- Per-page separate annotations ---
            for page_idx, dt in enumerate(group):
                save_doctags.append(dt)
                if has_images:
                    if isinstance(img_entry, list):
                        save_images.append(img_entry[page_idx])
                    else:
                        save_images.append(img_entry)
                if has_sources and src_group:
                    save_sources.append(src_group[page_idx])

            # --- Combined annotation with <page_break> ---
            save_doctags.append(combine_doctag_pages(group))
            if has_images:
                save_images.append(img_entry)
            if has_sources and src_group:
                save_sources.append(combine_doctag_pages(src_group))

    return (
        save_doctags,
        save_images if has_images else None,
        save_sources if has_sources else None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Mongolian synthetic training data from "
                    "HuggingFaceM4/DoclingMatix (text-only, no images).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --output mn_data.jsonl --limit 100\n"
            "  %(prog)s --output mn_data.jsonl --start-instance 50 --end-instance 150\n"
            "  %(prog)s --output mn_data.jsonl --translator argos\n"
            "  %(prog)s --output mn_data.jsonl --render --render-dir mn_pages/\n"
            "  %(prog)s --output mn_data.jsonl --append\n"
        ),
    )

    # --- Data source ---
    src = parser.add_argument_group("data source")
    src.add_argument(
        "--max-shards", type=int, default=1,
        help="Maximum number of parquet shards to read (default: 1)",
    )
    src.add_argument(
        "--limit", type=int, default=100,
        help="Maximum number of dataset row groups to process (default: 100)",
    )
    src.add_argument(
        "--start-instance", type=int, default=0,
        help="First dataset row index to process, 0-based (default: 0)",
    )
    src.add_argument(
        "--end-instance", type=int, default=None,
        help="Last dataset row index to process, exclusive (default: no limit)",
    )
    src.add_argument(
        "--input", "-i", default=None,
        help="Read DocTag strings from a local JSONL/JSON file instead of "
             "downloading from HuggingFace",
    )
    src.add_argument(
        "--hf-token", default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )

    # --- Translation ---
    tr = parser.add_argument_group("translation")
    tr.add_argument(
        "--translator", choices=list(TRANSLATOR_FACTORIES.keys()),
        default="google",
        help="Translation backend (default: google)",
    )
    tr.add_argument(
        "--source-lang", default="en",
        help="Source language code (default: en)",
    )
    tr.add_argument(
        "--target-lang", default="mn",
        help="Target language code (default: mn for Mongolian)",
    )

    # --- Output ---
    out = parser.add_argument_group("output")
    out.add_argument(
        "--output", "-o", default="mongolian_data.jsonl",
        help="Output JSONL path (default: mongolian_data.jsonl)",
    )
    out.add_argument(
        "--split-ratio", type=float, default=0.0,
        help="Train/val split ratio; 0 = no split (default: 0)",
    )
    out.add_argument(
        "--keep-source", action="store_true",
        help="Include original English DocTag alongside translation in output",
    )
    out.add_argument(
        "--append", action="store_true",
        help="Append translated annotations to an existing JSONL file "
             "instead of overwriting it",
    )

    # --- Rendering ---
    rnd = parser.add_argument_group("rendering")
    rnd.add_argument(
        "--render", action="store_true",
        help="Render translated DocTags to images using build_document.py",
    )
    rnd.add_argument(
        "--render-dir", default="mongolian_renders",
        help="Directory for rendered pages (default: mongolian_renders/)",
    )
    rnd.add_argument(
        "--render-format", choices=["png", "pdf"], default="png",
        help="Rendered image format (default: png)",
    )

    args = parser.parse_args()

    # ---- 1. Load DocTag strings (grouped by source row) ----
    if args.input:
        from build_document import load_doctags_from_file  # type: ignore[import-untyped]
        flat_doctags = load_doctags_from_file(args.input)
        # Wrap each doctag in its own single-element group
        source_groups: list[list[str]] = [[dt] for dt in flat_doctags]
        # Apply instance range if requested
        source_groups = source_groups[
            args.start_instance: args.end_instance
        ]
        print(f"Loaded {len(source_groups)} DocTag group(s) from {args.input}")
    else:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        source_groups = load_doctags_from_doclingmatix(
            max_shards=args.max_shards,
            limit=args.limit,
            hf_token=hf_token,
            start_instance=args.start_instance,
            end_instance=args.end_instance,
        )

    if not source_groups:
        print("No DocTag strings found. Nothing to translate.", file=sys.stderr)
        sys.exit(1)

    total_pages = sum(len(g) for g in source_groups)
    print(f"\nTranslating {total_pages} DocTag page(s) across "
          f"{len(source_groups)} group(s) "
          f"[{args.source_lang}→{args.target_lang}] "
          f"using {args.translator} backend …")

    # ---- 2. Translate ----
    translate_fn = TRANSLATOR_FACTORIES[args.translator](
        args.source_lang, args.target_lang,
    )

    t0 = time.time()
    translated_groups = batch_translate_doctag_groups(source_groups, translate_fn)
    elapsed = time.time() - t0
    print(f"Translation complete in {elapsed:.1f}s "
          f"({total_pages} pages in {len(translated_groups)} groups)")

    # ---- 3. Optionally render ----
    image_path_groups: list[str | list[str]] | None = None
    if args.render:
        print(f"\nRendering translated DocTags to {args.render_dir} …")
        image_path_groups = render_doctag_groups(
            translated_groups, args.render_dir, fmt=args.render_format,
        )

    # ---- 4. Build flat save lists (expand multi-page groups) ----
    src_groups_for_save = source_groups if args.keep_source else None
    save_doctags, save_images, save_sources = _build_save_entries(
        translated_groups, image_path_groups, src_groups_for_save,
    )

    # ---- 5. Save ----
    if args.split_ratio > 0:
        train_path, val_path = split_and_save(
            save_doctags, args.output,
            split_ratio=args.split_ratio,
            source_doctags=save_sources,
            image_paths=save_images,
            append=args.append,
        )
        print(f"\nDone! Train: {train_path}")
        if val_path:
            print(f"       Val:   {val_path}")
    else:
        written = save_jsonl(
            save_doctags, args.output,
            source_doctags=save_sources,
            image_paths=save_images,
            append=args.append,
        )
        print(f"\nDone! Saved {len(save_doctags)} records to {written}")

    # ---- 6. Render hint ----
    if not args.render:
        print(
            "\nTip: to render the translated DocTags to images, re-run with:\n"
            f"  python generate_mongolian_data.py --input {args.output} "
            f"--render --render-dir mongolian_renders/\n"
            "Or use build_document.py directly:\n"
            f"  python build_document.py --input {args.output} "
            "--output mongolian_renders/ --format png"
        )


if __name__ == "__main__":
    main()
