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
  - Batch translation to minimise API round-trips
  - Supports offline translation via argostranslate

Usage:
    # Generate 100 translated samples (default)
    python generate_mongolian_data.py --output mongolian_data.jsonl

    # Process more samples from more parquet shards
    python generate_mongolian_data.py --output mongolian_data.jsonl \\
        --limit 1000 --max-shards 5

    # Use offline argostranslate backend
    python generate_mongolian_data.py --translator argos \\
        --output mongolian_data.jsonl

    # Also render translated DocTags to PNG images for training
    python generate_mongolian_data.py --output mongolian_data.jsonl \\
        --render --render-dir mongolian_renders/

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


def _find_doctag_blocks(text: str) -> list[str]:
    """Find all <doctag>…</doctag> blocks in a string."""
    return _DOCTAG_BLOCK_RE.findall(text)


def load_doctags_from_doclingmatix(
    *,
    max_shards: int = 1,
    limit: int = 100,
    hf_token: str | None = None,
) -> list[str]:
    """Load DocTag strings from HuggingFaceM4/DoclingMatix.

    Uses direct parquet column-projection to read ONLY the ``texts``
    column, completely bypassing image data for maximum speed.

    Falls back to ``datasets`` streaming if direct parquet reading fails.
    """
    print(f"Loading DoclingMatix (max_shards={max_shards}, limit={limit}) …")

    doctags = _load_via_parquet(
        max_shards=max_shards, limit=limit, hf_token=hf_token,
    )
    if doctags is not None:
        return doctags

    print("Direct parquet loading failed; falling back to datasets streaming …")
    return _load_via_streaming(limit=limit, hf_token=hf_token)


def _load_via_parquet(
    *,
    max_shards: int,
    limit: int,
    hf_token: str | None,
) -> list[str] | None:
    """Read DocTag strings directly from parquet files on the Hub.

    Returns None if the required libraries are unavailable or the
    parquet files cannot be listed.
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

    doctags: list[str] = []
    for shard_path in shards:
        if len(doctags) >= limit:
            break
        print(f"  Reading {shard_path.split('/')[-1]} …")
        try:
            with fs.open(shard_path) as fh:
                table = pq.read_table(fh, columns=["texts"])
            for i in range(len(table)):
                if len(doctags) >= limit:
                    break
                texts = table.column("texts")[i].as_py()
                found = _extract_doctags_from_texts(texts)
                doctags.extend(found)
        except Exception as exc:
            print(f"  [warn] Failed to read shard: {exc}", file=sys.stderr)
            continue

    print(f"  Extracted {len(doctags)} DocTag strings")
    return doctags[:limit]


def _load_via_streaming(
    *, limit: int, hf_token: str | None,
) -> list[str]:
    """Fallback: use ``datasets`` streaming mode."""
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

    doctags: list[str] = []
    for row in ds:
        if len(doctags) >= limit:
            break
        texts = row.get("texts", [])
        found = _extract_doctags_from_texts(texts)
        doctags.extend(found)

    print(f"  Extracted {len(doctags)} DocTag strings (streaming)")
    return doctags[:limit]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_jsonl(
    doctags: list[str],
    output_path: str,
    *,
    source_doctags: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> str:
    """Write translated DocTag strings to a JSONL file.

    Each line is a JSON object with at least a ``doctag`` field, and
    optionally ``source_doctag`` (original English) and ``image``
    (rendered page path).
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as fh:
        for i, dt in enumerate(doctags):
            record: dict = {"doctag": dt}
            if source_doctags is not None:
                record["source_doctag"] = source_doctags[i]
            if image_paths is not None and i < len(image_paths):
                record["image"] = image_paths[i]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(p)


def split_and_save(
    doctags: list[str],
    output_base: str,
    split_ratio: float = 0.9,
    *,
    source_doctags: list[str] | None = None,
    image_paths: list[str] | None = None,
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
        source_doctags=train_sources, image_paths=train_images,
    )
    print(f"Saved {split_idx} training records to {train_path}")

    val_written = None
    if split_idx < n:
        save_jsonl(
            doctags[split_idx:], val_path,
            source_doctags=val_sources, image_paths=val_images,
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
            "  %(prog)s --output mn_data.jsonl --translator argos\n"
            "  %(prog)s --output mn_data.jsonl --render --render-dir mn_pages/\n"
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
        help="Maximum number of DocTag samples to process (default: 100)",
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

    # ---- 1. Load DocTag strings ----
    if args.input:
        from build_document import load_doctags_from_file  # type: ignore[import-untyped]
        source_doctags = load_doctags_from_file(args.input)
        print(f"Loaded {len(source_doctags)} DocTag(s) from {args.input}")
    else:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        source_doctags = load_doctags_from_doclingmatix(
            max_shards=args.max_shards,
            limit=args.limit,
            hf_token=hf_token,
        )

    if not source_doctags:
        print("No DocTag strings found. Nothing to translate.", file=sys.stderr)
        sys.exit(1)

    # ---- 2. Translate ----
    print(f"\nTranslating {len(source_doctags)} DocTag(s) "
          f"[{args.source_lang}→{args.target_lang}] "
          f"using {args.translator} backend …")

    translate_fn = TRANSLATOR_FACTORIES[args.translator](
        args.source_lang, args.target_lang,
    )

    t0 = time.time()
    translated_doctags = batch_translate_doctags(source_doctags, translate_fn)
    elapsed = time.time() - t0
    print(f"Translation complete in {elapsed:.1f}s "
          f"({len(translated_doctags)} samples)")

    # ---- 3. Optionally render ----
    image_paths: list[str] | None = None
    if args.render:
        print(f"\nRendering translated DocTags to {args.render_dir} …")
        image_paths = render_doctags(
            translated_doctags, args.render_dir, fmt=args.render_format,
        )

    # ---- 4. Save ----
    sources = source_doctags if args.keep_source else None

    if args.split_ratio > 0:
        train_path, val_path = split_and_save(
            translated_doctags, args.output,
            split_ratio=args.split_ratio,
            source_doctags=sources,
            image_paths=image_paths,
        )
        print(f"\nDone! Train: {train_path}")
        if val_path:
            print(f"       Val:   {val_path}")
    else:
        written = save_jsonl(
            translated_doctags, args.output,
            source_doctags=sources,
            image_paths=image_paths,
        )
        print(f"\nDone! Saved {len(translated_doctags)} records to {written}")

    # ---- 5. Render hint ----
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
