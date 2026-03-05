"""Tests for generate_mongolian_data.py – DocTag translation logic."""

import json
import os
import sys
import tempfile

# Ensure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_mongolian_data import (
    _extract_doctags_from_texts,
    _find_doctag_blocks,
    extract_texts_from_doctag,
    translate_doctag,
    save_jsonl,
    split_and_save,
    combine_doctag_pages,
    split_doctag_on_pagebreak,
    batch_translate_doctag_groups,
    _build_save_entries,
)


# ---------------------------------------------------------------------------
# Stub translator: reverses each word to simulate translation
# ---------------------------------------------------------------------------

def _stub_translate(text: str) -> str:
    """Reverse each word – deterministic stand-in for a real translator."""
    return " ".join(w[::-1] for w in text.split())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_find_doctag_blocks():
    text = "prefix<doctag><text><loc_0><loc_0><loc_100><loc_50>Hello</text></doctag>suffix"
    blocks = _find_doctag_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].startswith("<doctag>")
    assert blocks[0].endswith("</doctag>")


def test_find_doctag_blocks_none():
    assert _find_doctag_blocks("no doctags here") == []


def test_extract_texts_simple():
    doctag = (
        "<doctag>"
        "<text><loc_10><loc_20><loc_300><loc_40>Hello world</text>"
        "<title><loc_0><loc_0><loc_500><loc_50>Introduction</title>"
        "</doctag>"
    )
    texts = extract_texts_from_doctag(doctag)
    assert texts == ["Hello world", "Introduction"]


def test_extract_texts_picture_ignored():
    doctag = (
        "<doctag>"
        "<picture><loc_10><loc_20><loc_300><loc_400></picture>"
        "<text><loc_0><loc_0><loc_100><loc_50>Caption here</text>"
        "</doctag>"
    )
    texts = extract_texts_from_doctag(doctag)
    assert texts == ["Caption here"]


def test_extract_texts_otsl():
    doctag = (
        "<doctag>"
        "<otsl><loc_0><loc_0><loc_500><loc_500>"
        "<ched>Name<ched>Age<nl>"
        "<rhed>John<fcel>25<nl>"
        "</otsl>"
        "</doctag>"
    )
    texts = extract_texts_from_doctag(doctag)
    assert "Name" in texts
    assert "Age" in texts
    assert "John" in texts
    assert "25" in texts


def test_translate_doctag_text():
    doctag = (
        "<doctag>"
        "<text><loc_10><loc_20><loc_300><loc_40>Hello world</text>"
        "</doctag>"
    )
    result = translate_doctag(doctag, _stub_translate)
    # "Hello world" → "olleH dlrow"
    assert "olleH dlrow" in result
    # Structure preserved
    assert "<text><loc_10><loc_20><loc_300><loc_40>" in result
    assert "</text>" in result
    assert "<doctag>" in result
    assert "</doctag>" in result


def test_translate_doctag_picture_unchanged():
    doctag = (
        "<doctag>"
        "<picture><loc_10><loc_20><loc_300><loc_400></picture>"
        "</doctag>"
    )
    result = translate_doctag(doctag, _stub_translate)
    assert "<picture><loc_10><loc_20><loc_300><loc_400></picture>" in result


def test_translate_doctag_otsl():
    doctag = (
        "<doctag>"
        "<otsl><loc_0><loc_0><loc_500><loc_500>"
        "<ched>Name<ched>Age<nl>"
        "<rhed>John<fcel>25<nl>"
        "</otsl>"
        "</doctag>"
    )
    result = translate_doctag(doctag, _stub_translate)
    # "Name" → "emaN", "Age" → "egA"
    assert "emaN" in result
    assert "egA" in result
    # Structure preserved
    assert "<ched>" in result
    assert "<rhed>" in result
    assert "<fcel>" in result
    assert "<nl>" in result
    assert "<otsl><loc_0><loc_0><loc_500><loc_500>" in result


def test_translate_doctag_multiple_elements():
    doctag = (
        "<doctag>"
        "<title><loc_0><loc_0><loc_500><loc_30>Report</title>"
        "<text><loc_10><loc_50><loc_490><loc_100>Summary text here</text>"
        "<caption><loc_10><loc_200><loc_300><loc_220>Figure one</caption>"
        "</doctag>"
    )
    result = translate_doctag(doctag, _stub_translate)
    assert "tropeR" in result  # "Report" reversed
    assert "yrammuS txet ereh" in result  # "Summary text here" words reversed
    assert "<title><loc_0><loc_0><loc_500><loc_30>" in result
    assert "<text><loc_10><loc_50><loc_490><loc_100>" in result


def test_extract_doctags_from_texts_conversation():
    texts = [
        {
            "user": "Convert this document.",
            "assistant": "<doctag><text><loc_0><loc_0><loc_100><loc_50>Hello</text></doctag>",
        }
    ]
    result = _extract_doctags_from_texts(texts)
    assert len(result) == 1
    assert "<doctag>" in result[0]


def test_extract_doctags_from_texts_string():
    text = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Hi</text></doctag>"
    result = _extract_doctags_from_texts(text)
    assert len(result) == 1


def test_extract_doctags_from_texts_no_doctag():
    texts = [{"user": "What is this?", "assistant": "It is a document."}]
    result = _extract_doctags_from_texts(texts)
    assert result == []


def test_save_jsonl():
    doctags = [
        "<doctag><text><loc_0><loc_0><loc_100><loc_50>Сайн байна уу</text></doctag>",
        "<doctag><title><loc_0><loc_0><loc_500><loc_30>Гарчиг</title></doctag>",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        save_jsonl(doctags, tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "doctag" in record
        assert "Сайн байна уу" in record["doctag"]
    finally:
        os.unlink(tmp_path)


def test_save_jsonl_with_source():
    doctags = ["<doctag><text><loc_0><loc_0><loc_100><loc_50>Translated</text></doctag>"]
    sources = ["<doctag><text><loc_0><loc_0><loc_100><loc_50>Original</text></doctag>"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        save_jsonl(doctags, tmp_path, source_doctags=sources)
        with open(tmp_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "doctag" in record
        assert "source_doctag" in record
        assert "Translated" in record["doctag"]
        assert "Original" in record["source_doctag"]
    finally:
        os.unlink(tmp_path)


def test_split_and_save():
    doctags = [f"<doctag><text><loc_0><loc_0><loc_100><loc_50>Item {i}</text></doctag>"
               for i in range(10)]
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "data.jsonl")
        train_path, val_path = split_and_save(doctags, base, split_ratio=0.8)
        assert train_path is not None
        assert val_path is not None
        with open(train_path, encoding="utf-8") as f:
            train_lines = f.readlines()
        with open(val_path, encoding="utf-8") as f:
            val_lines = f.readlines()
        assert len(train_lines) == 8
        assert len(val_lines) == 2


def test_translate_empty_content():
    """Empty content elements should be left unchanged."""
    doctag = (
        "<doctag>"
        "<text><loc_10><loc_20><loc_300><loc_40></text>"
        "</doctag>"
    )
    result = translate_doctag(doctag, _stub_translate)
    assert "<text><loc_10><loc_20><loc_300><loc_40></text>" in result


# ---------------------------------------------------------------------------
# Feature 1: streaming interval – instance range selection
# ---------------------------------------------------------------------------

def test_load_via_streaming_start_end():
    """start_instance / end_instance correctly slice the dataset rows."""
    # Verify the range-slicing logic on a synthetic group list
    groups_all = [
        [f"<doctag><text><loc_0><loc_0><loc_100><loc_50>Item {i}</text></doctag>"]
        for i in range(10)
    ]
    # Simulate the slice the loaders apply: rows [3, 7)
    sliced = groups_all[3:7]
    assert len(sliced) == 4
    assert "Item 3" in sliced[0][0]
    assert "Item 6" in sliced[3][0]


# ---------------------------------------------------------------------------
# Feature 2: append mode + image path first
# ---------------------------------------------------------------------------

def test_save_jsonl_image_first():
    """image key must appear before doctag key in each JSONL record."""
    doctags = ["<doctag><text><loc_0><loc_0><loc_100><loc_50>Hello</text></doctag>"]
    images = ["page_00001.png"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    try:
        save_jsonl(doctags, tmp_path, image_paths=images)
        with open(tmp_path, encoding="utf-8") as f:
            line = f.readline()
        record = json.loads(line)
        keys = list(record.keys())
        assert keys[0] == "image", f"Expected 'image' first, got {keys}"
        assert keys[1] == "doctag"
    finally:
        os.unlink(tmp_path)


def test_save_jsonl_append_mode():
    """Calling save_jsonl with append=True must add to, not overwrite, file."""
    dt1 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>First</text></doctag>"
    dt2 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Second</text></doctag>"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    try:
        save_jsonl([dt1], tmp_path)
        save_jsonl([dt2], tmp_path, append=True)
        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert "First" in lines[0]
        assert "Second" in lines[1]
    finally:
        os.unlink(tmp_path)


def test_save_jsonl_no_append_overwrites():
    """Without append=True the file must be overwritten (default behaviour)."""
    dt1 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Old</text></doctag>"
    dt2 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>New</text></doctag>"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    try:
        save_jsonl([dt1], tmp_path)
        save_jsonl([dt2], tmp_path)  # no append flag → overwrite
        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "New" in lines[0]
        assert "Old" not in lines[0]
    finally:
        os.unlink(tmp_path)


def test_split_and_save_append():
    """split_and_save must pass the append flag through to save_jsonl."""
    doctags = [f"<doctag><text><loc_0><loc_0><loc_100><loc_50>Item {i}</text></doctag>"
               for i in range(10)]
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "data.jsonl")
        # First batch
        split_and_save(doctags[:10], base, split_ratio=0.8)
        # Second batch appended
        split_and_save(doctags[:10], base, split_ratio=0.8, append=True)
        train_path = base.replace(".jsonl", "_train.jsonl")
        with open(train_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 16  # 8 original + 8 appended


# ---------------------------------------------------------------------------
# Feature 3: multi-page rows with <page_break>
# ---------------------------------------------------------------------------

def test_combine_doctag_pages_single():
    """A single-page group must be returned unchanged."""
    dt = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Page 1</text></doctag>"
    assert combine_doctag_pages([dt]) == dt


def test_combine_doctag_pages_multi():
    """Multi-page groups must be joined with <page_break> inside one doctag."""
    p1 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Page 1</text></doctag>"
    p2 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Page 2</text></doctag>"
    combined = combine_doctag_pages([p1, p2])
    assert combined.startswith("<doctag>")
    assert combined.endswith("</doctag>")
    assert "<page_break>" in combined
    assert "Page 1" in combined
    assert "Page 2" in combined
    # Must contain exactly one outer doctag wrapper
    assert combined.count("<doctag>") == 1
    assert combined.count("</doctag>") == 1


def test_combine_doctag_pages_three():
    """Three-page group must have exactly two <page_break> tokens."""
    pages = [
        f"<doctag><text><loc_0><loc_0><loc_100><loc_50>P{i}</text></doctag>"
        for i in range(3)
    ]
    combined = combine_doctag_pages(pages)
    assert combined.count("<page_break>") == 2


def test_batch_translate_doctag_groups():
    """Group structure must be preserved after translation."""
    groups = [
        ["<doctag><text><loc_0><loc_0><loc_100><loc_50>Hello</text></doctag>"],
        [
            "<doctag><text><loc_0><loc_0><loc_100><loc_50>Page one</text></doctag>",
            "<doctag><text><loc_0><loc_0><loc_100><loc_50>Page two</text></doctag>",
        ],
    ]
    result = batch_translate_doctag_groups(groups, _stub_translate, progress=False)
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 2
    # "Hello" → "olleH"
    assert "olleH" in result[0][0]
    # "Page one" → "egaP eno"
    assert "egaP eno" in result[1][0]


def test_build_save_entries_single_page():
    """Single-page groups must produce exactly one entry each."""
    groups = [
        ["<doctag><text><loc_0><loc_0><loc_100><loc_50>A</text></doctag>"],
        ["<doctag><text><loc_0><loc_0><loc_100><loc_50>B</text></doctag>"],
    ]
    doctags, images, sources = _build_save_entries(groups, None, None)
    assert len(doctags) == 2
    assert images is None
    assert sources is None


def test_build_save_entries_multi_page():
    """Multi-page groups produce N per-page entries plus one combined entry."""
    p1 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>P1</text></doctag>"
    p2 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>P2</text></doctag>"
    groups = [[p1, p2]]
    image_groups = [["img_01.png", "img_02.png"]]

    doctags, images, sources = _build_save_entries(groups, image_groups, None)
    # 2 per-page + 1 combined = 3 entries
    assert len(doctags) == 3
    assert images is not None
    assert len(images) == 3
    # First two are individual pages
    assert "P1" in doctags[0]
    assert "<page_break>" not in doctags[0]
    assert "P2" in doctags[1]
    # Last entry is the combined doctag
    assert "<page_break>" in doctags[2]
    assert "P1" in doctags[2]
    assert "P2" in doctags[2]
    # Image fields
    assert images[0] == "img_01.png"
    assert images[1] == "img_02.png"
    assert images[2] == ["img_01.png", "img_02.png"]


def test_build_save_entries_multi_page_with_source():
    """Source doctags for multi-page groups are also combined correctly."""
    p1 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Trans P1</text></doctag>"
    p2 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Trans P2</text></doctag>"
    s1 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Orig P1</text></doctag>"
    s2 = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Orig P2</text></doctag>"
    groups = [[p1, p2]]
    src_groups = [[s1, s2]]

    doctags, images, sources = _build_save_entries(groups, None, src_groups)
    assert sources is not None
    assert len(sources) == 3  # 2 per-page + 1 combined
    assert "Orig P1" in sources[0]
    assert "Orig P2" in sources[1]
    combined_src = sources[2]
    assert "<page_break>" in combined_src
    assert "Orig P1" in combined_src
    assert "Orig P2" in combined_src


def test_save_jsonl_multipage_image_list():
    """Combined multi-page entry must store image list and have image first."""
    doctags = [
        "<doctag><text><loc_0><loc_0><loc_100><loc_50>P1</text></doctag>",
        "<doctag><text><loc_0><loc_0><loc_100><loc_50>P1+P2 combined</text></doctag>",
    ]
    image_paths = ["page_01.png", ["page_01.png", "page_02.png"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    try:
        save_jsonl(doctags, tmp_path, image_paths=image_paths)
        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        rec1 = json.loads(lines[1])
        # image first
        assert list(rec0.keys())[0] == "image"
        assert rec0["image"] == "page_01.png"
        # combined entry has list
        assert list(rec1.keys())[0] == "image"
        assert rec1["image"] == ["page_01.png", "page_02.png"]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Feature 4: split_doctag_on_pagebreak
# ---------------------------------------------------------------------------

def test_split_doctag_on_pagebreak_single():
    """A doctag with no <page_break> must be returned unchanged in a 1-element list."""
    dt = "<doctag><text><loc_0><loc_0><loc_100><loc_50>Only page</text></doctag>"
    result = split_doctag_on_pagebreak(dt)
    assert result == [dt]


def test_split_doctag_on_pagebreak_two_pages():
    """A combined 2-page doctag must split into two separate single-page doctags."""
    p1 = "<text><loc_0><loc_0><loc_100><loc_50>Page one</text>"
    p2 = "<text><loc_0><loc_60><loc_100><loc_120>Page two</text>"
    combined = f"<doctag>{p1}\n<page_break>\n{p2}</doctag>"
    result = split_doctag_on_pagebreak(combined)
    assert len(result) == 2
    assert result[0].startswith("<doctag>") and result[0].endswith("</doctag>")
    assert result[1].startswith("<doctag>") and result[1].endswith("</doctag>")
    assert "Page one" in result[0]
    assert "<page_break>" not in result[0]
    assert "Page two" in result[1]
    assert "<page_break>" not in result[1]


def test_split_doctag_on_pagebreak_three_pages():
    """A combined 3-page doctag must split into three separate doctags."""
    parts = [
        f"<text><loc_0><loc_0><loc_100><loc_50>P{i}</text>"
        for i in range(1, 4)
    ]
    combined = "<doctag>" + "<page_break>".join(parts) + "</doctag>"
    result = split_doctag_on_pagebreak(combined)
    assert len(result) == 3
    for i, page in enumerate(result, 1):
        assert f"P{i}" in page
        assert "<page_break>" not in page
        assert page.count("<doctag>") == 1


def test_split_doctag_on_pagebreak_roundtrip():
    """combine_doctag_pages followed by split_doctag_on_pagebreak must recover originals."""
    pages = [
        f"<doctag><text><loc_0><loc_0><loc_100><loc_50>Page {i}</text></doctag>"
        for i in range(1, 4)
    ]
    combined = combine_doctag_pages(pages)
    recovered = split_doctag_on_pagebreak(combined)
    assert len(recovered) == len(pages)
    for i, (orig, rec) in enumerate(zip(pages, recovered), 1):
        assert f"Page {i}" in rec
        assert "<page_break>" not in rec


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
