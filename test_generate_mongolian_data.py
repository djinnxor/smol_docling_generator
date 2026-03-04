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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
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
