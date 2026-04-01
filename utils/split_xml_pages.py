"""
Split a multi-page XML annotation file into separate per-page XML files.

Input XML structure (a document wrapping multiple pages):

    <document>
        <page num="1" image="scan_001">...</page>
        <page num="2" image="scan_002">...</page>
    </document>

Output: one XML file per <page> element, written to --output_dir and named
after the page's "image" attribute value:

    <output_dir>/scan_001_annotations.xml
    <output_dir>/scan_002_annotations.xml

If the root element is already a <page> (single-page file), it is written
as-is.

Usage:
    # Split a single file
    python split_xml_pages.py annotations.xml --output_dir dataset/xml_annotations

    # Split all XML files in a directory
    python split_xml_pages.py raw_xml_dir/ --output_dir dataset/xml_annotations
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


# Characters allowed in output filenames (beyond alphanumerics)
_FILENAME_SAFE_CHARS = "-_."


def _page_to_xml_string(page_el: ET.Element) -> str:
    """Serialise a <page> element to a pretty-printed XML string."""
    raw_xml = ET.tostring(page_el, encoding="unicode")
    try:
        pretty = minidom.parseString(raw_xml).toprettyxml(indent="  ")
        lines = pretty.split("\n")
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines)
    except Exception:
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + raw_xml


def split_xml_pages(xml_path: str, output_dir: str) -> list[str]:
    """
    Parse *xml_path* and write one XML file per ``<page>`` element to
    *output_dir*.

    Each output file is named ``{image_value}_annotations.xml`` where
    ``image_value`` is taken from the page's ``image`` attribute.  If that
    attribute is absent or empty, the page number is used as a fallback.

    Returns
    -------
    list[str]
        Absolute paths of every file written.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    os.makedirs(output_dir, exist_ok=True)

    if root.tag == "page":
        pages = [root]
    else:
        pages = root.findall("page")

    if not pages:
        print(f"  No <page> elements found in {xml_path}; skipping.")
        return []

    written: list[str] = []
    for page in pages:
        image_value = (page.get("image") or "").strip()
        if not image_value:
            image_value = f"page_{page.get('num', 'unknown')}"

        # Sanitise to a safe filename (keep alphanumerics and _FILENAME_SAFE_CHARS)
        safe_name = "".join(
            c if (c.isalnum() or c in _FILENAME_SAFE_CHARS) else "_" for c in image_value
        )
        out_filename = f"{safe_name}_annotations.xml"
        out_path = os.path.join(output_dir, out_filename)

        xml_output = _page_to_xml_string(page)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(xml_output)

        written.append(out_path)
        print(f"  Wrote {out_path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split multi-page XML annotation files into individual per-page files. "
            "Each output file is named after the page's 'image' attribute value."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Path to a multi-page XML file, or a directory whose *.xml files "
            "should all be processed."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/xml_annotations",
        help=(
            "Directory where the per-page XML files are written "
            "(default: dataset/xml_annotations)."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        xml_files = sorted(input_path.glob("*.xml"))
        if not xml_files:
            print(f"No XML files found in {args.input}")
            sys.exit(1)
        total_written = 0
        for xml_file in xml_files:
            print(f"Processing {xml_file.name} …")
            written = split_xml_pages(str(xml_file), args.output_dir)
            total_written += len(written)
        print(
            f"\nDone. Extracted {total_written} page file(s) to {args.output_dir}"
        )

    elif input_path.is_file():
        print(f"Processing {input_path.name} …")
        written = split_xml_pages(str(input_path), args.output_dir)
        print(
            f"\nDone. Extracted {len(written)} page file(s) to {args.output_dir}"
        )

    else:
        print(f"Error: '{args.input}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
