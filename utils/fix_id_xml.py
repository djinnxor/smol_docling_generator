import os
import xml.etree.ElementTree as ET


def fix_xml_in_folder(xml_dir: str, new_width: str, new_height: str):
    """
    For each XML file in xml_dir, replace the root tag <id_document> with <page>,
    keep the width, height, and type from the original, set image to '1', and save the file.
    """
    for fname in os.listdir(xml_dir):
        if not fname.endswith('.xml'):
            continue
        fpath = os.path.join(xml_dir, fname)
        tree = ET.parse(fpath)
        root = tree.getroot()
        if root.tag != 'id_document':
            print(f"Skipping {fname}: root tag is not <id_document>")
            continue
        # Get attributes
        t = root.attrib.get('type', '')
        w = root.attrib.get('width', '')
        h = root.attrib.get('height', '')
        # Replace tag and attributes
        root.tag = 'page'
        root.attrib['type'] = t
        root.attrib['image'] = '1'
        root.attrib['width'] = w
        root.attrib['height'] = h
        # Remove any other attributes
        for k in list(root.attrib.keys()):
            if k not in {'type', 'image', 'width', 'height'}:
                del root.attrib[k]
        # Remove all children (if any)
        for child in list(root):
            root.remove(child)
        tree.write(fpath, encoding='utf-8', xml_declaration=True)
        print(f"Fixed {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix XML files for ID document to page conversion.")
    parser.add_argument('--xml_dir', required=True, help='Path to folder containing XML files')
    args = parser.parse_args()
    fix_xml_in_folder(args.xml_dir, None, None)
