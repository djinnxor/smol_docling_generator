import json
from pathlib import Path

from json_fixer import combine_dataset


def test_combine_dataset(tmp_path):
    root = tmp_path / "dataset"
    images = root / "images"
    labels = root / "doctag_labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    image_path = images / "mn_doc_00000.png"
    image_path.write_bytes(b"fakepng")

    payload = {"doctag": "<doctag>\n<title>Sample</title>\n</doctag>"}
    (labels / "mn_doc_00000.json").write_text(json.dumps(payload), encoding="utf-8")

    records = combine_dataset(str(root), str(root / "combined.jsonl"))

    assert len(records) == 1
    assert records[0]["image"].endswith("images/mn_doc_00000.png")
    assert records[0]["doctag"] == payload["doctag"]
    assert (root / "combined.jsonl").exists()
