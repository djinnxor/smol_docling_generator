import json
import os
from pathlib import Path

# Set your paths here
INPUT_JSONL = "all.jsonl"  # or your input file
OUTPUT_JSONL = "training_data_val_matched.jsonl"  # output file
IMAGES_DIR = "labeler_data/uploads"  # directory with images

# Get all image filenames in the images directory
image_files = set(os.listdir(IMAGES_DIR))

matched = 0
skipped = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as infile, open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    for line in infile:
        record = json.loads(line)
        image_path = record.get("image")
        if image_path is None:
            skipped += 1
            continue
        # Only use the filename part for matching
        image_name = os.path.basename(image_path)
        if image_name in image_files:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            matched += 1
        else:
            skipped += 1

print(f"Matched records: {matched}")
print(f"Skipped records: {skipped}")
print(f"Output written to {OUTPUT_JSONL}")
