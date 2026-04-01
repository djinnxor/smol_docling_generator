import json
import re

INPUT_JSONL = "training_data_train.jsonl"  # input file
OUTPUT_JSONL = "training_data_train_matched_picture.jsonl"  # output file

# Regex to count <picture> tags
PICTURE_TAG_PATTERN = re.compile(r"<picture>")

matched = 0
skipped = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as infile, open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    for line in infile:
        record = json.loads(line)
        image_field = record.get("image")
        doctag_field = record.get("doctag", "")
        # Always 1 image since image is a string
        image_count = 1 if image_field else 0
        picture_count = len(PICTURE_TAG_PATTERN.findall(doctag_field))
        if image_count == picture_count:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            matched += 1
        else:
            skipped += 1

print(f"Matched records: {matched}")
print(f"Skipped records: {skipped}")
print(f"Output written to {OUTPUT_JSONL}")
