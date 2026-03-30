import json
import random
from pathlib import Path

def split_json_file(input_file, output_train, output_test, split_ratio=0.8):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # Print first 5 rows if data is a list, otherwise read as JSONL
    if isinstance(data, list):
        for i, row in enumerate(data[:5]):
            print(json.dumps(row, ensure_ascii=False))
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(line.strip())
                else:
                    break
        data = [json.loads(line) for line in open(input_file, "r", encoding="utf-8")]

    random.shuffle(data)

    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]


    with open(output_train, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_test, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Split {len(data)} records into {len(train_data)} for training and {len(test_data)} for testing.")

if __name__ == "__main__":
    split_json_file("/home/rookie/Documents/Minje/smol_docling_generator/all.jsonl", "/home/rookie/Documents/Minje/smol_docling_generator/train.jsonl", "/home/rookie/Documents/Minje/smol_docling_generator/test.jsonl")