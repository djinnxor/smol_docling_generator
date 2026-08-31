import argparse
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch


def compute_token_stats(jsonl_path, model_id, adapter_path=None, field="doctag", batch_size=16, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    # For token statistics, only the tokenizer is needed. Adapter/model loading is not required.

    counts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            text = obj[field]
            tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
            counts.append(len(tokens))

    if not counts:
        print("No samples found.")
        return
    print(f"Token statistics for {len(counts)} samples:")
    print(f"  Avg: {sum(counts)/len(counts):.2f}")
    print(f"  Min: {min(counts)}")
    print(f"  Max: {max(counts)}")
    print(f"  Median: {sorted(counts)[len(counts)//2]}")
    print(f"  95th percentile: {sorted(counts)[int(len(counts)*0.95)]}")
    print(f"  99th percentile: {sorted(counts)[int(len(counts)*0.99)]}")
    print(processor.image_processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute token statistics for a dataset using a (finetuned) tokenizer.")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file with dataset.")
    parser.add_argument("--model_id", required=True, help="Base model or tokenizer path.")
    # Adapter argument is ignored for token stats, kept for backward compatibility
    parser.add_argument("--adapter", default=None, help="(Ignored) Optional: LoRA/PEFT adapter path.")
    parser.add_argument("--field", default="doctag", help="Field in JSONL to tokenize (default: doctag)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    compute_token_stats(args.jsonl, args.model_id, None, args.field, args.batch_size, args.max_samples)
