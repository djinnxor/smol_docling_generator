"""
Evaluate an ID model on a JSONL dataset.

For each sample the model is run on the image and the predicted doctag is
compared against the ground truth.  The script reports:

  • Tag-level metrics  – precision/recall/F1 for every ID field tag
  • Text accuracy      – exact match and CER (character error rate) per tag
  • Character analysis – which characters are most often mis-predicted
  • Worst images       – the N images with the highest CER, with side-by-side diff

Usage
-----
python eval_id_model.py \\
    --model_id doctag_model_id_resume_v4 \\
    --data     id_data_train_split.jsonl \\
    --device   cuda \\
    --limit    100          # optional: only evaluate first N samples
    --output   eval_results.json   # optional: save full per-sample results
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from inference import detect_and_crop_id, DEFAULT_YOLO_MODEL_PATH

# ── Tag extraction ───────────────────────────────────────────────────────────

# All known ID field tags (order matters for display only)
ID_TAGS = [
    "register_number",
    "familyname",
    "surname",
    "name",
    "birthday",
    "gender",
]

_TAG_PATTERN = re.compile(
    r"<(?P<tag>[a-z_]+)>"          # opening tag
    r"(?:<loc_\d+>)*"              # optional loc tokens
    r"(?P<text>[^<]*)"             # text content (may be empty)
    r"</(?P=tag)>",                # matching closing tag
    re.DOTALL,
)


def parse_doctag_fields(doctag: str) -> dict[str, str]:
    """Return {tag_name: text_value} for every ID field found in *doctag*."""
    fields: dict[str, str] = {}
    for m in _TAG_PATTERN.finditer(doctag):
        tag = m.group("tag")
        if tag in ID_TAGS:
            fields[tag] = m.group("text").strip()
    return fields


# ── Character Error Rate ──────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(
                dp[j - 1] if ca == cb else 1 + min(dp[j], ndp[-1], dp[j - 1])
            )
        dp = ndp
    return dp[-1]


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate (0 = perfect, 1 = completely wrong, can exceed 1)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


# ── Character-level diff helpers ─────────────────────────────────────────────

def char_errors(ref: str, hyp: str) -> list[tuple[str, str]]:
    """
    Return (ref_char, hyp_char) pairs that are mismatched via a simple
    Needleman–Wunsch alignment (match=0, mismatch/gap=1).
    Gaps are represented by the empty string "".
    """
    # Build alignment table
    n, m = len(ref), len(hyp)
    INF = n + m + 1
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Traceback
    errors: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            errors.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            errors.append((ref[i - 1], ""))  # deletion
            i -= 1
        else:
            errors.append(("", hyp[j - 1]))  # insertion
            j -= 1
    return errors


# ── Model loading (reuses inference.py helpers) ───────────────────────────────

def _load_model(model_id: str, device: str, dtype: torch.dtype):
    model_path = Path(model_id)
    is_peft = (model_path / "adapter_config.json").exists()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    if is_peft:
        from peft import PeftModel

        with open(model_path / "adapter_config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        base_id = cfg["base_model_name_or_path"]
        print(f"Loading LoRA adapter: {model_id}  (base: {base_id})", file=sys.stderr)
        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except (ValueError, OSError):
            processor = AutoProcessor.from_pretrained(base_id)
        base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=dtype)
        base.resize_token_embeddings(len(processor.tokenizer))
        base = base.to(device)
        model = PeftModel.from_pretrained(base, model_id)
        model = model.merge_and_unload()
    else:
        print(f"Loading model: {model_id}", file=sys.stderr)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device)

    model.config.use_cache = True
    model.eval()
    return processor, model


def _infer(image: Image.Image, processor, model, device: str, max_new_tokens: int) -> str:
    SYSTEM_PROMPT = "Convert this document page to DocTags format."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT}],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image.convert("RGB"), text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    input_len = inputs["input_ids"].shape[1]
    generated = out[:, input_len:]
    return processor.batch_decode(generated, skip_special_tokens=False)[0].strip()


# ── Per-sample scoring ────────────────────────────────────────────────────────

def score_sample(gt_fields: dict[str, str], pred_fields: dict[str, str]) -> dict:
    """Return a rich score dict for one sample."""
    tag_scores: dict[str, dict] = {}
    overall_cer_values: list[float] = []

    for tag in ID_TAGS:
        gt_val = gt_fields.get(tag, "")
        pred_val = pred_fields.get(tag, "")
        present_in_gt = tag in gt_fields
        present_in_pred = tag in pred_fields
        exact = gt_val == pred_val
        c = cer(gt_val, pred_val)
        overall_cer_values.append(c)
        tag_scores[tag] = {
            "gt": gt_val,
            "pred": pred_val,
            "present_in_gt": present_in_gt,
            "present_in_pred": present_in_pred,
            "exact_match": exact,
            "cer": round(c, 4),
        }

    mean_cer = sum(overall_cer_values) / len(overall_cer_values) if overall_cer_values else 0.0
    return {"tags": tag_scores, "mean_cer": round(mean_cer, 4)}


# ── Aggregate analysis ────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    tag_exact: dict[str, list[bool]] = defaultdict(list)
    tag_cer: dict[str, list[float]] = defaultdict(list)
    tag_tp: dict[str, int] = defaultdict(int)   # present in gt & pred
    tag_fp: dict[str, int] = defaultdict(int)   # present in pred, not in gt
    tag_fn: dict[str, int] = defaultdict(int)   # present in gt, not in pred

    # character confusion: (ref_char, hyp_char) → count
    char_confusion: dict[tuple[str, str], int] = defaultdict(int)
    # per-character CER contribution: ref_char → [cer contributions]
    char_error_counts: dict[str, int] = defaultdict(int)
    char_ref_counts: dict[str, int] = defaultdict(int)

    for r in results:
        for tag, ts in r["score"]["tags"].items():
            tag_exact[tag].append(ts["exact_match"])
            tag_cer[tag].append(ts["cer"])
            in_gt = ts["present_in_gt"]
            in_pred = ts["present_in_pred"]
            if in_gt and in_pred:
                tag_tp[tag] += 1
            elif in_pred and not in_gt:
                tag_fp[tag] += 1
            elif in_gt and not in_pred:
                tag_fn[tag] += 1

            # character-level
            ref, hyp = ts["gt"], ts["pred"]
            for c in ref:
                char_ref_counts[c] += 1
            for ref_c, hyp_c in char_errors(ref, hyp):
                char_confusion[(ref_c, hyp_c)] += 1
                char_error_counts[ref_c] += 1

    # Tag metrics
    tag_metrics: dict[str, dict] = {}
    for tag in ID_TAGS:
        tp = tag_tp[tag]
        fp = tag_fp[tag]
        fn = tag_fn[tag]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        exact_list = tag_exact[tag]
        cer_list = tag_cer[tag]
        tag_metrics[tag] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "exact_match_rate": round(sum(exact_list) / len(exact_list), 4) if exact_list else 0.0,
            "mean_cer": round(sum(cer_list) / len(cer_list), 4) if cer_list else 0.0,
        }

    # Character error rate per character (sorted by error rate desc)
    char_cer_table: list[dict] = []
    for ch, ref_count in char_ref_counts.items():
        err_count = char_error_counts.get(ch, 0)
        char_cer_table.append(
            {
                "char": ch,
                "ref_count": ref_count,
                "error_count": err_count,
                "error_rate": round(err_count / ref_count, 4),
            }
        )
    char_cer_table.sort(key=lambda x: (-x["error_rate"], -x["ref_count"]))

    # Top confusions (ref → hyp pairs, sorted by count)
    top_confusions = [
        {"ref": rc, "pred": hc, "count": cnt}
        for (rc, hc), cnt in sorted(char_confusion.items(), key=lambda x: -x[1])
        if rc != hc  # skip correct pairs
    ][:30]

    return {
        "tag_metrics": tag_metrics,
        "char_error_table": char_cer_table[:50],
        "top_char_confusions": top_confusions,
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(results: list[dict], agg: dict, top_n_worst: int = 10) -> None:
    total = len(results)
    mean_cer_all = sum(r["score"]["mean_cer"] for r in results) / total if total else 0.0

    print("=" * 70)
    print(f"  ID MODEL EVALUATION REPORT  ({total} samples)")
    print("=" * 70)
    print(f"\nOverall mean CER : {mean_cer_all:.4f}  {_bar(min(mean_cer_all, 1.0))}")
    print()

    # ── Tag-level metrics ──────────────────────────────────────────────────
    print("─" * 70)
    print("TAG-LEVEL METRICS")
    print("─" * 70)
    hdr = f"{'Tag':<20} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'Exact%':>7}  {'CER':>6}"
    print(hdr)
    print("─" * len(hdr))
    for tag, m in agg["tag_metrics"].items():
        print(
            f"{tag:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['exact_match_rate']*100:>6.1f}%  {m['mean_cer']:>6.4f}"
        )
    print()

    # ── Character error table ──────────────────────────────────────────────
    print("─" * 70)
    print("CHARACTER ERROR RATE  (top 30, sorted by error rate)")
    print("─" * 70)
    print(f"{'Char':<8} {'Ref count':>10} {'Errors':>8} {'Error rate':>11}")
    print("─" * 40)
    for row in agg["char_error_table"][:30]:
        ch = repr(row["char"]) if row["char"] else '""'
        print(
            f"{ch:<8} {row['ref_count']:>10} {row['error_count']:>8} "
            f"{row['error_rate']*100:>10.1f}%"
        )
    print()

    # ── Top character confusions ───────────────────────────────────────────
    print("─" * 70)
    print("TOP CHARACTER CONFUSIONS  (ref → predicted, top 20)")
    print("─" * 70)
    print(f"{'Ref':<6} {'Pred':<6} {'Count':>8}")
    print("─" * 22)
    for row in agg["top_char_confusions"][:20]:
        ref_repr = repr(row["ref"]) if row["ref"] else '"<del>"'
        hyp_repr = repr(row["pred"]) if row["pred"] else '"<ins>"'
        print(f"{ref_repr:<6} → {hyp_repr:<6}  {row['count']:>6}")
    print()

    # ── Worst images ──────────────────────────────────────────────────────
    print("─" * 70)
    print(f"WORST {top_n_worst} IMAGES  (highest mean CER)")
    print("─" * 70)
    worst = sorted(results, key=lambda r: -r["score"]["mean_cer"])[:top_n_worst]
    for rank, r in enumerate(worst, 1):
        print(f"\n#{rank}  {r['image']}")
        print(f"     mean CER = {r['score']['mean_cer']:.4f}")
        for tag in ID_TAGS:
            ts = r["score"]["tags"].get(tag)
            if ts is None:
                continue
            if not ts["present_in_gt"] and not ts["present_in_pred"]:
                continue
            status = "✓" if ts["exact_match"] else "✗"
            print(f"     {status} {tag:<20}  CER={ts['cer']:.3f}")
            if not ts["exact_match"]:
                print(f"       GT  : {ts['gt']}")
                print(f"       PRED: {ts['pred']}")
    print()
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an ID DocTag model on a JSONL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_id", required=True, help="Path to LoRA adapter or model ID")
    parser.add_argument(
        "--data",
        default="id_data_train.jsonl",
        help="JSONL file with 'image' and 'doctag' fields (default: id_data_train.jsonl)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N samples"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max tokens per inference (default 512)"
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON file to save full per-sample results"
    )
    parser.add_argument(
        "--worst", type=int, default=10, help="Number of worst images to show (default 10)"
    )
    parser.add_argument(
        "--id_image",
        action="store_true",
        help="Use YOLO to detect and crop the ID card before inference (same as inference.py --id_image)",
    )
    parser.add_argument(
        "--yolo_model",
        default=DEFAULT_YOLO_MODEL_PATH,
        help=f"Path to YOLO model weights for ID detection (default: {DEFAULT_YOLO_MODEL_PATH})",
    )
    args = parser.parse_args()

    dtype: torch.dtype = (
        torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    )

    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data file not found: {args.data}")
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {args.data}", file=sys.stderr)

    # Load model
    processor, model = _load_model(args.model_id, args.device, dtype)

    # Run evaluation
    results: list[dict] = []
    for idx, sample in enumerate(samples):
        image_path = sample["image"]
        gt_doctag = sample.get("doctag", "")
        print(f"[{idx + 1}/{len(samples)}] {image_path}", file=sys.stderr)

        if not Path(image_path).exists():
            print(f"  WARNING: image not found, skipping", file=sys.stderr)
            continue

        try:
            if args.id_image:
                image = detect_and_crop_id(image_path, yolo_model_path=args.yolo_model)
            else:
                image = Image.open(image_path).convert("RGB")
            pred_doctag = _infer(image, processor, model, args.device, args.max_new_tokens)
        except Exception as e:
            print(f"  ERROR during inference: {e}", file=sys.stderr)
            pred_doctag = ""

        gt_fields = parse_doctag_fields(gt_doctag)
        pred_fields = parse_doctag_fields(pred_doctag)
        score = score_sample(gt_fields, pred_fields)

        results.append(
            {
                "image": image_path,
                "gt_doctag": gt_doctag,
                "pred_doctag": pred_doctag,
                "score": score,
            }
        )

    if not results:
        sys.exit("No results to report.")

    # Aggregate and print
    agg = aggregate(results)
    print_report(results, agg, top_n_worst=args.worst)

    # Optionally save full results
    if args.output:
        out_path = Path(args.output)
        payload = {"aggregate": agg, "samples": results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
