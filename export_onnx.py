"""export_onnx.py — Convert the doctag_model_id_resume_v6 LoRA fine-tune to ONNX.

doctag_model_id_resume_v6/ is a PEFT/LoRA *adapter* on top of
ibm-granite/granite-docling-258M (plus extra ID special tokens), not a full
model. ONNX export therefore happens in two stages:

  1. merge_lora()  — load the base model, resize its embeddings to the
                     fine-tuned vocab, load the LoRA adapter and fold it into the
                     base weights (merge_and_unload), then save a full fp32
                     HuggingFace model + processor to <MERGED_DIR>.
  2. export_onnx() — run Optimum's ONNX exporter on the merged model. This is a
                     vision-language generative model (Idefics3 / granite-docling),
                     so Optimum emits several ONNX graphs (vision encoder, token
                     embeddings, decoder with KV-cache) under <ONNX_DIR>.

Requirements (Optimum is not installed by default):
    pip install "optimum[exporters]" onnx onnxruntime
    # (optimum-onnx provides the ONNX exporter backend)

Usage:
    python export_onnx.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT        = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT / "doctag_model_id_resume_v6"
MERGED_DIR  = ROOT / "doctag_model_id_resume_v6_merged"
ONNX_DIR    = ROOT / "doctag_model_id_resume_v6_onnx"
BASE_MODEL  = "ibm-granite/granite-docling-258M"  # from adapter_config.json

# Optimum task for a generative image+text -> text model (with KV-cache).
# Use "auto" to let Optimum infer it from the architecture if this string is
# rejected by your Optimum version.
EXPORT_TASK = "image-text-to-text-with-past"


def merge_lora() -> None:
    """Merge the LoRA adapter into the base model and save a full HF model."""
    if (MERGED_DIR / "config.json").exists():
        print(f"[merge] {MERGED_DIR.name} already exists — skipping merge.")
        return

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("peft is required: pip install peft") from exc

    # The adapter dir carries the fine-tuned tokenizer + extra ID tokens.
    # Fall back to the base processor if it can't be loaded from the checkpoint.
    try:
        processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
    except (ValueError, OSError):
        print("[merge] processor not in adapter dir, loading from base model.")
        processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print(f"[merge] loading base model: {BASE_MODEL}")
    base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.float32)

    # Match the fine-tuned vocabulary (extra <register_number>, <gender>, … tokens).
    base.resize_token_embeddings(len(processor.tokenizer))

    print(f"[merge] loading + merging LoRA adapter: {ADAPTER_DIR.name}")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model = model.merge_and_unload()   # fold LoRA weights into the base

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    processor.save_pretrained(MERGED_DIR)
    print(f"[merge] merged model + processor saved to {MERGED_DIR}")


def export_onnx() -> None:
    """Export the merged model to ONNX with the Optimum CLI."""
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "onnx",
        "--model", str(MERGED_DIR),
        "--task", EXPORT_TASK,
        # "--opset", "18",           # uncomment to pin an ONNX opset
        # "--device", "cuda",        # export on GPU if desired
        str(ONNX_DIR),
    ]
    print("[onnx] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        # Fallback: call the Python API directly if the module entrypoint differs.
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=str(MERGED_DIR),
            output=str(ONNX_DIR),
            task=EXPORT_TASK,
        )
    print(f"[onnx] exported ONNX graphs to {ONNX_DIR}")


if __name__ == "__main__":
    merge_lora()
    export_onnx()
