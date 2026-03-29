"""
Run DocTag inference on an image file or a PDF file using the locally trained
model in the doctag_model/ folder (LoRA fine-tune of ibm-granite/granite-docling-258M).
If that folder is absent the upstream HuggingFace model is used as a fallback.

For each input page the model is prompted with the page image and asked to
produce a <doctag>…</doctag> string.  Results are printed to stdout by default
and can optionally be written to a JSON or plain-text file.

Usage examples
--------------
# Single image  (uses doctag_model/ automatically)
python inference.py --input page.png

# Multi-page PDF
python inference.py --input document.pdf

# Override the model (any HuggingFace ID or local path)
python inference.py --input document.pdf \\
                    --model_id ./my_other_model \\
                    --output results.json

# Limit to the first 5 pages, use GPU
python inference.py --input document.pdf --pages 5 --device cuda

# CPU-only, mixed precision disabled
python inference.py --input document.pdf --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from build_document import parse_doctag, render_doctag, save_output  


# Local folder produced by train.py --output_dir (preferred default)
LOCAL_DOCTAG_MODEL = "doctag_model"
# Upstream model used when the local folder is absent
FALLBACK_MODEL_ID = "ibm-granite/granite-docling-258M"
# Resolved default: use local folder if it exists, else HuggingFace
DEFAULT_MODEL_ID = LOCAL_DOCTAG_MODEL if Path(LOCAL_DOCTAG_MODEL).is_dir() else FALLBACK_MODEL_ID
SYSTEM_PROMPT = "Convert this document page to DocTags format."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _resolve_device_dtype(device: str, dtype: torch.dtype | None) -> tuple[str, torch.dtype]:
    """Pick the best available device and matching dtype when not specified by the caller.

    Priority: CUDA > MPS > CPU.  bf16 is preferred on CUDA (Ampere+) and CPU
    when the hardware supports it; fp16 is used on MPS; fp32 is the safe fallback.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if dtype is None:
        if device.startswith("cuda"):
            # bf16 is the fastest safe dtype on Ampere+ GPUs (compute ≥ 8.0).
            # fp16 is safe on Pascal+ (compute ≥ 6.0).  Fall back to fp32 on
            # anything older to avoid unsupported-dtype errors.
            cap = torch.cuda.get_device_capability()
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif cap[0] >= 6:
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

    return device, dtype


def _load_model_and_processor(model_id: str, device: str, dtype: torch.dtype):
    """Load the processor and model onto *device* with *dtype*.

    If *model_id* points to a directory that contains ``adapter_config.json``
    (i.e. a PEFT / LoRA adapter saved by train.py) the base model named inside
    that config is loaded first, then the LoRA weights are merged into it for
    efficient inference.  Any other path or HuggingFace model ID is loaded
    directly with AutoModelForImageTextToText.
    """
    model_path = Path(model_id)
    is_peft_adapter = (model_path / "adapter_config.json").exists()

    if is_peft_adapter:
        import json as _json
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required to load a LoRA adapter. Install it with:\n"
                "  pip install peft"
            ) from exc

        with open(model_path / "adapter_config.json", encoding="utf-8") as f:
            adapter_cfg = _json.load(f)
        base_model_id = adapter_cfg["base_model_name_or_path"]

        print(f"Loading LoRA adapter from: {model_id}", file=sys.stderr)
        print(f"  base model: {base_model_id}", file=sys.stderr)
        # Checkpoints only contain adapter weights, not the full processor config.
        # Try the checkpoint first (works for full fine-tune saves), then fall back
        # to the base model (required for raw trainer checkpoints).
        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except (ValueError, OSError):
            print(f"  processor not found in checkpoint, loading from base model", file=sys.stderr)
            processor = AutoProcessor.from_pretrained(base_model_id)
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id, dtype=dtype
        )

        # Resize embeddings to match the processor's vocabulary (which may
        # include extra special tokens added during fine-tuning).
        base_model.resize_token_embeddings(len(processor.tokenizer))

        base_model = base_model.to(device)
        model = PeftModel.from_pretrained(base_model, model_id)
        # Merge LoRA weights into the base model for faster inference
        model = model.merge_and_unload()
    else:
        print(f"Loading model: {model_id}", file=sys.stderr)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype
        ).to(device)

    model.config.use_cache = True
    model.eval()
    return processor, model


def _pdf_to_images(pdf_path: str, dpi: int = 150, max_pages: int | None = None) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page)."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required to process PDF files. Install it with:\n"
            "  pip install pdf2image"
        ) from exc

    kwargs: dict = {"dpi": dpi}
    if max_pages is not None:
        kwargs["last_page"] = max_pages

    return convert_from_path(pdf_path, **kwargs)


def _infer_page(
    image: Image.Image,
    processor,
    model,
    device: str,
    max_new_tokens: int,
) -> str:
    """Run the model on a single page image and return the raw DocTag string."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        images=image.convert("RGB"),
        text=prompt_text,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Decode only the newly generated tokens (skip the prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    # skip_special_tokens=False is intentional: DocTag tokens such as <doctag>,
    # <text>, <loc_N>, etc. are registered as special tokens in the model's
    # vocabulary.  Stripping them would corrupt the DocTag output.
    doctag = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return doctag.strip()


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def run_inference_from_checkpoint(
    checkpoint_path: str,
    input_path: str,
    output_path: str | None = None,
    device: str = "auto",
    dtype: torch.dtype | None = None,
    max_new_tokens: int = 9000,
    dpi: int = 96,
    max_pages: int | None = None,
) -> list[dict]:
    """
    Run DocTag inference using a training checkpoint directory.

    *checkpoint_path* should point to a directory that contains
    ``adapter_config.json`` and ``adapter_model.safetensors`` (e.g.
    ``doctag_model/checkpoint-3764``).  The base model is resolved
    automatically from ``adapter_config.json``.

    All other parameters are forwarded to :func:`run_inference`.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")
    if not (checkpoint / "adapter_config.json").exists():
        raise ValueError(
            f"No adapter_config.json found in {checkpoint_path}. "
            "Make sure this is a valid PEFT / LoRA checkpoint directory."
        )
    print(f"Using checkpoint: {checkpoint_path}", file=sys.stderr)
    return run_inference(
        input_path=input_path,
        model_id=str(checkpoint),
        output_path=output_path,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        dpi=dpi,
        max_pages=max_pages,
    )


def run_inference(
    input_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    output_path: str | None = None,
    device: str = "auto",
    dtype: torch.dtype | None = None,
    max_new_tokens: int = 1536,  # ~8% headroom above the observed 1420-token max
    dpi: int = 96,
    max_pages: int | None = None,
) -> list[dict]:
    """
    Run DocTag inference on *input_path* (image or PDF).

    Returns a list of dicts with keys:
        page   – 1-based page number
        doctag – the generated <doctag>…</doctag> string

    If *output_path* is given the list is also written to disk:
        *.json  → JSON array
        other   → plain text, one doctag per page separated by a blank line
    """
    input_path = str(input_path)
    suffix = Path(input_path).suffix.lower()

    device, dtype = _resolve_device_dtype(device, dtype)
    processor, model = _load_model_and_processor(model_id, device, dtype)

    # Collect page images
    if suffix == ".pdf":
        print(f"Converting PDF to images (dpi={dpi}) …", file=sys.stderr)
        images = _pdf_to_images(input_path, dpi=dpi, max_pages=max_pages)
    elif suffix in IMAGE_EXTENSIONS:
        images = [Image.open(input_path)]
    else:
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            f"Supported: PDF and {sorted(IMAGE_EXTENSIONS)}"
        )

    results: list[dict] = []
    doctags = []
    total = len(images)
    import time
    t0 = time.time()
    page_times = []
    for i, img in enumerate(images, 1):
        print(f"Processing page {i}/{total} …", file=sys.stderr)
        t_page_start = time.time()
        doctag = _infer_page(img, processor, model, device, max_new_tokens)
        t_page_end = time.time()
        elapsed = t_page_end - t_page_start
        page_times.append(elapsed)
        print(f"  Inference time: {elapsed:.2f} sec", file=sys.stderr)
        results.append({"page": i, "doctag": doctag})
        doctags.append(doctag)

    t1 = time.time()
    total_time = t1 - t0

    images = []
    for i, dt in enumerate(doctags):
        img = render_doctag(dt, 1240, 1754, 20)
        images.append(img)
        print(f"  Page {i + 1}: {len(parse_doctag(dt))} elements")

    fmt = "pdf" if len(images) > 1 else "png"
    if fmt == "pdf":
        written = save_output(images, "doctag_render.pdf", fmt)
    else:
        written = save_output(images, "doctag_render.png", fmt)
    for fp in written:
        print(f"  → {fp}")
    print(f"Total inference time: {total_time:.2f} sec", file=sys.stderr)
    print("Done.")

    # Save / print
    if output_path is not None:
        _save_results(results, output_path)
    else:
        _print_results(results)

    return results


def _save_results(results: list[dict], output_path: str) -> None:
    """Write inference results to *output_path*."""
    path = Path(output_path)
    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} page(s) to {output_path}", file=sys.stderr)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"=== Page {r['page']} ===\n")
                f.write(r["doctag"] + "\n\n")
        print(f"Saved {len(results)} page(s) to {output_path}", file=sys.stderr)


def _print_results(results: list[dict]) -> None:
    """Print inference results to stdout."""
    for r in results:
        print(f"=== Page {r['page']} ===")
        print(r["doctag"])
        print()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DocTag inference on an image or PDF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image file (.png, .jpg, …) or a PDF file",
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help=(
            "HuggingFace model ID or path to a fine-tuned checkpoint / LoRA adapter. "
            f"Defaults to '{LOCAL_DOCTAG_MODEL}' (local trained model) when that folder "
            f"exists, otherwise falls back to '{FALLBACK_MODEL_ID}'. "
            "Mutually exclusive with --checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a training checkpoint directory containing adapter_config.json "
            "and adapter_model.safetensors (e.g. doctag_model/checkpoint-3764). "
            "Mutually exclusive with --model_id."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to save the results. "
            "Use a .json extension for JSON output, any other extension for plain text. "
            "Omit to print to stdout."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device to run inference on: 'cpu', 'cuda', 'cuda:0', 'mps', etc. "
            "Defaults to 'cpu'."
        ),
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bfloat16 precision (auto-selected on CUDA when supported)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force float16 precision",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=9000,
        help="Maximum number of tokens to generate per page (default: 9000)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=96,
        help="DPI used when converting PDF pages to images (default: 96)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Maximum number of PDF pages to process (default: all pages)",
    )
    args = parser.parse_args()

    if args.checkpoint and args.model_id:
        parser.error("--checkpoint and --model_id are mutually exclusive")

    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    # Let _resolve_device_dtype auto-select dtype unless the user forced one
    dtype: torch.dtype | None = (
        torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else None)
    )

    common_kwargs = dict(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        dpi=args.dpi,
        max_pages=args.pages,
    )

    if args.checkpoint:
        run_inference_from_checkpoint(checkpoint_path=args.checkpoint, **common_kwargs)
    else:
        run_inference(model_id=args.model_id or DEFAULT_MODEL_ID, **common_kwargs)


if __name__ == "__main__":
    main()
