"""
Run DocTag inference on an image file or a PDF file using
ibm-granite/granite-docling-258M (or any compatible fine-tuned checkpoint).

For each input page the model is prompted with the page image and asked to
produce a <doctag>…</doctag> string.  Results are printed to stdout by default
and can optionally be written to a JSON or plain-text file.

Usage examples
--------------
# Single image
python inference.py --input page.png

# Multi-page PDF
python inference.py --input document.pdf

# Use a locally fine-tuned checkpoint and save output
python inference.py --input document.pdf \\
                    --model_id ./doctag_model \\
                    --output results.json

# Limit to the first 5 pages, use GPU
python inference.py --input document.pdf --pages 5 --device cuda

# CPU-only, mixed precision disabled
python inference.py --input document.pdf --device cpu
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"
SYSTEM_PROMPT = "Convert this document page to DocTags format."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_model_and_processor(model_id: str, device: str, dtype: torch.dtype):
    """Load the processor and model onto *device* with *dtype*."""
    print(f"Loading model: {model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
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

def run_inference(
    input_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    output_path: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_new_tokens: int = 4096,
    dpi: int = 150,
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
    total = len(images)
    for i, img in enumerate(images, 1):
        print(f"Processing page {i}/{total} …", file=sys.stderr)
        doctag = _infer_page(img, processor, model, device, max_new_tokens)
        results.append({"page": i, "doctag": doctag})

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
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID or path to a fine-tuned checkpoint (default: {DEFAULT_MODEL_ID})",
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
        help="Device to run inference on: 'cpu', 'cuda', 'cuda:0', 'mps', etc. (default: cpu)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision (recommended for CUDA GPUs that support it)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate per page (default: 4096)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI used when converting PDF pages to images (default: 150)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Maximum number of PDF pages to process (default: all pages)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    dtype = (
        torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )

    run_inference(
        input_path=args.input,
        model_id=args.model_id,
        output_path=args.output,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        dpi=args.dpi,
        max_pages=args.pages,
    )


if __name__ == "__main__":
    main()
