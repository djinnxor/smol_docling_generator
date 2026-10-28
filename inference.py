"""
Run DocTag inference on an image file or a PDF file using the locally trained
model in the doctag_model/ folder (LoRA fine-tune of ibm-granite/granite-docling-258M).
If that folder is absent the upstream HuggingFace model is used as a fallback.


python inference.py --input page.png
python inference.py --input document.pdf

python inference.py --input document.pdf \\
                    --model_id ./my_other_model \\
                    --output results.json

python inference.py --input document.pdf --pages 5 --device cuda

python inference.py --input document.pdf --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
import tempfile

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor
from utils.build_document import (
    parse_doctag,
    render_doctag,
    draw_doctag_overlay,
    save_output,
)


# Local folder produced by train.py --output_dir (preferred default)
LOCAL_DOCTAG_MODEL = "doctag_restart_mixed_30___"
# Upstream model used when the local folder is absent
FALLBACK_MODEL_ID = "ibm-granite/granite-docling-258M"
# Resolved default: use local folder if it exists, else HuggingFace
DEFAULT_MODEL_ID = LOCAL_DOCTAG_MODEL if Path(LOCAL_DOCTAG_MODEL).is_dir() else FALLBACK_MODEL_ID
SYSTEM_PROMPT = "Convert this document page to DocTags format."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_YOLO_MODEL_PATH = "/home/rookie/Downloads/best.pt"


def detect_id(image_path: str, yolo_model_path: str = DEFAULT_YOLO_MODEL_PATH) -> dict:
    """Run YOLO ID detection on *image_path*.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO ID detection. Install it with:\n"
            "  pip install ultralytics"
        ) from exc

    model = YOLO(yolo_model_path)
    original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    angle_results: list[tuple[int, float]] = []  # (angle, best_conf)

    for angle in [0, 90, 270]:
        candidate = original.rotate(angle, expand=True)
        detections = model(candidate, verbose=False)
        conf = max(
            (float(b.conf[0]) for r in detections for b in (r.boxes or [])),
            default=0.0,
        )
        angle_results.append((angle, conf))
        print(f"[YOLO] angle={angle:3d}°  conf={conf:.4f}", file=sys.stderr)

    best_angle, best_conf_probe = max(angle_results, key=lambda x: x[1])

    if best_angle != 0:
        print(
            f"[YOLO] Rotating image by {best_angle}° (conf={best_conf_probe:.4f})",
            file=sys.stderr,
        )
    else:
        print(
            f"[YOLO] No rotation needed (best angle=0°, conf={best_conf_probe:.4f})",
            file=sys.stderr,
        )

    oriented = original.rotate(best_angle, expand=True)

    # ── Final detection on the best-oriented image ────────────────────────
    final_results = model(oriented, verbose=False)

    best_box = None
    best_conf_final = -1.0
    for result in final_results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf_final:
                best_conf_final = conf
                best_box = box

    iw, ih = oriented.size
    if best_box is None:
        print("[YOLO] No ID detected – using EXIF-corrected original image.", file=sys.stderr)
        return {
            "oriented": oriented,
            "cropped": oriented,
            "id_box": None,
            "crop_box": (0, 0, iw, ih),
            "angle": best_angle,
        }

    dx1, dy1, dx2, dy2 = (int(v) for v in best_box.xyxy[0].tolist())

    # Add a small margin so diagonal cards don't lose corner text
    margin_x = int((dx2 - dx1) * 0.04)
    margin_y = int((dy2 - dy1) * 0.04)
    x1 = max(0, dx1 - margin_x)
    y1 = max(0, dy1 - margin_y)
    x2 = min(iw, dx2 + margin_x)
    y2 = min(ih, dy2 + margin_y)

    print(
        f"[YOLO] ID detected (conf={best_conf_final:.4f}, angle={best_angle}°) – "
        f"crop [{x1}, {y1}, {x2}, {y2}]",
        file=sys.stderr,
    )
    cropped = oriented.crop((x1, y1, x2, y2))
    return {
        "oriented": oriented,
        "cropped": cropped,
        "id_box": (dx1, dy1, dx2, dy2),
        "crop_box": (x1, y1, x2, y2),
        "angle": best_angle,
    }


def detect_and_crop_id(image_path: str, yolo_model_path: str = DEFAULT_YOLO_MODEL_PATH) -> Image.Image:
    """Backward-compatible wrapper that returns only the cropped ID image."""
    return detect_id(image_path, yolo_model_path=yolo_model_path)["cropped"]


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
    else:
        if device.startswith("cuda"):
            cap = torch.cuda.get_device_capability()
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                dtype = torch.float32
            elif dtype == torch.float16 and cap[0] < 6:
                dtype = torch.float32
        elif device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        elif device == "mps" and dtype == torch.bfloat16:
            dtype = torch.float16

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


def _resolve_render_output(render_output: str | None, page_count: int) -> tuple[str, str]:
    """Return the visualization output path and inferred format.

    If a path is supplied explicitly, its extension determines the format.
    Otherwise the legacy default remains: PDF for multi-page output and PNG for
    single-page output.
    """
    if render_output is not None:
        path = str(render_output)
        suffix = Path(path).suffix.lower()
        fmt = "pdf" if suffix == ".pdf" else "png"
        return path, fmt

    default_path = "doctag_render.pdf" if page_count > 1 else "doctag_render.png"
    fmt = "pdf" if default_path.endswith(".pdf") else "png"
    return default_path, fmt


def _infer_page(
    image: Image.Image,
    processor,
    model,
    device: str,
    max_new_tokens: int,
    process_size: int | None = None,
) -> str:
    """Run the model on a single page image and return the raw DocTag string.

    *process_size*, when given, is passed to the image processor as
    ``size={"longest_edge": process_size}`` so the processor keeps the image at
    that resolution instead of resizing to its default (longest_edge=2048).
    Use this together with a matching pre-resize to feed the model a fixed
    resolution (e.g. 512).
    """
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

    proc_kwargs = dict(
        images=image.convert("RGB"),
        text=prompt_text,
        return_tensors="pt",
    )
    if process_size is not None:
        proc_kwargs["size"] = {"longest_edge": process_size}
    inputs = processor(**proc_kwargs)
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
    process_size: int | None = None,
    render_output: str | None = None,
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
        process_size=process_size,
        render_output=render_output,
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
    process_size: int | None = None,
    render_output: str | None = None,
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
        doctag = _infer_page(img, processor, model, device, max_new_tokens, process_size)
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

    render_output_path, fmt = _resolve_render_output(render_output, len(images))
    written = save_output(images, render_output_path, fmt)
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
        "--render_output",
        default=None,
        help=(
            "Where to save the rendered DocTag visualization. "
            "Accepts .png or .pdf. Defaults to doctag_render.png for single-page output "
            "and doctag_render.pdf for multi-page output."
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
        default=6000,
        help="Maximum number of tokens to generate per page (default: 6000)",
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
    parser.add_argument(
        "--id_image",
        action="store_true",
        help=(
            "Treat --input as an ID document image. The image is first passed through "
            "the YOLO ID detection model; the highest-confidence detection crop is then "
            "used for DocTag inference."
        ),
    )
    parser.add_argument(
        "--id_resize",
        type=int,
        default=512,
        help=(
            "Square edge (pixels) the YOLO-cropped ID is resized to before DocTag "
            "inference. The processor is told to keep this resolution (size.longest_edge) "
            "instead of upscaling to its 2048 default. Only used with --id_image "
            "(default: 512)."
        ),
    )
    parser.add_argument(
        "--yolo_model",
        default=DEFAULT_YOLO_MODEL_PATH,
        help=(
            f"Path to the YOLO model weights used for ID detection "
            f"(default: {DEFAULT_YOLO_MODEL_PATH}). Only used with --id_image."
        ),
    )
    parser.add_argument(
        "--id_img_show",
        action="store_true",
        help="Show the cropped ID image after YOLO detection. Only used with --id_image."
    )
    parser.add_argument(
        "--overlay_output",
        default="id_overlay.png",
        help=(
            "Where to save the overlay image (YOLO ID bbox + DocTag element "
            "boxes + Mongolian text drawn on the original oriented image). "
            "Only produced with --id_image (default: id_overlay.png)."
        ),
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

    # --id_image: run YOLO crop first, save to a temp file, then infer
    input_path = args.input
    _tmp_file = None
    detection: dict | None = None
    if args.id_image:
        detection = detect_id(args.input, yolo_model_path=args.yolo_model)
        cropped = detection["cropped"]
        suffix = Path(args.input).suffix or ".png"
        _tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        cropped.save(_tmp_file.name)

        if args.id_img_show:
            import cv2, numpy as np
            cv2_img = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
            max_display = 900
            dh, dw = cv2_img.shape[:2]
            scale = min(max_display / dw, max_display / dh, 1.0)
            display = cv2.resize(cv2_img, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow("Cropped ID", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        _tmp_file.close()
        input_path = _tmp_file.name

    common_kwargs = dict(
        input_path=input_path,
        output_path=args.output,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        dpi=args.dpi,
        max_pages=args.pages,
        render_output=args.render_output,
    )

    try:
        if args.checkpoint:
            results = run_inference_from_checkpoint(checkpoint_path=args.checkpoint, **common_kwargs)
        else:
            results = run_inference(model_id=args.model_id or DEFAULT_MODEL_ID, **common_kwargs)
    finally:
        if _tmp_file is not None:
            import os
            os.unlink(_tmp_file.name)

    # For ID images, draw the YOLO box, the DocTag element boxes and the
    # Mongolian text onto the full oriented image and save it.
    if detection is not None and results:
        overlay = draw_doctag_overlay(
            detection["oriented"],
            results[0]["doctag"],
            region=detection["crop_box"],
            id_box=detection["id_box"],
        )
        overlay.save(args.overlay_output)
        print(f"  → {args.overlay_output} (ID bbox + DocTag boxes + text overlay)")


if __name__ == "__main__":
    main()
