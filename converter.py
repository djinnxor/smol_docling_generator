import json
import time
import os
import fitz  # PyMuPDF

from google_ocr import GoogleOCR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE = "tables.pdf"          # PDF to process
OUTPUT_DIR = "output_pages"    # directory for page images + per-page JSONs
ZOOM = 2.0                     # rendering scale factor (higher = sharper images)

# Provide either an API key or the path to a Google service-account JSON file.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")

# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

ocr = GoogleOCR(
    api_key=GOOGLE_API_KEY if GOOGLE_API_KEY else None,
    credentials_path=GOOGLE_CREDENTIALS if GOOGLE_CREDENTIALS else None,
)

doc = fitz.open(SOURCE)
start_time = time.time()

for page_index in range(len(doc)):
    page = doc[page_index]
    page_num = page_index + 1

    # Render page to a raster image
    mat = fitz.Matrix(ZOOM, ZOOM)
    pix = page.get_pixmap(matrix=mat)
    image_name = f"page_{page_num:03d}.png"
    image_path = os.path.join(OUTPUT_DIR, image_name)
    pix.save(image_path)

    img_width = pix.width
    img_height = pix.height

    # Use Google Vision OCR instead of docling's built-in OCR (e.g. rapidOCR)
    text_regions = ocr.ocr_image(image_path)

    # Build per-page annotation entry
    page_data = {
        "image": image_name,
        "image_width": img_width,
        "image_height": img_height,
        "ocr_results": [
            {"text": text, "bbox": bbox}
            for text, bbox in text_regions
        ],
    }

    # Save JSON named after the corresponding page image
    json_name = f"{os.path.splitext(image_name)[0]}.json"
    json_path = os.path.join(OUTPUT_DIR, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {json_name} (image: {img_width}x{img_height}, "
          f"{len(text_regions)} OCR regions)")

doc.close()
end_time = time.time()
print(f"\nConversion completed in {end_time - start_time:.2f} seconds")
print(f"Per-page JSONs saved to: {OUTPUT_DIR}/")