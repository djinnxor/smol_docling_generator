from docling.document_converter import DocumentConverter
import json
import time
import os
import mlx

# Disable GPU for MLX/ML frameworks (if supported)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

source = "tables.pdf"  # document per local path or URL
converter = DocumentConverter(allowed_formats=["pdf"])
start_time = time.time()
result = converter.convert(source)
end_time = time.time()
print(result.document.export_to_markdown())
print(f"Conversion completed in {end_time - start_time:.2f} seconds")

# Keep doctags as a list so we can append structured entries
doctags_data = {"doctags": []}

# Optional: include raw doctags export as a separate field
doctags_data["raw_doctags"] = result.document.export_to_doctags(result.document)

# If pages is a dict, iterate with .items(); if list-like, use enumerate fallback
pages = result.document.pages
if hasattr(pages, "items"):
    page_iter = pages.items()  # (page_num, page_obj)
else:
    page_iter = enumerate(pages, 1)  # (page_num, page_obj)

for page_num, page in page_iter:
    for element in getattr(page, "children", []):
        bbox = getattr(element, "bbox", None)
        doctags_data["doctags"].append({
            "type": element.__class__.__name__,
            "content": element.export_to_markdown() if hasattr(element, "export_to_markdown") else str(element),
            "page": page_num,
            "bbox": bbox.as_tuple() if bbox else None,
            "location": {
                "page": page_num,
                "x": getattr(bbox, "x", None),
                "y": getattr(bbox, "y", None),
                "width": getattr(bbox, "width", None),
                "height": getattr(bbox, "height", None),
            }
        })

with open("output_doctags.json", "w", encoding="utf-8") as f:
    json.dump(doctags_data, f, ensure_ascii=False, indent=2)

print("Exported structured DocTags to output_doctags.json")