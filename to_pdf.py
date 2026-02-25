from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import json
import time

source = "tables.pdf"

# Enable layout detection model
pipeline_options = PdfPipelineOptions()
pipeline_options.do_layout = True
pipeline_options.do_table_structure = True
pipeline_options.do_ocr = False  # Set True if you want OCR

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

start_time = time.time()
result = converter.convert(source)
end_time = time.time()
print(f"Conversion completed in {end_time - start_time:.2f} seconds")

docling_output = {"pages": []}

for item, level in result.document.iterate_items():
    label = getattr(item, "label", None)
    if label is None:
        continue

    prov = getattr(item, "prov", [])
    for p in prov:
        bbox = getattr(p, "bbox", None)
        page_no = getattr(p, "page_no", None)

        if "table" in str(label).lower():
            content = item.export_to_dataframe().to_string() if hasattr(item, "export_to_dataframe") else str(item)
        elif hasattr(item, "text"):
            content = item.text
        else:
            content = str(item)

        entry = {
            "label": str(label),
            "content": content,
            "page": page_no,
            "location": {
                "x": getattr(bbox, "l", None),
                "y": getattr(bbox, "t", None),
                "width": getattr(bbox, "r", None) - getattr(bbox, "l", 0) if bbox else None,
                "height": getattr(bbox, "b", None) - getattr(bbox, "t", 0) if bbox else None,
            } if bbox else None
        }
        docling_output["pages"].append(entry)

with open("output_doctags.json", "w", encoding="utf-8") as f:
    json.dump(docling_output, f, ensure_ascii=False, indent=2)

print(json.dumps(docling_output, ensure_ascii=False, indent=2))
print("Exported to output_doctags.json")