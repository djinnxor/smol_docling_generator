from docling.document_converter import DocumentConverter
import json
import time
import os
import mlx
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

def rebuild_detections_on_blank(source_pdf, elements_by_page, out_dir="rebuilt_pages", zoom=2.0):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(source_pdf)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    for page_num, page_elements in elements_by_page.items():
        pdf_page = doc[page_num - 1]
        pdf_w, pdf_h = pdf_page.rect.width, pdf_page.rect.height
        
        # Create blank white page
        img_w, img_h = int(pdf_w * zoom), int(pdf_h * zoom)
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        
        sx, sy = img_w / pdf_w, img_h / pdf_h

        for element in page_elements:
            bbox = getattr(element, "bbox", None)
            if not bbox:
                continue

            x = getattr(bbox, "x", None)
            y = getattr(bbox, "y", None)
            w = getattr(bbox, "width", None)
            h = getattr(bbox, "height", None)
            if None in (x, y, w, h):
                continue

            # Convert PDF coords (bottom-left origin) to image coords (top-left origin)
            x1 = x * sx
            y1 = (pdf_h - (y + h)) * sy
            x2 = (x + w) * sx
            y2 = (pdf_h - y) * sy

            element_type = element.__class__.__name__.lower()
            content = element.export_to_markdown() if hasattr(element, "export_to_markdown") else str(element)

            if "table" in element_type:
                # Draw table structure
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                # Split table into lines and render
                lines = content.split("\n")
                line_height = 14
                y_offset = y1 + 5
                for line in lines[:20]:  # Limit lines to fit
                    draw.text((x1 + 5, y_offset), line.strip(), fill="black", font=font)
                    y_offset += line_height
            else:
                # Draw text content
                draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
                # Word wrap text to fit bbox
                lines = []
                words = content.split()
                current_line = []
                for word in words:
                    test_line = " ".join(current_line + [word])
                    bbox_test = draw.textbbox((0, 0), test_line, font=font)
                    if bbox_test[2] - bbox_test[0] <= (x2 - x1 - 10):
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(" ".join(current_line))
                
                # Render wrapped lines
                line_height = 14
                y_offset = y1 + 5
                for line in lines:
                    if y_offset + line_height > y2:
                        break
                    draw.text((x1 + 5, y_offset), line, fill="black", font=font)
                    y_offset += line_height

        out_path = os.path.join(out_dir, f"page_{page_num:03d}.png")
        img.save(out_path)

    doc.close()
    print(f"Saved rebuilt pages to: {out_dir}")

source = "tables.pdf"
converter = DocumentConverter(allowed_formats=["pdf"])
start_time = time.time()
result = converter.convert(source)
end_time = time.time()

print(result.document.export_to_markdown())
print(f"Conversion completed in {end_time - start_time:.2f} seconds")

doctags_data = {"doctags": []}
doctags_data["raw_doctags"] = result.document.export_to_doctags(result.document)

pages = result.document.pages
if hasattr(pages, "items"):
    page_iter = pages.items()
else:
    page_iter = enumerate(pages, 1)

elements_by_page = defaultdict(list)

for page_num, page in page_iter:
    for element in getattr(page, "children", []):
        elements_by_page[page_num].append(element)
        bbox = getattr(element, "bbox", None)
        doctags_data["doctags"].append({
            "type": element.__class__.__name__,
            "content": element.export_to_markdown() if hasattr(element, "export_to_markdown") else str(element),
            "page": page_num,
            "bbox": bbox.as_tuple() if bbox and hasattr(bbox, "as_tuple") else None,
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

rebuild_detections_on_blank(source, elements_by_page, out_dir="rebuilt_pages", zoom=2.0)

print("Exported structured DocTags to output_doctags.json")