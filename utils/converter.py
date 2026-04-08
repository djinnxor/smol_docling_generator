# --- Function: Convert image to DocTag using custom model ---
def image_to_doctag(image_path, model_id="doctag_model_currently_the_best"):
    """
    Convert an image to a DocTag string using the specified model and processor.
    Returns the DocTag string with <picture> tags removed.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id)
    model.eval()
    SYSTEM_PROMPT = "Convert this document page to DocTags format."
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
        images=Image.open(image_path).convert("RGB"),
        text=prompt_text,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1536,
            do_sample=False,
            use_cache=True,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    doctag = processor.batch_decode(generated_ids, skip_special_tokens=False)[0].strip()
    doctag = re.sub(r'<picture>.*?</picture>', '', doctag, flags=re.DOTALL)
    return doctag

import pytesseract
from pytesseract import Output
from docling.document_converter import DocumentConverter
import json
import time
import os
import mlx
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from build_document import render_doctag
import re

model_id = "doctag_model_currently_the_best"

# Disable GPU for MLX/ML frameworks (if supported)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id)
model.eval()

SYSTEM_PROMPT = "Convert this document page to DocTags format."
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

doctag = re.sub(r'<picture>.*?</picture>', '', doctag, flags=re.DOTALL)

# --- Example usage: Convert image to DocTag and render ---
image_path = "/home/rookie/Documents/Minje/list/403/document_image.jpg"
doctag = image_to_doctag(image_path)
print("Predicted DocTag (no <picture>):", doctag)
img = render_doctag(doctag)
img.save("rendered_doctag.png")
# --- New logic: Predict words and group by y coordinate ---
def extract_words_and_group_by_y(image_path, y_threshold=10):
    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=Output.DICT, lang="eng")
    words = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            words.append({'word': word, 'x': x, 'y': y, 'w': w, 'h': h})

    # Group words by y coordinate (line separation)
    words.sort(key=lambda w: w['y'])
    lines = []
    current_line = []
    last_y = None
    for word in words:
        if last_y is None or abs(word['y'] - last_y) <= y_threshold:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
        last_y = word['y']
    if current_line:
        lines.append(current_line)
    return lines

lines = extract_words_and_group_by_y(image_path)

# Print grouped words by line
for i, line in enumerate(lines):
    line_text = ' '.join([w['word'] for w in line])
    print(f"Line {i+1}: {line_text}")

# Optionally, reconstruct a new image with separated words (draw bounding boxes)

# --- Render predicted words on a new blank image at their predicted locations ---
from PIL import ImageDraw, ImageFont

# Use the same size as the original image for the blank canvas
orig_image = Image.open(image_path).convert("RGB")
blank = Image.new("RGB", orig_image.size, (255, 255, 255))
draw = ImageDraw.Draw(blank)

# Try to use a default font; fallback to PIL's default if not found
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except Exception:
    font = ImageFont.load_default()

for line in lines:
    for word in line:
        x, y, w, h = word['x'], word['y'], word['w'], word['h']
        draw.text((x, y), word['word'], fill=(0, 0, 0), font=font)
        draw.rectangle([x, y, x + w, y + h], outline=(200, 0, 0), width=1)

blank.save('words_rendered_on_blank.png')
print("Saved rendered words to words_rendered_on_blank.png")