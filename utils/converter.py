from docling.document_converter import DocumentConverter
import json
import time
import os
import mlx
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from build_document import save_output  


# Disable GPU for MLX/ML frameworks (if supported)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

source = "/home/rookie/Downloads/page_001.png"  # document per local path or URL
image = [Image.open(source)]
image_name = os.path.basename(source)
model_id = "ibm-granite/granite-docling-258M"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float16
        ).to(device)
SYSTEM_PROMPT = "Convert this document page to DocTags format."

prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
_prompt_text = processor.apply_chat_template(
    prompt_messages, add_generation_prompt=True
)
inputs = processor(
        images=image.convert("RGB"),
        text=_prompt_text,
        return_tensors="pt",
    )
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=4000,
        do_sample=False,
        use_cache=True,
    )

input_len = inputs["input_ids"].shape[1]
generated_ids = output_ids[:, input_len:]
doctag = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
doctag = doctag.strip()

written = save_output(image, "doctag_render.png", fmt="png")
print(f"  → {written}")

converter = DocumentConverter(allowed_formats=["pdf"])
start_time = time.time()
result = converter.convert(source)
end_time = time.time()
print(result.document.export_to_markdown())
print(f"Conversion completed in {end_time - start_time:.2f} seconds")

with open("output_doctag.json", "w", encoding="utf-8") as f:
    json.dump(doctag, f, ensure_ascii=False, indent=2)