import os
import shutil
import base64
import requests
from difflib import SequenceMatcher
from PIL import Image, ImageDraw
import json
import torch
import csv
from pdf2image import convert_from_path
import os

class GoogleOCR:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.vision_api_url = "https://vision.googleapis.com/v1/images:annotate"
    
    def google_vision_ocr(self, image_path):
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        encoded_image = base64.b64encode(content).decode('utf-8')
        if self.api_key:
            url = f"{self.vision_api_url}?key={self.api_key}"
            headers = {'Content-Type': 'application/json'}
        else:
            url = self.vision_api_url
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
        request_body = {
            "requests": [
                {
                    "image": {
                        "content": encoded_image
                    },
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, json=request_body, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if 'responses' in result and len(result['responses']) > 0:
                response_data = result['responses'][0]
                text_annotations = response_data.get('textAnnotations', [])
                if text_annotations:
                    print(text_annotations[0]['description'])  # Print only the full text
                    return text_annotations[0]['description']
        return ""
    
    def calculate_similarity(self, text1, text2):
        def normalize(text):
            import re
            normalized = re.sub(r'\s+', ' ', text.strip())
            return normalized.lower()

        text1_norm = normalize(text1)
        text2_norm = normalize(text2)

        return SequenceMatcher(None, text1_norm, text2_norm).ratio()

    def _get_mime_type(self, image_path):
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        return mime_types.get(ext, 'image/jpeg')

    def _bbox_to_percent(self, bbox, img_width, img_height):
        return {
            'x': (bbox['x'] / img_width) * 100,
            'y': (bbox['y'] / img_height) * 100,
            'width': (bbox['width'] / img_width) * 100,
            'height': (bbox['height'] / img_height) * 100
        }

if __name__ == "__main__":
    ocr = GoogleOCR(api_key="AIzaSyCFdvigsgMlOnn_gQFaLQgqfhwtWDA7tYs")
    
    pdf_path = "tables.pdf"
    tmp_dir = "tmp_images"
    os.makedirs(tmp_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300, output_folder=tmp_dir, fmt='png')
    results = []
    for idx, img in enumerate(images):
        img_path = os.path.join(tmp_dir, f"page_{idx+1:03d}.png")
        img.save(img_path)
        text_regions = ocr.google_vision_ocr(img_path)
        results.append((idx + 1, text_regions))
    print(results)