import logging as _logging

_logger = _logging.getLogger(__name__)

# Default timeout (seconds) for Vision API HTTP requests.
# Override by setting the environment variable GOOGLE_OCR_TIMEOUT at startup.
import os as _os
import os
import shutil
import base64
import requests
from requests.exceptions import RequestException
_VISION_TIMEOUT: float = float(os.environ.get("GOOGLE_OCR_TIMEOUT", "30"))
from difflib import SequenceMatcher
from PIL import Image, ImageDraw
import json
import torch
import csv
from pdf2image import convert_from_path
import cv2
import numpy as np

class GoogleOCR:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.vision_api_url = "https://vision.googleapis.com/v1/images:annotate"
    
    def _build_request_body(self, encoded_image: str) -> dict:
        """Shared Vision API request body for DOCUMENT_TEXT_DETECTION."""
        return {
            "requests": [
                {
                    "image": {"content": encoded_image},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                }
            ]
        }

    def _get_auth_headers(self) -> dict:
        """Return HTTP headers for the Vision API (api_key or bearer token)."""
        if self.api_key:
            return {"Content-Type": "application/json"}
        return {
            "Authorization": f"Bearer {getattr(self, 'access_token', '')}",
            "Content-Type": "application/json",
        }

    def _get_url(self) -> str:
        """Return the Vision API URL with optional api_key query parameter."""
        if self.api_key:
            return f"{self.vision_api_url}?key={self.api_key}"
        return self.vision_api_url

    def _post_vision_raw(self, encoded_image: str) -> dict:
        """Post to the Vision API and return the raw JSON response dict (or {}).

        Timeout is controlled by the ``GOOGLE_OCR_TIMEOUT`` env var (default 30 s).
        """
        body = self._build_request_body(encoded_image)
        try:
            response = requests.post(
                self._get_url(), json=body, headers=self._get_auth_headers(),
                timeout=_VISION_TIMEOUT,
            )
        except RequestException as exc:
            _logger.warning("Google Vision API request failed: %s", exc)
            return {}
        if response.status_code == 200:
            return response.json()
        _logger.warning(
            "Google Vision API returned status %s: %s",
            response.status_code, response.text[:200],
        )
        return {}

    def _post_vision(self, encoded_image: str) -> str:
        """Post to the Vision API and return the detected text (or '').

        Delegates to _post_vision_raw and extracts the top-level text block.
        """
        result = self._post_vision_raw(encoded_image)
        if result and "responses" in result and result["responses"]:
            annotations = result["responses"][0].get("textAnnotations", [])
            if annotations:
                return annotations[0].get("description", "")
        return ""

    def google_vision_ocr(self, image_path):
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        return self._post_vision(base64.b64encode(content).decode("utf-8"))

    def google_vision_ocr_bytes(self, image_bytes: bytes) -> str:
        """Run OCR on raw image bytes (avoids writing a temp file to disk)."""
        return self._post_vision(base64.b64encode(image_bytes).decode("utf-8"))

    def google_vision_ocr_words_bytes(self, image_bytes: bytes) -> list:
        """OCR image bytes and return word-level bounding box results.

        Returns a list of dicts::

            {"text": str, "vertices": [[x, y], [x, y], [x, y], [x, y]]}

        The ``vertices`` list contains four pixel-coordinate corners (top-left,
        top-right, bottom-right, bottom-left) of each detected word/phrase.
        Caller is responsible for normalising to image dimensions.
        """
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = self._post_vision_raw(encoded)
        words = []
        if response and "responses" in response:
            annotations = response["responses"][0].get("textAnnotations", [])
            # index 0 = full-page text block; indices 1+ = individual word tokens
            for ann in annotations[1:]:
                text = ann.get("description", "").strip()
                if not text:
                    continue
                poly = ann.get("boundingPoly", {}).get("vertices", [])
                vertices = [[float(v.get("x", 0)), float(v.get("y", 0))] for v in poly]
                if len(vertices) == 4:
                    words.append({"text": text, "vertices": vertices})
        return words
    
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
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")
    
    def recognize(self, image: np.ndarray) -> str:
        """Run OCR on a single image/cell crop and return text."""
        if image is None or image.size == 0:
            return ""
        # Skip tiny crops
        h, w = image.shape[:2]
        if w < 5 or h < 5:
            return ""
        encoded_image = self._image_to_base64(image)
        
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
        try:
            resp = requests.post(self.url, json=request_body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            annotations = data.get("responses", [{}])[0].get("textAnnotations", [])
            if annotations:
                return annotations[0].get("description", "").strip().replace("\n", " ")
        except Exception as e:
            print(f"[GoogleVisionOCR] Error: {e}")
        return ""


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