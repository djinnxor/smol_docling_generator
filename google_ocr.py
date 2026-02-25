import os
import base64
import requests


class GoogleOCR:
    """Wraps Google Vision API for OCR, returning line-grouped text regions."""

    def __init__(self, api_key=None, credentials_path=None):
        self.api_key = api_key
        self.credentials_path = credentials_path
        self.vision_api_url = "https://vision.googleapis.com/v1/images:annotate"
        if credentials_path and not api_key:
            self.access_token = self._get_access_token()
        else:
            self.access_token = None

    def _get_access_token(self):
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-vision"],
        )
        credentials.refresh(Request())
        return credentials.token

    def ocr_image(self, image_path):
        """
        Run Google Vision OCR on *image_path*.

        Returns a list of ``(text, bbox)`` tuples where *bbox* is a dict with
        keys ``x``, ``y``, ``width``, ``height`` (pixel coordinates).
        Each tuple represents one detected line of text.
        """
        with open(image_path, "rb") as f:
            content = f.read()

        encoded_image = base64.b64encode(content).decode("utf-8")

        url = self.vision_api_url
        if self.api_key:
            url += f"?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
        else:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

        request_body = {
            "requests": [
                {
                    "image": {"content": encoded_image},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                }
            ]
        }

        response = requests.post(url, json=request_body, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Google Vision API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        if "responses" not in result or not result["responses"]:
            return []

        response_data = result["responses"][0]
        text_annotations = response_data.get("textAnnotations", [])

        if not text_annotations:
            return []

        print(
            f"Google Vision OCR found {len(text_annotations) - 1} text regions "
            f"in {os.path.basename(image_path)}"
        )
        return self._group_into_lines(text_annotations[1:])

    def _group_into_lines(self, word_annotations, y_threshold=8, x_threshold=100):
        """Group individual word annotations into lines of text."""
        words = []
        for annotation in word_annotations:
            text = annotation["description"]
            vertices = annotation["boundingPoly"]["vertices"]
            x_coords = [v.get("x", 0) for v in vertices]
            y_coords = [v.get("y", 0) for v in vertices]
            bbox = {
                "x": min(x_coords),
                "y": min(y_coords),
                "width": max(x_coords) - min(x_coords),
                "height": max(y_coords) - min(y_coords),
            }
            words.append((text, bbox))

        if not words:
            return []

        # Sort by y-coordinate first
        words.sort(key=lambda w: w[1]["y"])

        # Group into rows by y proximity
        lines = []
        current_line = [words[0]]
        for word, bbox in words[1:]:
            if abs(bbox["y"] - current_line[-1][1]["y"]) <= y_threshold:
                current_line.append((word, bbox))
            else:
                lines.append(current_line)
                current_line = [(word, bbox)]
        lines.append(current_line)

        # Split lines at large horizontal gaps
        final_lines = []
        for line in lines:
            if len(line) <= 1:
                final_lines.append(line)
                continue
            line.sort(key=lambda w: w[1]["x"])
            current_subline = [line[0]]
            for word_bbox in line[1:]:
                word, bbox = word_bbox
                prev_bbox = current_subline[-1][1]
                gap = bbox["x"] - (prev_bbox["x"] + prev_bbox["width"])
                if gap > x_threshold:
                    final_lines.append(current_subline)
                    current_subline = [word_bbox]
                else:
                    current_subline.append(word_bbox)
            final_lines.append(current_subline)

        # Build (text, bbox) pairs for each line
        text_regions = []
        for line in final_lines:
            line.sort(key=lambda w: w[1]["x"])
            line_text = " ".join(word for word, _ in line)
            all_x = [b["x"] for _, b in line] + [b["x"] + b["width"] for _, b in line]
            all_y = [b["y"] for _, b in line] + [b["y"] + b["height"] for _, b in line]
            line_bbox = {
                "x": min(all_x),
                "y": min(all_y),
                "width": max(all_x) - min(all_x),
                "height": max(all_y) - min(all_y),
            }
            text_regions.append((line_text, line_bbox))

        print(f"Grouped into {len(text_regions)} lines")
        return text_regions
