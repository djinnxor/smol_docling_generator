import os
import shutil
import base64
import requests
from difflib import SequenceMatcher
from PIL import Image, ImageDraw
import json
import torch
import csv


class OCRComparator:
    def __init__(self, input_folder, output_folder_match, output_folder_diff, google_credentials_path, api_key=None, config_path='mn.yaml', model_path='mn.pth', save_crops=True, save_csv=True, mismatches_and_matches=True):
        self.input_folder = input_folder
        self.output_folder_match = output_folder_match
        self.output_folder_diff = output_folder_diff
        self.google_credentials_path = google_credentials_path
        self.config_path = config_path
        self.model_path = model_path
        self.api_key = api_key
        self.save_crops = save_crops
        self.save_csv = save_csv
        self.mismatches_and_matches = mismatches_and_matches
        
        os.makedirs(output_folder_match, exist_ok=True)
        os.makedirs(output_folder_diff, exist_ok=True)
        if save_crops:
            os.makedirs("dataset", exist_ok=True)

        self.vision_api_url = "https://vision.googleapis.com/v1/images:annotate"
        #self.access_token = self._get_access_token()
        self._load_custom_model()
    
    def _get_access_token(self):
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        credentials = service_account.Credentials.from_service_account_file(
            self.google_credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-vision']
        )

        credentials.refresh(Request())
        return credentials.token
    
    def _load_custom_model(self):
        import torch
        import yaml
        import torchvision.transforms as transforms

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        from mn import Model

        num_class = len(config['character_list']) + 1
        self.custom_model = Model(
            input_channel=config['network_params']['input_channel'],
            output_channel=config['network_params']['output_channel'],
            hidden_size=config['network_params']['hidden_size'],
            num_class=num_class
        )

        state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value

        self.custom_model.load_state_dict(new_state_dict)
        self.custom_model.eval()

        self.character_list = ['[blank]'] + list(config['character_list'])
        self.img_height = config['imgH']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def google_vision_ocr(self, image_path):
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        encoded_image = base64.b64encode(content).decode('utf-8')
        if self.api_key:
            self.vision_api_url += f"?key={self.api_key}"
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

        response = requests.post(self.vision_api_url, json=request_body, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if 'responses' in result and len(result['responses']) > 0:
                response_data = result['responses'][0]
                text_annotations = response_data.get('textAnnotations', [])
                print(f"Google Vision OCR found {len(text_annotations)-1} text regions in {os.path.basename(image_path)}")

                # Collect all word annotations with their bboxes
                words = []
                for annotation in text_annotations[1:]:
                    text = annotation['description']
                    vertices = annotation['boundingPoly']['vertices']
                    x_coords = [v.get('x', 0) for v in vertices]
                    y_coords = [v.get('y', 0) for v in vertices]
                    bbox = {
                        'x': min(x_coords),
                        'y': min(y_coords),
                        'width': max(x_coords) - min(x_coords),
                        'height': max(y_coords) - min(y_coords)
                    }
                    words.append((text, bbox))

                if words:
                    # Sort words by y-coordinate
                    words.sort(key=lambda w: w[1]['y'])

                    lines = []
                    current_line = [words[0]]
                    line_threshold = 8 

                    for word, bbox in words[1:]:
                        prev_bbox = current_line[-1][1]
                        if abs(bbox['y'] - prev_bbox['y']) <= line_threshold:
                            current_line.append((word, bbox))
                        else:
                            lines.append(current_line)
                            current_line = [(word, bbox)]
                    lines.append(current_line)

                    #splitting lines based on theri x gaps
                    final_lines = []
                    x_threshold = 100  
                    for line in lines:
                        if len(line) <= 1:
                            final_lines.append(line)
                            continue
                        # Sort by x first
                        line.sort(key=lambda w: w[1]['x'])
                        current_subline = [line[0]]
                        for word_bbox in line[1:]:
                            word, bbox = word_bbox
                            prev_bbox = current_subline[-1][1]
                            gap = bbox['x'] - (prev_bbox['x'] + prev_bbox['width'])
                            if gap > x_threshold:
                                final_lines.append(current_subline)
                                current_subline = [word_bbox]
                            else:
                                current_subline.append(word_bbox)
                        final_lines.append(current_subline)
                    lines = final_lines

                    text_regions = []
                    for line in lines:
                        # Sort words in line by x-coordinate
                        line.sort(key=lambda w: w[1]['x'])
                        line_text = ' '.join(word for word, _ in line)
                        print(f"Detected line: {line_text}")

                        # Compute overall bbox for the line
                        all_x = [bbox['x'] for _, bbox in line] + [bbox['x'] + bbox['width'] for _, bbox in line]
                        all_y = [bbox['y'] for _, bbox in line] + [bbox['y'] + bbox['height'] for _, bbox in line]
                        line_bbox = {
                            'x': min(all_x),
                            'y': min(all_y),
                            'width': max(all_x) - min(all_x),
                            'height': max(all_y) - min(all_y)
                        }
                        text_regions.append((line_text, line_bbox))

                    print(f"Grouped into {len(text_regions)} lines")
                else:
                    text_regions = []

                return text_regions
            return []
        else:
            raise Exception(f"Google Vision API error: {response.status_code} - {response.text}")
    
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

    def compare_and_save(self, image_path, threshold=0.95):
        filename = os.path.basename(image_path)

        try:
            text_regions = self.google_vision_ocr(image_path)

            perfect_matches = []
            mismatches = []

            full_image = Image.open(image_path).convert('L')
            img_width, img_height = full_image.size

            crops_data = []
            mismatches_data = []
            matched_data = []

            for idx, (google_word, bbox) in enumerate(text_regions):
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

                padding = 5
                x_crop = max(0, x - padding)
                y_crop = max(0, y - padding)
                w_crop = w + 2 * padding
                h_crop = h + 2 * padding

                cropped = full_image.crop((x_crop, y_crop, x_crop + w_crop, y_crop + h_crop))

                if self.save_crops:
                    # Save crop
                    crop_filename = f"{os.path.splitext(filename)[0]}_region_{idx}.png"
                    crop_path = os.path.join("dataset", crop_filename)
                    cropped.save(crop_path)
                else:
                    crop_filename = f"{os.path.splitext(filename)[0]}_region_{idx}.png"  # Still generate for CSV

                crop_w, crop_h = cropped.size
                if crop_h == 0:
                    continue

                aspect_ratio = crop_w / crop_h
                new_w = int(self.img_height * aspect_ratio)
                resized = cropped.resize((new_w, self.img_height), Image.LANCZOS)

                image_tensor = self.transform(resized).unsqueeze(0)

                with torch.no_grad():
                    preds = self.custom_model(image_tensor, None)

                _, preds_index = preds.max(2)
                preds_index = preds_index.squeeze(0)

                result = []
                prev_char = None
                for char_idx_tensor in preds_index:
                    char_idx = char_idx_tensor.item()
                    if char_idx != 0 and char_idx != prev_char:
                        result.append(self.character_list[char_idx])
                    prev_char = char_idx

                custom_word = ''.join(result).strip()

                similarity = self.calculate_similarity(google_word, custom_word)

                comparison_data = {
                    'index': idx,
                    'bbox': bbox,
                    'google': google_word,
                    'custom': custom_word,
                    'similarity': similarity,
                    'match': similarity >= threshold
                }

                if self.save_csv:
                    crops_data.append({
                        'crop_filename': crop_filename,
                        'google_word': google_word
                    })

                # os.makedirs("matches/matched", exist_ok=True)
                # os.makedirs("matches/mismatched", exist_ok=True)

                if comparison_data['match']:
                    #matched_path = os.path.join("matches", "matched", crop_filename)
                    #cropped.save(matched_path)
                    matched_data.append({
                        'crop_filename': crop_filename,
                        'custom_word': custom_word,
                        'google_word': google_word
                    })
                else:
                    #mismatched_path = os.path.join("matches", "mismatched", crop_filename)
                    #cropped.save(mismatched_path)
                    mismatches_data.append({
                        'crop_filename': crop_filename,
                        'custom_word': custom_word,
                        'google_word': google_word
                    })

                if similarity >= 0.95:
                    perfect_matches.append(comparison_data)
                else:
                    mismatches.append(comparison_data)

            all_comparisons = perfect_matches + mismatches

            if self.save_csv and crops_data:
                # Save crops data to CSV
                csv_path = "labels.csv"
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['crop_filename', 'google_word']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    for data in crops_data:
                        writer.writerow(data)

            if len(all_comparisons) > 0:
                avg_similarity = sum(r['similarity'] for r in all_comparisons) / len(all_comparisons)
                matching_regions = sum(1 for r in all_comparisons if r['match'])
                total_regions = len(all_comparisons)
                match_rate = (matching_regions / total_regions) * 100
            else:
                avg_similarity = 0
                matching_regions = 0
                total_regions = 0
                match_rate = 0

            if match_rate >= (threshold * 100):
                dest_folder = self.output_folder_match
                status = "MATCH"
            else:
                dest_folder = self.output_folder_diff
                status = "DIFF"

            dest_path = os.path.join(dest_folder, filename)
            shutil.copy2(image_path, dest_path)

            # Draw bounding boxes on the copied image
            # img = Image.open(dest_path)
            # draw = ImageDraw.Draw(img)
            # for comp in all_comparisons:
            #     bbox = comp['bbox']
            #     x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            #     color = 'green' if comp['similarity'] == 1.0 else 'red'
            #     draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            # img.save(dest_path)

            txt_path = os.path.join(dest_folder, f"{os.path.splitext(filename)[0]}_comparison.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== OCR COMPARISON REPORT ===\n")
                f.write(f"Image: {filename}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Overall Match Rate: {match_rate:.2f}% ({matching_regions}/{total_regions} regions)\n")
                f.write(f"Average Similarity: {avg_similarity:.2%}\n")
                f.write(f"Perfect Matches (100%): {len(perfect_matches)}\n")
                f.write(f"Mismatches (<100%): {len(mismatches)}\n")
                f.write(f"\n=== PER-REGION COMPARISON ===\n\n")

                for comp in all_comparisons:
                    match_symbol = "[MATCH]" if comp['match'] else "[DIFF]"
                    f.write(f"Region {comp['index'] + 1}: {match_symbol} (Similarity: {comp['similarity']:.2%})\n")
                    f.write(f"  Google:  '{comp['google']}'\n")
                    f.write(f"  Custom:  '{comp['custom']}'\n")
                    if not comp['match']:
                        f.write(f"  ** MISMATCH **\n")
                    f.write(f"\n")

                if mismatches:
                    f.write(f"\n=== MISMATCHED REGIONS SUMMARY ({len(mismatches)} regions) ===\n\n")
                    for comp in mismatches:
                        f.write(f"Region {comp['index'] + 1}: '{comp['google']}' -> '{comp['custom']}' (Similarity: {comp['similarity']:.2%})\n")
            
            #self._save_labelstudio_json(image_path, perfect_matches, dest_folder,f"{os.path.splitext(filename)[0]}_perfect_matches.json")
            #self._save_labelstudio_json(image_path, mismatches, dest_folder,f"{os.path.splitext(filename)[0]}_mismatches.json")

            print(f"{filename}: {status} - Match Rate: {match_rate:.2f}% ({matching_regions}/{total_regions}), "
                  f"Perfect: {len(perfect_matches)}, Mismatches: {len(mismatches)}")

            return {
                'filename': filename,
                'perfect_matches': perfect_matches,
                'mismatches': mismatches,
                'crops_data': crops_data,
                'mismatches_data': mismatches_data,
                'matched_data': matched_data
            }

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return None
    
        t

if __name__ == "__main__":
    INPUT_FOLDER = "input_folder"
    OUTPUT_MATCH = "output_matching"
    OUTPUT_DIFF = "output_different"
    GOOGLE_CREDENTIALS = "/home/rookie2/Documents/code/all_ml/ocr_gcv_pipeline/key.json"
    api_key = os.environ.get("GOOGLE_OCR_API_KEY", "")

    comparator = OCRComparator(INPUT_FOLDER, OUTPUT_MATCH, OUTPUT_DIFF,
                                api_key=api_key,
                               google_credentials_path=GOOGLE_CREDENTIALS,
                               model_path="mn.pth",
                               config_path="mn.yaml",
                               save_crops=False,
                               save_csv=False, 
                               mismatches_and_matches=False)
    comparator.process_folder(threshold=0.95)