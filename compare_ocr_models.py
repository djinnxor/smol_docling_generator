import torch
from pathlib import Path
from PIL import Image
from mn import Model
import torch
import yaml
import easyocr

with open("custom_models/mn.yaml", 'r') as f:
    config = yaml.safe_load(f)

def load_model(model_path):
    num_class = len(config['character_list']) + 1
    custom_model = Model(
            input_channel=config['network_params']['input_channel'],
            output_channel=config['network_params']['output_channel'],
            hidden_size=config['network_params']['hidden_size'],
            num_class=num_class
        )
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    custom_model.load_state_dict(new_state_dict)
    custom_model.eval()
    
    return custom_model

def run_ocr(model, image_path):
    image = Image.open(image_path).convert("L")
    # Assuming model expects a tensor input; adapt preprocessing as needed
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor, None)
    # Adapt output parsing as needed for your OCR model

    _, preds_index = output.max(2)
    preds_index = preds_index.squeeze(0)
    
    result = []
    prev_char = None
    for char_idx_tensor in preds_index:
        char_idx = char_idx_tensor.item()
        if char_idx != 0 and char_idx != prev_char:
            result.append(config['character_list'][char_idx])
        prev_char = char_idx

    custom_word = ''.join(result).strip()
    
    return custom_word

def compare_ocr_models(model_folder, image_path):
    model_folder = Path(model_folder)
    image_path = Path(image_path)
    results = {}
    for model_file in model_folder.glob("*.pth"):
        model = load_model(model_file)
        output = run_ocr(model, image_path)
        results[model_file.name] = output
    return results

# Example usage:
# results = compare_ocr_models("custom_models", "60/task_6313.png")
# for model_name, output in results.items():
#     print(f"{model_name}: {output}")

model = load_model("custom_models/mn.pth")
output = run_ocr(model, "60/task_6313.png")
print(output)

# Example using EasyOCR
def run_easyocr(image_path, lang_list=['en']):
    reader = easyocr.Reader(lang_list, gpu=False)
    results = reader.readtext(image_path)
    for bbox, text, conf in results:
        print(f"Text: {text}, Confidence: {conf}")

# Example usage:
# run_easyocr("60/task_6313.png")

# Example using RapidOCR (if installed)
# import rapidocr_onnxruntime as rapidocr
# def run_rapidocr(image_path):
#     ocr = rapidocr.RapidOCR()
#     result, _ = ocr(image_path)
#     for item in result:
#         print(f"Text: {item[1]}, Confidence: {item[2]}")