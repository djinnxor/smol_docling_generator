import numpy as np
import random
# Document themes
_THEMES = {
    "light":      {"paper": (250, 255), "warm_bias": 0,  "accent": (0, 0, 0),    "border": (100, 100, 100)},
    "warm":       {"paper": (245, 252), "warm_bias": 8,  "accent": (100, 50, 0),  "border": (150, 100, 50)},
    "blue_tint":  {"paper": (240, 250), "warm_bias": -5, "accent": (0, 0, 120),   "border": (70, 70, 140)},
    "aged":       {"paper": (230, 245), "warm_bias": 15, "accent": (80, 60, 30),  "border": (120, 100, 60)},
    "green_tint": {"paper": (242, 252), "warm_bias": -3, "accent": (0, 80, 0),    "border": (60, 120, 60)},
    "gray":       {"paper": (235, 248), "warm_bias": 0,  "accent": (50, 50, 50),  "border": (80, 80, 80)},
    "cream":      {"paper": (240, 250), "warm_bias": 12, "accent": (90, 50, 10),  "border": (140, 110, 70)},
}

# Theme augmentation
def apply_theme(img_np: np.ndarray, rng: random.Random) -> np.ndarray:
    theme_name = rng.choice(list(_THEMES.keys()))
    theme = _THEMES[theme_name]
    # Paper color
    paper_lo, paper_hi = theme["paper"]
    paper_val = rng.randint(paper_lo, paper_hi)
    # Add paper color as a bias to the image background
    img = img_np.astype(np.float32)
    # Estimate background as the 95th percentile per channel
    bg = np.percentile(img, 95, axis=(0,1))
    bias = paper_val - bg
    img = np.clip(img + bias, 0, 255)
    # Warm/cool bias
    warm_bias = theme["warm_bias"]
    if warm_bias != 0:
        img[..., 0] = np.clip(img[..., 0] + warm_bias, 0, 255)  # R
        img[..., 2] = np.clip(img[..., 2] - warm_bias, 0, 255)  # B
    return img.astype(np.uint8)

import os
import argparse
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import shutil
import cv2

# Augmentation functions

def color_jitter(img):
    enhancers = [
        (ImageEnhance.Brightness, 0.7, 1.3),
        (ImageEnhance.Contrast, 0.7, 1.3),
        (ImageEnhance.Color, 0.7, 1.3),
        (ImageEnhance.Sharpness, 0.8, 1.2),
    ]
    for enhancer, low, high in enhancers:
        if random.random() < 0.7:
            factor = random.uniform(low, high)
            img = enhancer(img).enhance(factor)
    return img

def gaussian_blur(img):
    if random.random() < 0.5:
        radius = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img

def add_gaussian_noise(img):
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 8, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img

def jpeg_compression(img):
    if random.random() < 0.5:
        from io import BytesIO
        buf = BytesIO()
        quality = random.randint(40, 85)
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf)
    return img

def random_gamma(img):
    if random.random() < 0.5:
        gamma = random.uniform(0.8, 1.2)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img

AUGS = [color_jitter, gaussian_blur, add_gaussian_noise, jpeg_compression, random_gamma]


# --- Advanced augmentations using numpy/cv2 ---
def _make_np_rng(rng):
    # Make a numpy RandomState from a Python random.Random
    return np.random.RandomState(rng.randint(0, 2**31-1))

def apply_nonlinear_illumination(img_np: np.ndarray, rng: random.Random) -> np.ndarray:
    np_rng = _make_np_rng(rng)
    h, w = img_np.shape[:2]
    cx = rng.uniform(0.2, 0.8) * w
    cy = rng.uniform(0.2, 0.8) * h
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2) / 2 + 1e-6
    vignette = 1.0 - rng.uniform(0.02, 0.08) * (dist / max_dist) ** 2
    direction = rng.choice(["horizontal", "vertical", "diagonal", "radial"])
    if direction == "horizontal":
        grad = np.linspace(rng.uniform(0.95, 0.99), rng.uniform(1.01, 1.05), w, dtype=np.float32)
        gradient = np.tile(grad, (h, 1))
    elif direction == "vertical":
        grad = np.linspace(rng.uniform(0.95, 0.99), rng.uniform(1.01, 1.05), h, dtype=np.float32)
        gradient = np.tile(grad[:, None], (1, w))
    elif direction == "diagonal":
        gx = np.linspace(0, 1, w, dtype=np.float32)
        gy = np.linspace(0, 1, h, dtype=np.float32)
        gradient = 0.5 * (np.tile(gx, (h, 1)) + np.tile(gy[:, None], (1, w)))
        gradient = rng.uniform(0.96, 0.99) + gradient * rng.uniform(0.02, 0.06)
    else:
        bx = rng.uniform(0.1, 0.9) * w
        by = rng.uniform(0.1, 0.9) * h
        d2 = np.sqrt((X - bx) ** 2 + (Y - by) ** 2)
        gradient = 1.0 + rng.uniform(0.01, 0.04) * np.exp(-(d2 / (max_dist * 0.5)) ** 2)
    combined = np.clip(vignette * gradient, 0.90, 1.10).astype(np.float32)
    combined += np_rng.normal(0, rng.uniform(0.002, 0.006), combined.shape).astype(np.float32)
    return np.clip(img_np.astype(np.float32) * combined[..., None], 0, 255).astype(np.uint8)

def apply_random_shadows(img_np: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img_np.shape[:2]
    result = img_np.astype(np.float32)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(rng.randint(1, 5)):
        sx, sy = rng.randint(0, w), rng.randint(0, h)
        sig_x = rng.randint(w // 5, w)
        sig_y = rng.randint(h // 5, h)
        blob = np.exp(-((X - sx) ** 2 / (2 * sig_x ** 2) + (Y - sy) ** 2 / (2 * sig_y ** 2)))
        darkness = rng.uniform(0.01, 0.07)
        result *= (1.0 - darkness * blob)[..., None]
    # Edge shadow
    if rng.random() < 0.3:
        edge = rng.choice(["left", "right", "top", "bottom"])
        qw, qh = w // 4, h // 4
        if edge == "left":
            g = np.pad(np.linspace(rng.uniform(0.90, 0.96), 1.0, qw, dtype=np.float32),
                       (0, w - qw), constant_values=1.0)
            result *= np.tile(g, (h, 1))[..., None]
        elif edge == "right":
            g = np.pad(np.linspace(1.0, rng.uniform(0.90, 0.96), qw, dtype=np.float32),
                       (w - qw, 0), constant_values=1.0)
            result *= np.tile(g, (h, 1))[..., None]
        elif edge == "top":
            g = np.pad(np.linspace(rng.uniform(0.92, 0.97), 1.0, qh, dtype=np.float32),
                       (0, h - qh), constant_values=1.0)
            result *= np.tile(g[:, None], (1, w))[..., None]
        else:
            g = np.pad(np.linspace(1.0, rng.uniform(0.92, 0.97), qh, dtype=np.float32),
                       (h - qh, 0), constant_values=1.0)
            result *= np.tile(g[:, None], (1, w))[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_scan_noise(img_np: np.ndarray, rng: random.Random) -> np.ndarray:
    np_rng = _make_np_rng(rng)
    img = img_np.astype(np.float32)
    sigma = rng.uniform(1.0, 4.0)
    img += np_rng.normal(0.0, sigma, img.shape).astype(np.float32)
    sp = rng.uniform(0.0001, 0.0008)
    ra = np_rng.random(img.shape[:2])
    img[ra < sp] = 255.0
    img[(ra >= sp) & (ra < 2 * sp)] = 0.0
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.5:
            x = rng.randint(0, img.shape[1] - 1)
            img[:, x:x + rng.randint(1, 2)] += rng.uniform(-30, 30)
        else:
            y = rng.randint(0, img.shape[0] - 1)
            img[y:y + rng.randint(1, 2), :] += rng.uniform(-30, 30)
    for _ in range(rng.randint(3, 25)):
        dy, dx = rng.randint(0, img.shape[0] - 1), rng.randint(0, img.shape[1] - 1)
        r = rng.randint(0, 2)
        y1, y2 = max(0, dy - r), min(img.shape[0], dy + r + 1)
        x1, x2 = max(0, dx - r), min(img.shape[1], dx + r + 1)
        img[y1:y2, x1:x2] = rng.choice([0, 255])
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (rng.choice([3, 5]),) * 2, 0)
    alpha = rng.uniform(0.93, 1.07)
    beta = rng.uniform(-5.0, 5.0)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def apply_random_lines(img_np: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img_np.shape[:2]
    result = img_np.copy()
    for _ in range(rng.randint(0, 4)):
        cv = rng.randint(190, 240)
        if rng.random() < 0.5:
            y = rng.randint(0, h)
            cv2.line(result, (0, y + rng.randint(-20, 20)),
                     (w, y + rng.randint(-20, 20)), (cv, cv, cv), rng.randint(1, 2))
        else:
            x = rng.randint(0, w)
            cv2.line(result, (x + rng.randint(-20, 20), 0),
                     (x + rng.randint(-20, 20), h), (cv, cv, cv), rng.randint(1, 2))
    return result

# List of all augmentations
ADV_AUGS = [apply_nonlinear_illumination, apply_random_shadows, apply_scan_noise, apply_random_lines]

def augment_image(img, rng=None):
    out = img.copy()
    for aug in AUGS:
        out = aug(out)
    # Convert to numpy for advanced augs
    img_np = np.array(out)
    if rng is None:
        rng = random.Random()
    # Apply theme first
    img_np = apply_theme(img_np, rng)
    for adv_aug in ADV_AUGS:
        if rng.random() < 0.7:
            img_np = adv_aug(img_np, rng)
    out = Image.fromarray(img_np)
    return out


def main():
    parser = argparse.ArgumentParser(description="Augment images in uploads folder (no rotate/flip/crop)")
    parser.add_argument('--input', type=str, default='/home/rookie/Documents/Minje/trad_back_docling/worker/labeler_data/uploads', help='Input uploads folder')
    parser.add_argument('--xml', type=str, default='/home/rookie/Documents/Minje/trad_back_docling/worker/labeler_data/xml_annotations', help='Input xml_annotations folder')
    parser.add_argument('--output', type=str, default='augmented_uploads', help='Output folder for augmented images')
    parser.add_argument('--n', type=int, default=2, help='Number of augmentations per image')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    uploads_dir = os.path.join(args.output, 'uploads')
    ann_dir = os.path.join(args.output, 'annotations')
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    images = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    print(f"Found {len(images)} images.")
    rng = random.Random(42)
    for fname in images:
        img_path = os.path.join(args.input, fname)
        img = Image.open(img_path).convert('RGB')
        base, ext = os.path.splitext(fname)
        xml_name = base + '_annotations.xml'
        xml_src = os.path.join(args.xml, xml_name)
        for i in range(args.n):
            aug_img = augment_image(img, rng)
            out_name = f"{base}_aug{i+1}{ext}"
            out_path = os.path.join(uploads_dir, out_name)
            aug_img.save(out_path)
            # Save annotation with same name as image but .xml
            ann_out = os.path.join(ann_dir, f"{base}_aug{i+1}.xml")
            if os.path.exists(xml_src):
                shutil.copy(xml_src, ann_out)
                print(f"Saved {out_path} and annotation {ann_out}")
            else:
                print(f"Saved {out_path} (no annotation found)")

if __name__ == "__main__":
    main()
