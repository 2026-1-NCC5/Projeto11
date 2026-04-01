import os
import cv2
import numpy as np

# Caminhos base
BASE_DIRS = {
    "train": {"images": "dataset/train/images", "labels": "dataset/train/labels"},
    "valid": {"images": "dataset/valid/images", "labels": "dataset/valid/labels"}
}

CLASS_MAP = {
    "arroz": 0,
    "feijao": 1,
    "acucar": 2,
    "macarrao":3,
    "oleo":4,
    "fuba":5,

}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

def normalize_bbox(x, y, w, h, img_w, img_h):
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h

def largest_contour_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500: return None
    return cv2.boundingRect(contour)

def try_threshold_fallback(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(cv2.dilate(edges, kernel, iterations=2), cv2.MORPH_CLOSE, kernel)

    candidates = [b for b in [largest_contour_bbox(thresh1), largest_contour_bbox(thresh2), largest_contour_bbox(edges)] if b is not None]
    return max(candidates, key=lambda b: b[2] * b[3]) if candidates else None

def generate_label_for_image(img_path, label_path, class_id):
    image = cv2.imread(img_path)
    if image is None: return False

    img_h, img_w = image.shape[:2]
    bbox = try_threshold_fallback(image)

    # Lógica de Fallback se a detecção falhar (cria uma caixa no centro da imagem)
    if bbox is None:
        x_center, y_center, width, height = 0.5, 0.5, 0.8, 0.8
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        return True

    x, y, w, h = bbox
    area_ratio = (w * h) / (img_w * img_h)

    if area_ratio < 0.02 or area_ratio > 0.98:
        x_center, y_center, width, height = 0.5, 0.5, 0.8, 0.8
    else:
        # Adiciona padding e normaliza
        pad_w, pad_h = int(w * 0.03), int(h * 0.03)
        x, y = max(0, x - pad_w), max(0, y - pad_h)
        w, h = min(img_w - x, w + 2 * pad_w), min(img_h - y, h + 2 * pad_h)
        x_center, y_center, width, height = normalize_bbox(x, y, w, h, img_w, img_h)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return True

def main():
    total, success = 0, 0

    for split_name, paths in BASE_DIRS.items():
        img_dir, lbl_dir = paths["images"], paths["labels"]
        if not os.path.exists(img_dir): continue

        for category in sorted(os.listdir(img_dir)):
            if category not in CLASS_MAP: continue
            
            category_path = os.path.join(img_dir, category)
            label_category_path = os.path.join(lbl_dir, category)
            os.makedirs(label_category_path, exist_ok=True)

            for file_name in sorted(os.listdir(category_path)):
                if not file_name.lower().endswith(VALID_EXTENSIONS): continue
                total += 1
                img_path = os.path.join(category_path, file_name)
                label_path = os.path.join(label_category_path, os.path.splitext(file_name)[0] + ".txt")

                if generate_label_for_image(img_path, label_path, CLASS_MAP[category]):
                    success += 1

    print(f"\nLabels gerados com sucesso: {success}/{total}")

if __name__ == "__main__":
    main()