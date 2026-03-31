import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "dataset_base"
OUTPUT_DIR = "dataset/train/images" 

IMAGES_PER_INPUT = 10
TARGET_SIZE = (640, 640)

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_rotate(image):
    angle = np.random.uniform(-20, 20)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def random_flip(image):
    flip_code = np.random.choice([-1, 0, 1])
    return cv2.flip(image, flip_code)

def random_blur(image):
    k = np.random.choice([3, 5])
    return cv2.GaussianBlur(image, (k, k), 0)

def random_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def random_zoom(image):
    h, w = image.shape[:2]
    scale = np.random.uniform(0.85, 1.15)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    if scale > 1.0:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        canvas = np.zeros_like(image)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return canvas

def augment_image(image):
    aug = image.copy()
    if np.random.rand() < 0.7: aug = random_rotate(aug)
    if np.random.rand() < 0.5: aug = random_flip(aug)
    if np.random.rand() < 0.5:
        alpha, beta = np.random.uniform(0.8, 1.2), np.random.randint(-30, 30)
        aug = adjust_brightness_contrast(aug, alpha, beta)
    if np.random.rand() < 0.3: aug = random_blur(aug)
    if np.random.rand() < 0.3: aug = random_noise(aug)
    if np.random.rand() < 0.5: aug = random_zoom(aug)
    return cv2.resize(aug, TARGET_SIZE)

# Processamento Principal
os.makedirs(OUTPUT_DIR, exist_ok=True)

for category in os.listdir(INPUT_DIR):
    input_category = os.path.join(INPUT_DIR, category)
    output_category = os.path.join(OUTPUT_DIR, category)
    
    if not os.path.isdir(input_category): continue
    os.makedirs(output_category, exist_ok=True)

    images = [f for f in os.listdir(input_category) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_name in tqdm(images, desc=f"Processando {category}"):
        img_path = os.path.join(input_category, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        image = cv2.resize(image, TARGET_SIZE)
        base_name = os.path.splitext(img_name)[0]

        cv2.imwrite(os.path.join(output_category, f"{base_name}_orig.jpg"), image)

        for i in range(IMAGES_PER_INPUT):
            augmented = augment_image(image)
            cv2.imwrite(os.path.join(output_category, f"{base_name}_aug_{i}.jpg"), augmented)

print("Dataset aumentado criado com sucesso.")