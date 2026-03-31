import os
import shutil
import random

IMAGE_DIR = "dataset/train/images"
VAL_DIR = "dataset/valid/images" 

SPLIT_RATIO = 0.2

for category in os.listdir(IMAGE_DIR):
    category_path = os.path.join(IMAGE_DIR, category)
    val_category_path = os.path.join(VAL_DIR, category)
    
    if not os.path.isdir(category_path): continue
    os.makedirs(val_category_path, exist_ok=True)

    images = [f for f in os.listdir(category_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    split_size = int(len(images) * SPLIT_RATIO)
    val_images = images[:split_size]

    for img in val_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(val_category_path, img)
        shutil.move(src, dst)

print("Divisão 80/20 (train/valid) concluída com sucesso.")