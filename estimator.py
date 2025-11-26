import cv2
import numpy as np
import pandas as pd
from pathlib import Path

density_mapping = {
    "apple": 0.78, "banana": 0.91, "bread": 0.18, "bun": 0.34, "doughnut": 0.31,
    "egg": 1.03, "fired_dough_twist": 0.58, "grape": 0.97, "lemon": 0.96, "litchi": 1.00,
    "mango": 1.07, "mooncake": 0.96, "orange": 0.90, "pear": 1.02, "peach": 0.96,
    "plum": 1.01, "qiwi": 0.97, "sachima": 0.22, "tomato": 0.98
}

energy_mapping = {
    "apple": 0.52, "banana": 0.89, "bread": 3.15, "bun": 2.23, "doughnut": 4.34,
    "egg": 1.43, "fired_dough_twist": 24.16, "grape": 0.69, "lemon": 0.29, "litchi": 0.66,
    "mango": 0.60, "mooncake": 18.83, "orange": 0.63, "pear": 0.39, "peach": 0.57,
    "plum": 0.46, "qiwi": 0.61, "sachima": 21.45, "tomato": 0.27
}

shape_mapping = {
    "apple": "ellipsoid", "banana": "irregular", "bread": "column", "bun": "irregular",
    "doughnut": "irregular", "egg": "ellipsoid", "fired_dough_twist": "irregular",
    "grape": "irregular", "lemon": "ellipsoid", "mango": "irregular", "litchi": "irregular",
    "mooncake": "column", "orange": "ellipsoid", "pear": "irregular", "peach": "ellipsoid",
    "plum": "ellipsoid", "qiwi": "ellipsoid", "sachima": "column", "tomato": "ellipsoid"
}

class_names = [
    "apple", "banana", "bread", "bun", "doughnut", "egg", "fired_dough_twist",
    "grape", "lemon", "litchi", "mango", "mooncake", "orange", "pear", "peach",
    "plum", "qiwi", "sachima", "tomato", "coin"
]

def load_yolo_predictions(pred_file, class_names):
    bboxes = []
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_idx = int(parts[0])
            bbox = tuple(map(float, parts[1:5]))
            bboxes.append({'class': class_names[class_idx], 'bbox': bbox})
    return bboxes

def crop_from_image(image_path, bbox):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    x_center, y_center, width_box, height_box = bbox
    x_center *= w
    y_center *= h
    width_box *= w
    height_box *= h
    x1 = int(x_center - width_box / 2)
    y1 = int(y_center - height_box / 2)
    x2 = int(x_center + width_box / 2)
    y2 = int(y_center + height_box / 2)
    return image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def grabcut_segmentation(crop_img, iter_count=5):
    mask = np.zeros(crop_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w = crop_img.shape[:2]
    rect = (1, 1, w-2, h-2)
    cv2.grabCut(crop_img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return crop_img * mask2[:, :, np.newaxis], mask2

def compute_alpha(crop_coin):
    real_diameter_mm = 25
    h, w = crop_coin.shape[:2]
    return real_diameter_mm / max(h, w)

def estimate_volume(mask_side, alpha_side, alpha_top, shape_type):
    beta = 1.0
    H_s, W_s = mask_side.shape
    Lk = np.sum(mask_side == 1, axis=1)
    L_max = np.max(Lk)
    if shape_type == "ellipsoid":
        return beta * (np.pi / 4) * (alpha_side ** 3) * np.sum(Lk ** 2)
    elif shape_type == "column":
        area = np.sum(mask_side == 1)
        return beta * (alpha_top ** 2) * (alpha_side * H_s) * area
    elif shape_type == "irregular":
        area = np.sum(mask_side == 1)
        return beta * (alpha_top ** 2) * (alpha_side) * np.sum((Lk / L_max) ** 2) * area
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

def estimate_mass_and_calories(volume_mm3, food_class, density_mapping, energy_mapping):
    volume_cm3 = volume_mm3 / 1000
    mass = volume_cm3 * density_mapping.get(food_class, 1.0)
    energy = energy_mapping.get(food_class, 1.0)
    return mass, mass * energy

def process_single_image(top_image_path, side_image_path, top_label_path, side_label_path,
                         class_names, density_mapping, energy_mapping):
    top_preds = load_yolo_predictions(top_label_path, class_names)
    side_preds = load_yolo_predictions(side_label_path, class_names)

    food_bbox_top = next((p['bbox'] for p in top_preds if p['class'] != 'coin'), None)
    coin_bbox_top = next((p['bbox'] for p in top_preds if p['class'] == 'coin'), None)
    food_bbox_side = next((p['bbox'] for p in side_preds if p['class'] != 'coin'), None)
    coin_bbox_side = next((p['bbox'] for p in side_preds if p['class'] == 'coin'), None)

    if not all([food_bbox_top, coin_bbox_top, food_bbox_side, coin_bbox_side]):
        return None

    food_top = crop_from_image(top_image_path, food_bbox_top)
    coin_top = crop_from_image(top_image_path, coin_bbox_top)
    food_side = crop_from_image(side_image_path, food_bbox_side)
    coin_side = crop_from_image(side_image_path, coin_bbox_side)

    _, mask_top = grabcut_segmentation(food_top)
    _, mask_side = grabcut_segmentation(food_side)

    alpha_top = compute_alpha(coin_top)
    alpha_side = compute_alpha(coin_side)

    food_class = next((p['class'] for p in side_preds if p['class'] != 'coin'), "unknown")
    shape_type = shape_mapping.get(food_class, "ellipsoid")

    volume = estimate_volume(mask_side, alpha_side, alpha_top, shape_type)
    mass, calories = estimate_mass_and_calories(volume, food_class, density_mapping, energy_mapping)

    return {
        "image": top_image_path.name,
        "class": food_class,
        "volume_mm3": volume,
        "mass_g": mass,
        "calories": calories
    }
