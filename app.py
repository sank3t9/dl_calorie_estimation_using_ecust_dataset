import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile
from estimator import (
    crop_from_image,
    grabcut_segmentation,
    compute_alpha,
    estimate_volume,
    estimate_mass_and_calories,
    shape_mapping,
    density_mapping,
    energy_mapping,
    class_names
)

st.set_page_config(page_title="Real-Time Food Volume & Calorie Estimator")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Real-Time Food Volume & Calorie Estimator")

top_img = st.file_uploader("Upload Top View Image", type=["jpg", "png"])
side_img = st.file_uploader("Upload Side View Image", type=["jpg", "png"])

if top_img and side_img:
    with tempfile.TemporaryDirectory() as tmpdir:
        top_path = Path(tmpdir) / top_img.name
        side_path = Path(tmpdir) / side_img.name
        top_bytes = top_img.read()
        side_bytes = side_img.read()
        top_path.write_bytes(top_bytes)
        side_path.write_bytes(side_bytes)

        
        st.subheader("Uploaded Images")
        st.image(top_bytes, caption="Top View", use_column_width=True)
        st.image(side_bytes, caption="Side View", use_column_width=True)

        
        preds_top = model.predict(source=top_path, save=False, conf=0.25)[0]
        preds_side = model.predict(source=side_path, save=False, conf=0.25)[0]

       
        def extract_bbox(results):
            food_bbox, coin_bbox, food_class = None, None, None
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                bbox = box.xywhn[0].tolist()  
                if cls_name == "coin":
                    coin_bbox = bbox
                else:
                    food_bbox = bbox
                    food_class = cls_name
            return food_bbox, coin_bbox, food_class

        top_food_bbox, top_coin_bbox, _ = extract_bbox(preds_top)
        side_food_bbox, side_coin_bbox, food_class = extract_bbox(preds_side)

        if not all([top_food_bbox, top_coin_bbox, side_food_bbox, side_coin_bbox]):
            st.warning("Could not find both food and coin in both views.")
        else:
            
            top_crop_food = crop_from_image(top_path, top_food_bbox)
            top_crop_coin = crop_from_image(top_path, top_coin_bbox)
            side_crop_food = crop_from_image(side_path, side_food_bbox)
            side_crop_coin = crop_from_image(side_path, side_coin_bbox)

            _, top_mask = grabcut_segmentation(top_crop_food)
            _, side_mask = grabcut_segmentation(side_crop_food)

            alpha_top = compute_alpha(top_crop_coin)
            alpha_side = compute_alpha(side_crop_coin)
            shape_type = shape_mapping.get(food_class, "ellipsoid")

            volume = estimate_volume(side_mask, alpha_side, alpha_top, shape_type)
            mass, calories = estimate_mass_and_calories(volume, food_class, density_mapping, energy_mapping)

            st.success("Estimation Complete!")
            st.write(f"**Class**: `{food_class}`")
            st.write(f"**Volume**: `{volume:.2f}` mm³")
            st.write(f"**Mass**: `{mass:.2f}` g")
            st.write(f"**Calories**: `{calories:.2f}` kcal")
