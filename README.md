# 🍎 DeepDish - Deep Learning Calorie Estimator 

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-green.svg)](https://streamlit.io/)

**An end-to-end deep learning pipeline that estimates food volume, mass, and calories using top and side-view images.**

---

### 🎥 [Watch the Video Demo](https://youtu.be/zN8sgxGbPHs)

---

## 🚀 Features
* **Dual-View Analysis**: Uses **Top** and **Side** images with a reference coin for accurate 3D volume estimation.
* **Object Detection**: Utilizes **YOLOv8** for real-time food and coin detection.
* **Precise Segmentation**: Implements **GrabCut** to isolate food boundaries from the background.
* **Geometric Modeling**: Calculates volume using shape-specific formulas (Ellipsoid, Column, Irregular).
* **Interactive App**: Real-time analysis via a **Streamlit** web interface.

## 📊 Performance
I benchmarked three object detection models on the **ECUST Food Dataset** (19 classes) to determine the best approach for this pipeline:

| Model | mAP@0.5 | Precision | F1 Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** | **0.882** | **0.964** | **0.949** | **Selected for Pipeline (Best Balance)** |
| Faster R-CNN | 0.975 | 0.670 | 0.799 | Highest mAP, but slower inference |
| Custom YOLO | 0.189 | 0.402 | 0.480 | Baseline comparison |

## 🧠 How It Works
1.  **Input**: Users upload Top & Side images containing the food and a One Yuan coin.
2.  **Detection**: The system (YOLOv8) locates the food item and the coin.
3.  **Segmentation**: GrabCut generates a binary mask of the food to refine boundaries.
4.  **Scaling**: The coin serves as a reference to convert pixel measurements to centimeters.
5.  **Estimation**: 
    * **Volume**: Calculated based on shape (e.g., Apple $\rightarrow$ Ellipsoid, Bread $\rightarrow$ Column).
    * **Calories**: $Volume \times Density \times Energy (kcal/g)$.

## 🛠️ Quick Start

**1. Install Dependencies**
```bash
pip install ultralytics streamlit opencv-python numpy
```

**2. Run the App**

```bash
streamlit run app.py
```