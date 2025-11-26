# 🍎 Calorie Estimation from Food Images
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Deep Learning](https://img.shields.io/badge/Object%20Detection-YOLOv8-orange.svg)
![Segmentation](https://img.shields.io/badge/Segmentation-GrabCut-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-green.svg)

This project implements an end-to-end deep learning pipeline that estimates the **volume, mass, and calorie content** of food items using a **top** and **side** image.  
The system combines object detection, segmentation, geometric modeling, and nutritional density mapping to deliver accurate, image-based calorie estimation.  

---

## 🚀 Features
- YOLOv8 for fast, accurate food + coin detection  
- Comparison with Faster R-CNN and a custom YOLO-from-scratch model  
- GrabCut segmentation to isolate food boundaries  
- Geometric modeling for volume estimation (ellipsoid, cylinder, irregular shapes)  
- Calorie prediction using density × volume × kcal/g  
- Streamlit web application for real-time estimation

---

## 📊 Dataset
- **ECUST Food Dataset** with top + side views  
- 2,000+ images across 19 food classes  
- Includes a **One Yuan Coin** for metric scale calibration  

---

## 🧠 Pipeline Overview
1. Detect food item + reference coin using YOLOv8  
2. Segment food using GrabCut  
3. Convert pixel measurements to centimeters using the coin  
4. Estimate 3D volume using geometric assumptions  
5. Convert: **Volume → Mass → Calories**  
6. Display results via Streamlit
