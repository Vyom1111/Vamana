
# 🧪 Vamana Video Analysis using YOLOv8 Segmentation

A novel Computer Vision system that brings automation to **Ayurvedic research** by analyzing the *Vamana* procedure from Panchkarma using deep learning techniques. This project aims to quantify and qualify vomital events from clinical videos using **YOLOv8 segmentation**, color analysis, and pixel-based volume estimation.

---

## 🧭 Project Overview

> **Vamana** is a traditional Ayurvedic detoxification procedure that induces controlled vomiting. Traditionally evaluated subjectively, our goal is to build an objective, automated pipeline to analyze the process from videos.

This repository hosts the implementation of a **YOLOv8-Segmentation model** that:
- Detects when the patient is in the vomiting posture (head-down).
- Segments expelled fluid in real-time.
- Classifies its color.
- Estimates the volume based on segmented area.
- Logs data with timestamps for medical review.

---

## 📸 Sample Use Case

![example](assets/sample_detection.gif) <!-- Include a sample image or GIF of the detection if available -->

---

## 🛠️ Features

- 🎯 **Custom-trained YOLOv8s-Seg** on Vamana-specific data
- 🧪 **Pixel-area to volume** conversion via regression
- 🎨 **Fluid color classification** based on RGB to CSS3 color names
- 🕒 **Timestamped event logging**
- 📊 **Excel report generation** for clinicians

---

## 🧱 Methodology

### 1. Dataset Creation &amp; Annotation
- Extracted frames from Ayurvedic procedure videos.
- Annotated vomital regions using **Roboflow**.
- Manually verified masks to maintain clinical accuracy.

### 2. Model Selection
- **YOLOv8s-Segmentation (Ultralytics)**
  - 📦 ~2.4M parameters
  - 🖼️ Input size: 640x640
  - ⏱️ Trained for 60 epochs on Google Colab (T4 GPU)

### 3. volume &amp; Color Estimation
- Color: Matched RGB values to nearest CSS3 color names (e.g., Light Yellow, Dark Green)
- volume: `volume = area_pixels × pixel_to_ml_ratio` (via linear regression)

---

## 📂 Folder Structure

```
├── data/                   # Annotated dataset
├── yolov8/                 # YOLOv8 training and prediction scripts
├── utils/                  # Color mapping, volume estimation scripts
├── outputs/                # Segmented frames and analysis results
├── results.xlsx            # Time-stamped fluid data
├── README.md
└── requirements.txt
```

---

## 🚀 Getting started

### ✅ Prerequisites

Install dependencies:
\`\`\`bash
pip install ultralytics opencv-python pillow==9.4.0 matplotlib webcolors
\`\`\`

optional:
\`\`\`bash
pip install roboflow numpy pandas
\`\`\`

---

### 🔧 Training the Model

\`\`\`Python
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")
model.train(data="data.yaml", epochs=60, imgsz=640)
\`\`\`

---

### 🎥 Inference on Video

\`\`\`Python
model.predict(source="sample_video.mp4", conf=0.25, save=True)
\`\`\`

---

### 🟡 Color Detection

\`\`\`Python
from PIL import Image
image = Image.open("segmented_output.jpg")
pixels = list(image.getdata())
# Perform RGB classification using webcolors
\`\`\`

---

### 💧 Volume Estimation

\`\`\`Python
def estimate_volume(area_pixels, pixel_to_ml_ratio=0.05):
    return area_pixels * pixel_to_ml_ratio
\`\`\`

---

## 📉 Model Evaluation

- Manually validated predictions using Ayurvedic practitioner logs.
- Detected sink-region fluid only when head-down posture was confirmed.
- Limitations: camera angle, lighting, lack of depth info.

---

## 🔬 Challenges

- Manual dataset annotation required domain expertise.
- Color and volume variability across patients.
- High inference load due to per-frame computation.

---

## 📈 Future Work

- ⏱️ Integrate **LSTM** for better temporal event detection
- 📏 Add **depth estimation** for accurate volume prediction
- 🌈 Multi-class segmentation for different vomital substances
- ☁️ Deploy as a **cloud dashboard** for Ayurvedic clinics

---

## 🤝 Contributors

- **Vyom Shah** - [@VyomShah](https://github.com/Vyom1111)  
- **Ishan Shah**  
- Supervised by **Dr. Sumit Kalra**, IIT Jodhpur  
- Co-Supervised by **Dr. Pooja Pathak**, AIIA

---

## 📄 License

This project is for research purposes only. For clinical or commercial use, please contact the authors.

---

## 🪔 Bridging Tradition and Technology

> *“Where ancient wisdom meets cutting-edge AI”*
