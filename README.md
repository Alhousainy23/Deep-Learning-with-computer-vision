# 🧠 Deep Learning with Computer Vision

> A comprehensive course repository covering Computer Vision fundamentals through advanced Deep Learning architectures — from pixel manipulation to state-of-the-art object detection.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📚 Table of Contents

- [About](#-about)
- [Course Structure](#-course-structure)
- [Topics Covered](#-topics-covered)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Research Papers](#-research-papers)
- [License](#-license)

---

## 📌 About

This repository contains all the code, notebooks, projects, and resources for the **Deep Learning with Computer Vision** course. It is designed to take you from the basics of image processing all the way to implementing and understanding modern deep learning architectures used in real-world computer vision tasks.

Whether you are a beginner getting started with OpenCV or an intermediate learner looking to understand YOLO and transformer-based vision models, this course has something for you.

---

## 🗂 Course Structure

| # | Module | Topics |
|---|--------|--------|
| 01 | **OpenCV Basics** | Image I/O, Color Spaces, Filters, Transformations |
| 02 | **Contours & Segmentation** | Edge Detection, Contour Analysis, Labeling |
| 03 | **Classical ML for Vision** | HOG, Feature Extraction, SVM Classifier |
| 04 | **Deep Learning Foundations** | Perceptrons, Backprop, Activation Functions |
| 05 | **CNN Architectures** | Conv Layers, Pooling, Fully Connected, Dropout |
| 06 | **Famous Architectures & Papers** | AlexNet, VGGNet, ResNet, InceptionNet |
| 07 | **Transfer Learning** | Fine-tuning Pretrained Models |
| 08 | **Object Detection** | R-CNN, Fast R-CNN, Faster R-CNN |
| 09 | **YOLO Algorithm** | YOLOv3, YOLOv5, YOLOv8 — Theory & Implementation |
| 10 | **Vision Transformers** | ViT, Attention Mechanism, DETR |

---

## 🔍 Topics Covered

### 🖼 OpenCV Basics
- Reading, writing, and displaying images & videos
- Color space conversions (BGR, RGB, HSV, Grayscale)
- Geometric transformations (resize, rotate, flip, warp)
- Image filtering (blur, sharpen, edge detection)
- Morphological operations (erosion, dilation, opening, closing)
- Histogram equalization & image thresholding

### 🔲 Contours & Segmentation
- Binary thresholding techniques (Otsu, Adaptive)
- Contour detection with `cv2.findContours`
- Contour properties (area, perimeter, bounding box, moments)
- Interactive contour labeling system
- Region-based segmentation

### 🤖 Deep Learning (CNN & Beyond)
- Building CNNs from scratch with PyTorch / Keras
- Batch Normalization, Dropout, Data Augmentation
- Training pipelines and evaluation metrics
- Overfitting diagnosis and regularization techniques

### 📄 Famous Architectures & Papers
| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **AlexNet** | 2012 | First deep CNN to win ImageNet, introduced ReLU & Dropout |
| **VGGNet** | 2014 | Very deep networks with small 3×3 convolutions |
| **GoogLeNet / Inception** | 2014 | Inception modules for multi-scale feature extraction |
| **ResNet** | 2015 | Residual connections to train very deep networks |
| **Transformer** | 2017 | Attention is All You Need — foundation of modern AI |
| **ViT** | 2020 | Vision Transformer — applying attention to image patches |

### 🎯 Object Detection
- **R-CNN family:** R-CNN → Fast R-CNN → Faster R-CNN
- Anchor boxes, Region Proposal Networks (RPN)
- Intersection over Union (IoU) and Non-Maximum Suppression (NMS)

### ⚡ YOLO Algorithm
- YOLO architecture: grid cells, bounding box prediction, class probabilities
- YOLOv3, YOLOv5, YOLOv8 — differences and improvements
- Training YOLO on custom datasets
- Real-time inference on images and video streams

---

## ✅ Prerequisites

Before starting, make sure you are comfortable with:

- Python basics (loops, functions, OOP)
- NumPy and basic linear algebra
- Basic understanding of machine learning concepts

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Deep-Learning-with-computer-vision.git
cd Deep-Learning-with-computer-vision

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```txt
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
Pillow>=9.5.0
customtkinter>=5.2.0
ultralytics>=8.0.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

---

## 📁 Repository Structure

```
Deep-Learning-with-computer-vision/
│
├── 01_OpenCV_Basics/
│   ├── images/
│   ├── 01_read_display.py
│   ├── 02_color_spaces.py
│   ├── 03_transformations.py
│   └── 04_filters.py
│
├── 02_Contours_Segmentation/
│   ├── images/
│   ├── contour_detection.py
│   └── contour_labeling_app.py      ← CustomTkinter GUI App
│
├── 03_CNN_Architectures/
│   ├── alexnet.py
│   ├── vggnet.py
│   └── resnet.py
│
├── 04_Object_Detection/
│   ├── rcnn/
│   ├── faster_rcnn/
│   └── utils/
│
├── 05_YOLO/
│   ├── yolov5_inference.py
│   ├── yolov8_custom_train.py
│   └── datasets/
│
├── 06_Vision_Transformers/
│   └── vit_demo.py
│
├── papers/                          ← PDFs & summaries of key papers
│   ├── alexnet_2012.pdf
│   ├── vgg_2014.pdf
│   ├── resnet_2015.pdf
│   └── attention_is_all_you_need_2017.pdf
│
├── assets/                          ← Images used in README
│
├── requirements.txt
└── README.md
```

---

## 📎 Research Papers

Key papers studied throughout this course:

- [AlexNet (2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — ImageNet Classification with Deep CNNs
- [VGGNet (2014)](https://arxiv.org/abs/1409.1556) — Very Deep Convolutional Networks
- [ResNet (2015)](https://arxiv.org/abs/1512.03385) — Deep Residual Learning for Image Recognition
- [Faster R-CNN (2015)](https://arxiv.org/abs/1506.01497) — Towards Real-Time Object Detection
- [YOLO (2016)](https://arxiv.org/abs/1506.02640) — You Only Look Once
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — Transformer Architecture
- [ViT (2020)](https://arxiv.org/abs/2010.11929) — An Image is Worth 16×16 Words

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ — Happy Learning 🚀</sub>
</div>
