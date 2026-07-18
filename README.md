# 🚀 Computer Vision & Deep Learning Portfolio (`AI-VISION-INTERNSHIP`)

This repository contains the projects, experiments, implementation notes, source code, and weekly technical reports I developed while studying Computer Vision, Object Detection, Multi-Object Tracking, and Deep Learning. 

Rather than a single project, this repository serves as a continuously growing portfolio documenting both the theoretical concepts and practical implementations developed throughout my learning journey. The repository documents the progression from classical computer vision techniques to modern deep learning-based object detection systems.

## 📚 Repository Overview

Throughout this repository, I focus not only on using existing libraries but also on understanding how the underlying algorithms work by implementing real-world applications and training custom object detection models from scratch. 

Moving forward, the repository will heavily focus on the mathematical and structural depths of **Convolutional Neural Networks (CNN)**, with active developments in **Image Classification** and **Segmentation**.

---

## 🛠️ Technologies & Tools

*   **Programming Language:** Python
*   **Libraries & Frameworks:** OpenCV, NumPy, Ultralytics YOLOv8, SAHI, Supervision, Matplotlib
*   **Tools:** Google Colab, Label Studio, Git, GitHub
*   **Core Concepts:** Computer Vision, Image Processing, Object Detection, Multi-Object Tracking (MOT), NMS, Re-ID, HSV/BGR Color Spaces, Image Preprocessing (Blur, Canny, Sobel, Thresholding)

---

## 💻 Implemented Projects

### 1. Classical Computer Vision & Color-Based Object Tracking
Implemented classical computer vision applications to understand how images are processed before deep learning.
*   **Drone Detection & Tracking:** Detected a drone in the sky using HSV color space transformations, morphological operations (dilate, erode), and contour analysis.
*   **Geometric Filtering:** Applied area and aspect ratio metrics to successfully filter out false positives like building shadows and city backgrounds.
*   **Real-time Red Ball Tracking:** Tracked a red ball using HSV masking and instant center/radius calculations.

### 2. Deep Learning Detection & Multi-Object Tracking (MOT)
*   **Cafe Human Tracking:** Detected humans in close-up environments using YOLOv8, optimized with Non-Maximum Suppression (NMS).
*   **Identity (ID) Preservation in Crowded Scenes:** Integrated advanced tracking algorithms (ByteTrack, BoT-SORT, Deep OC-SORT, and TrackTrack) and Re-ID configurations to solve ID swapping and maintain tracking consistency during occlusions.

### 3. SAHI (Slicing Aided Hyper Inference) Integration
*   **Overcoming Model Blindness:** Addressed YOLO's inability to detect small objects in high-resolution, wide-angle cameras (e.g., Üsküdar video) by slicing images into overlapping patches and performing independent inferences.

### 4. Custom YOLO Model Training & Real-Time People Counting
Built a complete pipeline from dataset preparation to model evaluation.
*   **Dataset Preparation:** Extracted video frames and created a custom annotated dataset using Label Studio.
*   **Model Training:** Trained a YOLOv8n architecture on Google Colab for 50 epochs, achieving high success metrics.
*   **Line Crossing System:** Developed a real-time, bi-directional people counting application using the custom trained model, ByteTrack, and Supervision's LineZone component.

**Model Training Results:**
| Metric | Score |
| :--- | :--- |
| Precision | 0.948 |
| Recall | 0.935 |
| mAP@50 | 0.959 |
| mAP@50-95 | 0.601 |

### 5. Preprocessing and Face Detection (Classical vs. Modern)
*   **Filtering and Edge Detection:** Applied Box, Gaussian, Median, and Bilateral filters alongside Sobel and Canny edge detection algorithms.
*   **Haar Cascade Face Detection:** Implemented Viola-Jones architecture for face detection and conducted a comparative analysis of its angle/rotation limitations against modern YOLO models.

---

## 📂 Repository Structure

```text
AI-VISION-INTERNSHIP/
│
├── week_1/
│   ├── day-01/
│   ├── day-02/
│   ├── day-03/
│   ├── day-04/
│   ├── day-05/
│   ├── Hafta1_Rapor.md
│   ├── people_in_the_cafe_IDsiz.ipynb
│   ├── people_in_the_cafe.ipynb
│   └── people_in_uskudar.ipynb
│
├── week_2/
│   ├── day-01/
│   ├── day-02/
│   ├── day-03/
│   │   ├── door_coordinates.py
│   │   ├── Gun3_16.07.2026_Not.md
│   │   ├── market_kare.png
│   │   ├── market_people_count.ipynb
│   │   └── market_video.mp4
│   ├── day-04/
│   └── Hafta_2_Raporu_last.md
│
├── week_3/
├── week_4/
│
├── .gitignore
├── README.md
└── yolov8m.pt

## 📖 Weekly Reports

Each weekly report in the `week_X/` directory documents the learning process in detail, including:
*   Daily progress and implemented projects
*   Theoretical research notes and technical observations
*   Challenges encountered and applied solutions
*   Weekly learning outcomes and next week's objectives


## 👨‍💻 Author

**Işıl Bilasa**
*Computer Engineering Student at Eskişehir Technical University*

Interested in Artificial Intelligence, Deep Learning, Computer Vision, and Autonomous Systems.