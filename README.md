# 🐦 Birdhouse Object Detection & Web Application

A modular, lightweight Python pipeline and interactive web interface that uses the **YOLO object detection algorithm** to process images and cleanly isolate specific target labels (`bird`).

This project scans local folders or accepts user uploads, identifies the highest-confidence object in an image, and utilizes a secondary custom classification layer to manage business-logic rules.

---

## 🛠️ Project Architecture

The codebase is built on a modular **Separation of Concerns**, splitting the pipeline into distinct layers:

1. **`BirdDetector` (in `src.py`)**: Interacts directly with the `ultralytics` YOLO model. It can loop through folders or read single images, running inference to extract the single highest-confidence object class and score.
2. **`BirdClassifier` (in `src.py`)**: A data-scrubbing layer. It takes the detection outputs and enforces business rules: if an object is a `bird`, it remains unchanged; if it is anything else (e.g., a cat, dog, or truck), it recodes the label to `'None'` and resets the confidence score to `0.0`.
3. **Streamlit App (`app.py`)**: An interactive user interface allowing users to upload images dynamically via their web browser to test the `BirdDetector` class in real-time.

---

## 📂 Repository Structure

```text
birdhouse/
├── .streamlit/         # Web app configuration folder
│   └── config.toml     # Suppresses UI development warnings
├── images/             # Local test directory
│   ├── image_with_bird.jpg
│   └── image_without_bird.jpg
├── app.py              # Streamlit Web Application
├── src.py              # Core Python classes (Detector & Classifier)
└── README.md           # Documentation