# 🤟 Real-Time American Sign Language (ASL) Recognition System

This project was developed as part of the **CIS 583: Introduction to Deep Learning** course. It focuses on creating a real-time system that recognizes American Sign Language (ASL) hand gestures and translates them into readable text using deep learning and computer vision techniques.

---

## 📌 Project Overview

- Recognizes **ASL gestures** for **alphabet letters (A–Z)** and **numerical digits (0–9)**
- Uses a **live webcam feed** with OpenCV for real-time hand tracking and gesture classification
- Powered by a **CNN-based MobileNetV2 model** trained on a custom dataset of ASL gestures

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **MediaPipe**
- **NumPy, Scikit-learn, Matplotlib**

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (Transfer Learning from ImageNet)
- **Added Layers:** Flatten → Dense (ReLU) → Dropout → Dense (Softmax)
- **Training:** 20 epochs with batch size 32
- **Preprocessing:** Resizing to 224x224, normalization
- **Augmentation:** Rotation, shifting, zooming, horizontal flipping

---

## 📂 Dataset

- Self-collected images of hand gestures captured using a webcam
- Covers 36 classes: A–Z and 0–9
- Augmented to improve generalization and prevent overfitting

---

## 🚀 Project Progress

During the **midterm phase**, we collected and preprocessed the dataset, trained a MobileNetV2-based CNN model, and achieved initial gesture classification with decent accuracy. However, we encountered difficulty distinguishing visually similar gestures such as **‘M’ vs ‘N’** and **‘V’ vs ‘2’**. For the **final phase**, we focused on enhancing the model’s accuracy by adding **custom rule-based logic** that analyzes finger states using hand landmark positions. This hybrid approach—combining model predictions with rule-based corrections—helped us significantly improve recognition accuracy for confusing gestures while maintaining real-time performance.

---

## ✅ Key Features

- Real-time gesture recognition using webcam input
- Full coverage of static ASL alphabet and numerical gestures
- Improved handling of similar gestures through logical enhancements
- Modular codebase for easy updates or retraining

---

## 🧪 Evaluation

- Evaluated using accuracy, precision, recall, and F1-score
- Additional confusion matrix and classification report included
- Tested across various lighting conditions and angles for robustness

---

Below is a sample frame from real-time prediction with the recognized class overlaid:
![Gesture Example](https://github.com/user-attachments/assets/87d771fa-77c3-4dcc-8ea2-2ee1bf51f761)

---

Below is an example comparing visually similar gestures and how the system distinguishes between them:
![Hand Landmarks](https://github.com/user-attachments/assets/606a8314-6777-4425-9e1c-4342d60c1c0b)
&nbsp;
![M vs N](https://github.com/user-attachments/assets/7b485785-c4b8-40f7-adf2-d612971149b3)

---

## 🖥️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asl-recognition.git
   cd asl-recognition
