# Brain-Tumor-MRI-Image-Classification-
End-to-end deep learning project for classifying brain MRI scans into multiple tumor categories. Built with TensorFlow, Keras, and Streamlit — featuring data preprocessing, model training, evaluation metrics, and an interactive web demo for real-time predictions.

# 🧠 Brain Tumor MRI Image Classification

This project is a **Deep Learning-based MRI image classifier** that identifies different types of brain tumors using Convolutional Neural Networks (CNN) and Transfer Learning (ResNet50).  
It demonstrates a complete end-to-end ML workflow — from data preprocessing and model training to evaluation and deployment via a Streamlit web application.

---

## 🚀 Project Overview

Brain tumors are life-threatening diseases that require early and accurate diagnosis.  
This project leverages **deep learning** to automatically classify brain MRI scans into multiple tumor categories such as:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

🎯 **Goal:** Build a reliable and accurate model that assists healthcare professionals in early detection and classification of brain tumors.

---

## 🧩 Key Features
- 🧠 **End-to-End Deep Learning Pipeline** — Data preprocessing → Model training → Evaluation → Deployment  
- 🏗️ **Transfer Learning with ResNet50** — Boosts accuracy and reduces training time  
- 📊 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- 🖥️ **Interactive Web App** — Streamlit-powered user interface for real-time predictions  
- ☁️ **Scalable Design** — Ready for cloud deployment (AWS, Render, Heroku)

---

## 🗂️ Dataset

- **Source:** Brain MRI Images dataset (Kaggle / Google Drive link from project brief)  
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor  
- **Format:** Images organized in class-based folders  
- **Preprocessing:** Image resizing (224×224), normalization, and augmentation for better generalization  

> ⚠️ **Note:** Dataset files are not included in this repository. Please place them under the `data/raw/` folder after download.

---

## 🧠 Model Architecture

- **Base Model:** ResNet50 (pre-trained on ImageNet)  
- **Fine-tuning Layers:** GlobalAveragePooling2D → Dense(256) → Dropout(0.4) → Dense(4, softmax)  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Metrics:** Accuracy, Precision, Recall, F1-Score  

---

## 🧮 Workflow

1. **Data Preprocessing** — Cleaning, augmentation, and splitting the dataset  
2. **Model Building** — Transfer learning with fine-tuning layers  
3. **Model Training** — Optimizing using Keras callbacks (EarlyStopping, ModelCheckpoint)  
4. **Evaluation** — Model metrics, confusion matrix, and visualization  
5. **Deployment** — Streamlit-based interactive application  



