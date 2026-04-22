# 🩺 Tuberculosis Detection using Chest X-Ray Images

A machine learning project that detects **Tuberculosis (TB)** from chest X-ray images using three different model architectures — CNN, ANN, and Decision Tree — with a deployed **Streamlit web application** for clinical screening.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Tuberculosis is a leading infectious disease globally, responsible for millions of deaths annually. Early detection through chest X-ray screening is critical — but manual diagnosis is time-consuming and requires expert radiologists.

This project builds an **automated TB screening tool** that classifies chest X-rays as either **Normal** or **Tuberculosis** using machine learning. Three models are compared to identify the best architecture for medical image classification.

---

## 🗂️ Project Structure

```
Tuberculosis-Detection-using-Chest-X-Ray-Images/
├── cnn_Model.py              # CNN model training & evaluation
├── ann_Model.py              # ANN model training & evaluation
├── decisionTree.py           # Decision Tree model training & evaluation
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── confusion_matrix_3500_normal_700_tb.png   # CNN confusion matrix result
├── prediction_result.txt     # Sample prediction output
└── README.md
```

---

## 📊 Dataset

- **Source:** TB Chest Radiography Database
- **Classes:** Normal | Tuberculosis
- **Training configuration:** 3,500 Normal + 700 TB images
- **Image size:** 128×128 pixels (grayscale)
- **Split:** 80% training / 20% testing (stratified)

---

## 🤖 Models & Results

Three models were implemented and compared on the same dataset:

| Model | Test Accuracy | TB Sensitivity | Notes |
|-------|:------------:|:--------------:|-------|
| **CNN** ✅ | **96.5%** | **90.7%** | Best performer — 3 Conv layers + Dropout |
| ANN | 94.2% | — | Fully connected neural network |
| Decision Tree | 72–78% | — | Baseline traditional ML |

### CNN Confusion Matrix (3500 Normal + 700 TB)

| | Predicted Normal | Predicted TB |
|---|:---:|:---:|
| **Actual Normal** | 696 ✅ | — |
| **Actual TB** | 13 ❌ | 127 ✅ |

> Only **13 TB cases were missed** out of 700 — making the CNN clinically relevant as a screening tool.

### Key Insight
The CNN significantly outperforms the Decision Tree, confirming that **convolutional architectures are superior for high-dimensional medical image data** where spatial feature extraction is critical.

---

## 🏗️ CNN Architecture

```
Input (128×128×1 grayscale)
    ↓
Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
    ↓
Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
    ↓
Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)
    ↓
Flatten → Dense(128, ReLU) → Dropout(0.5)
    ↓
Dense(1, Sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 10 | **Batch Size:** 32

---

## 🌐 Streamlit Web Application

The `app.py` provides an interactive web interface for clinicians and researchers:

**Features:**
- Upload any chest X-ray (PNG/JPG)
- Select or upload a trained `.h5` model
- Get instant prediction: **Normal** or **TB**
- View confidence score and probability
- See preprocessed image fed to the model
- Download prediction result as a text file

**Run the app:**

```bash
streamlit run app.py
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/jensikakdiya/Tuberculosis-Detection-using-Chest-X-Ray-Images.git
cd Tuberculosis-Detection-using-Chest-X-Ray-Images
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the **TB Chest Radiography Database** and place it in the project folder:
```
TB_Chest_Radiography_Database/
├── Normal/
└── Tuberculosis/
```

### 4. Train a model
```bash
python cnn_Model.py        # Train CNN
python ann_Model.py        # Train ANN
python decisionTree.py     # Train Decision Tree
```
This will save the trained model as a `.h5` file (e.g., `model_normal3500_tb700.h5`).

### 5. Run the web app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow>=2.10
streamlit
scikit-learn
numpy
matplotlib
seaborn
Pillow
```

---

## 🚀 Future Enhancements

- [ ] Transfer learning with pre-trained models (VGG16, ResNet50)
- [ ] Grad-CAM visualisation to highlight suspicious regions
- [ ] DICOM image format support
- [ ] REST API deployment
- [ ] Multi-class classification (TB severity grading)

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis.

---

## 👩‍💻 Author

**Jensi Kakdiya**
M.Sc. Data Science | Marwadi University, Rajkot

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/jensi-kakdiya-245585282)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/jensikakdiya)

---

⭐ If you found this project useful, please give it a star!
