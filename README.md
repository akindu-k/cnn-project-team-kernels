# 🧠 CNN Project - Team Kernels

## 📘 Module: EN3150 Pattern Recognition

**Assignment 03: Simple Convolutional Neural Network for Image Classification**

---

## 🧾 Project Overview

This project focuses on designing, training, and evaluating **Convolutional Neural Networks (CNNs)** for image classification using the **MNIST Handwritten Digit Recognition dataset**.
It explores the effects of different optimization algorithms and transfer learning methods on model performance.

The assignment involves:

* Building a custom CNN architecture from scratch
* Training with different optimizers: **SGD**, **SGD with Momentum**, and **Adam**
* Using **Transfer Learning** with pre-trained CNNs
* Comparing model performance using various metrics

---

## 🌐 Live Demo

🎯 **Deployed Application (Adam Optimizer Model):**
🔗 [MNIST Flask App on Render](https://mnist-flask-app-ef8w.onrender.com/)

▶️ **Demo Video:**
🔗 [Watch on YouTube](https://youtu.be/iCB3fTPAtTA?si=CnL3sSvZhkcejT_C)

💻 **Repository for Deployment:**
🔗 [mnist-flask-app Repository](https://github.com/akindu-k/mnist-flask-app.git)

This web application allows users to **draw a digit (0–9)** on a canvas and receive **real-time classification predictions** powered by the trained Adam optimizer CNN model.
It was built using **Flask** for the backend and **PyTorch** for inference.

---

## 🧮 Dataset Information

### **Dataset:** MNIST Handwritten Digit Recognition

* **Source:** [Yann LeCun’s MNIST Database](http://yann.lecun.com/exdb/mnist/)
* **Classes:** 10 (Digits 0–9)
* **Total Images:** 70,000 (60,000 training, 10,000 testing)
* **Image Size:** 28×28 pixels, grayscale
* **Split Used:**

  * 70% Training
  * 15% Validation
  * 15% Testing

This dataset is a gold standard for testing image classification models.

---

## 🏗️ Repository Structure

```bash
cnn-project-team-kernels/
│
├── code/
│   ├── Adam - Code/
│   │   ├── CNN_Adam.ipynb
│   │
│   ├── SGD - Code/
│   │   ├── CNN_Standard_SGD.ipynb
│   │
│   ├── SGD with Momentum - Code/
│   │   ├── CNN_SGD_Momentum.ipynb
│   │
│   └── Transfer Learning - Code/
│       ├── transfer_learning_models.ipynb
│
└── report/
    ├── Adam/
    ├── SGD/
    ├── SGD + Momentum/
    
```

* **code/** → Contains all implementation scripts for CNNs trained with different optimizers and transfer learning.
* **report/** → Contains results, plots, confusion matrices, and performance analysis for each model.

---

## 👨‍💻 Team Members

| Name              | Student ID |
| ----------------- | ---------- |
| Bandara A.H.M.D.T | 220057C    |
| Kalhan M.K.A      | 220298N    |
| Perera I.A.S      | 220464V    |
| Wijewardena L.T.N | 220728K    |

---

## 🧩 Assignment Tasks Breakdown

### 1. Dataset Preparation

* MNIST dataset split into training, validation, and testing sets.

### 2. Custom CNN Architecture

* Two convolutional layers with ReLU activation and max pooling
* One fully connected layer with dropout
* Softmax classification output

### 3. Optimizer Experiments

Trained using three optimizers:

* **SGD**
* **SGD with Momentum**
* **Adam**

### 4. Transfer Learning

* Fine-tuned **VGG16** and **ResNet18** models.
* Compared with custom CNNs on accuracy, loss, and generalization.

### 5. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Training & Validation Loss Curves

---

## 📊 Results Summary

| Optimizer      | Validation Accuracy | Training Accuracy | Notes                     |
| -------------- | ------------------- | ----------------- | ------------------------- |
| SGD            | 98.57%              | 98.05%            | lr = 1e-2                 |
| SGD + Momentum | 99.01%              | 99.50%            | lr = 1e-2, momentum = 0.9 |
| Adam           | 99.10%              | 98.21%            | lr = 1e-3                 |

✅ **Adam Optimizer** was selected for deployment due to its stable convergence and superior validation performance.

---

## ⚙️ Technologies Used

* **Language:** Python
* **Frameworks:** PyTorch, TensorFlow/Keras, Flask
* **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn
* **Deployment:** Render
* **Tools:** Google Colab, Jupyter Notebook

---

## 📈 Performance Metrics Explained

* **Accuracy:** Fraction of correctly predicted samples.
* **Precision:** Measure of true positive predictions among all positive predictions.
* **Recall:** Measure of how many actual positives were correctly predicted.
* **F1-Score:** Harmonic mean of Precision and Recall — useful for imbalanced data.

> For MNIST, **accuracy** is often the main metric due to balanced classes and clear visual separation between digits.

---

## 📚 References

* [CS231n: Neural Networks](https://cs231n.github.io/neural-networks-3/)
* [Keras Optimizers Documentation](https://keras.io/api/optimizers/)
* [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html)
* [FloydHub CNN Guide](https://blog.floydhub.com)
* Fukushima, K. (1980). *Neocognitron: A Self-Organizing Neural Network Model for Pattern Recognition.*
* LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*

---

## 🧾 Instructor

**Dr. Sampath K. Perera**
Department of Electronic and Telecommunication Engineering
University of Moratuwa
*September 17, 2025*

---

> *This repository was created as part of the EN3150 Pattern Recognition coursework to explore CNN-based image classification and optimization strategies. The deployed Flask app demonstrates real-time digit recognition powered by the Adam-optimized CNN model.*

---

