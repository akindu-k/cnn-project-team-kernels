# CNN Project - Team Kernels

## 📘 Module: EN3150 Pattern Recognition  
**Assignment 03: Simple Convolutional Neural Network for Image Classification**

---

## 🧠 Project Overview
This project focuses on designing, training, and evaluating Convolutional Neural Networks (CNNs) for image classification tasks.  
It explores different optimization algorithms and transfer learning techniques to analyze their effects on model performance.

The assignment involves:
- Building a simple CNN architecture from scratch.
- Experimenting with various optimizers (SGD, SGD with Momentum, Adam).
- Fine-tuning a pre-trained model using **Transfer Learning**.
- Comparing performance metrics across models.

---

## 🏗️ Repository Structure

```bash
cnn-project-team-kernels/
│
├── code/
│   ├── Adam - Code/
│   │   ├── model_adam.py
│   │   
│   │   
│   │      
│   │      
│   │
│   ├── SGD - Code/
│   │   
│   │   
│   │   
│   │    
│   │    
│   │
│   ├── SGD with Momentum - Code/
│   │   
│   │   
│   │   
│   │   
│   │       
│   │
│   └── Transfer Learning - Code/
│       
│
└── report/
    ├── Adam/
    │   
    │
    ├── SGD/
    │   
    │
    ├── SGD + Momentum/
    │   
    │
    └── Transfer Learning/
        

```

- **code/** → Contains all Python code implementations for different optimizers and transfer learning.  
- **report/** → Contains detailed analysis reports, results, and visualizations for each method.

---

## 👨‍💻 Team Members

| Name | Student ID |
|------|-------------|
| Bandara A.H.M.D.T | 220057C |
| Kalhan M.K.A | 220298N |
| Perera I.A.S | 220464V |
| Wijewardena L.T.N | 220728K |

---

## 🧩 Assignment Tasks Breakdown

### 1. Dataset Preparation
- Selected an appropriate dataset from the **UCI Machine Learning Repository** (excluding CIFAR-10).
- Split into **70% training**, **15% validation**, and **15% testing**.

### 2. Custom CNN Architecture
- Implemented a CNN with:
  - 2 Convolutional + MaxPooling layers
  - Fully connected layer with dropout
  - Softmax output layer  
- Experimented with:
  - Activation functions
  - Kernel sizes and filter counts
  - Dropout rates

### 3. Optimizers and Comparison
- Trained models using:
  - **SGD**
  - **SGD with Momentum**
  - **Adam**
- Compared performance using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
  - Training & Validation Loss Curves

### 4. Transfer Learning
- Fine-tuned state-of-the-art pretrained CNNs (e.g., **VGG16**, **ResNet18**) using the same dataset.
- Recorded training/validation loss per epoch and evaluated on the test set.
- Compared results with the custom CNN.

### 5. Analysis
- Discussed the impact of optimizers, learning rates, and momentum.
- Analyzed trade-offs between custom CNNs and pretrained models.
- Summarized advantages and limitations of transfer learning approaches.

---

## ⚙️ Technologies Used
- **Language:** Python  
- **Frameworks:** PyTorch / TensorFlow + Keras  
- **Libraries:** NumPy, Matplotlib, Scikit-learn, Pandas  
- **Tools:** Google Colab / Jupyter Notebook  

---

## 📊 Evaluation Metrics
- Training & Validation Loss  
- Accuracy  
- Precision & Recall  
- Confusion Matrix  

---

## 📈 Results Summary (to be updated after report completion)
| Optimizer | Validation Accuracy | Test Accuracy | Notes |
|------------|--------------------|----------------|-------|
| SGD | – | – | – |
| SGD + Momentum | – | – | – |
| Adam | – | – | – |
| Transfer Learning (VGG16/ResNet18) | – | – | – |

---

## 🧾 References
- [CS231n: Neural Networks](https://cs231n.github.io/neural-networks-3/)
- [Keras Optimizers Documentation](https://keras.io/api/optimizers/)
- [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html)
- [FloydHub CNN Guide](https://blog.floydhub.com)
- Fukushima, K. (1980). *Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position.*  
- LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*

---

## 📅 Instructor
**Sampath K. Perera**  
Department of Electrical Engineering  
University of Moratuwa  
*September 17, 2025*

---

> _This repository was created as part of EN3150 Pattern Recognition coursework to explore CNN-based image classification and optimization strategies._

