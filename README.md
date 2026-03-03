# 🩺 Breast Cancer Classification using PyTorch

A modular Deep Learning project implementing a Multi-Layer Perceptron (MLP) using PyTorch to classify tumors as Benign or Malignant.

This project demonstrates a complete machine learning workflow including data preprocessing, model development, training with early stopping, evaluation using standard metrics, and clean modular project structure following industry best practices.

---

## 📌 Project Overview

The objective of this project is to build a neural network from scratch using PyTorch for binary classification on medical data.

This project emphasizes:

- Clean and modular code structure  
- Proper data preprocessing  
- Implementation of CrossEntropyLoss  
- Training with early stopping  
- Model evaluation using classification metrics  
- Saving trained model for future inference  

---

## 📊 Dataset

**Breast Cancer Wisconsin Dataset (sklearn built-in dataset)**

- Total Samples: 569  
- Features: 30 numerical features  
- Target Classes:
  - `0` → Malignant  
  - `1` → Benign  

The dataset consists of computed features derived from digitized images of breast mass cell nuclei.

---

## 🧠 Model Architecture

The neural network consists of:

- **Input Layer:** 30 neurons  
- **Hidden Layer 1:** 16 neurons + ReLU  
- **Hidden Layer 2:** 8 neurons + ReLU  
- **Output Layer:** 2 neurons  

### Training Configuration

- Loss Function: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Learning Rate: `0.001`  
- Batch Size: `32`  
- Epochs: `50`  
- Early Stopping: Enabled  

---

## 🏗 Project Structure
breast_cancer_project/
│
├── data.py # Data loading & preprocessing
├── model.py # Neural network architecture
├── train.py # Training logic
├── evaluate.py # Evaluation metrics
├── main.py # Entry point
└── requirements.txt


---

## ⚙️ Workflow

1. Load dataset from sklearn  
2. Split dataset into train and test sets  
3. Normalize features using StandardScaler  
4. Convert data into PyTorch tensors  
5. Train neural network using DataLoader  
6. Apply early stopping based on validation loss  
7. Evaluate performance using classification report  
8. Save trained model (`.pth` file)  

---

## 📈 Evaluation Metrics

Model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 🚀 How to Run

### Clone Repository

```bash
git clone <your-repo-link>
cd breast_cancer_project

### Install dependencies  

pip install -r requirements.txt

### Execute Project
python main.py


### Model Output

breast_cancer_classifier.pth

### 🛠 Technologies Used

Python
PyTorch
NumPy
Scikit-learn

🎯 Key Learnings

Building neural networks with PyTorch
Understanding CrossEntropyLoss for classification
Using DataLoader for batching
Implementing early stopping
Structuring ML projects professionally
Evaluating classification models properly


👨‍💻 Author
Chetan
Machine Learning Enthusiast
Chetan Kamani
Machine Learning Enthusiast
