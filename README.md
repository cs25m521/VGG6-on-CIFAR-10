# VGG6-on-CIFAR-10
exploring VGG6 on CIFAR-10 with different configurations and analyzing model performance based on these configurations.

# Exploring VGG6 on CIFAR-10 with Different Configurations

### CS6886W – System Engineering for Deep Learning  
**Department of Computer Science and Engineering**  
**Indian Institute of Technology Madras**

---

## 📘 Overview
This repository contains the implementation and experiments for **Assignment 1** of the CS6886W course.  
The objective is to explore the performance of a **VGG6 Convolutional Neural Network** on the **CIFAR-10** dataset under different configurations — varying activation functions, optimizers, learning rates, and batch sizes.  

The project is implemented in **Python (PyTorch)** and supports both **local execution** and **Weights & Biases (W&B)** integration for automated experiment tracking and hyperparameter sweeps.

---

## 🧩 Repository Structure
```
vgg6_gelu_adam_modular
│
├── config.py
├── data.py
├── model.py
├── engine.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

### Python Version
- Python 3.10 or higher

### Dependencies
Install all required dependencies with:
```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, you can manually install the following key packages:
```bash
pip install torch torchvision torchaudio matplotlib numpy tqdm pyyaml wandb
```

---

## 🚀 Running the Code

### **Option 1.1: Train model Locally (Without W&B) **
You can train and evaluate the VGG6 model directly without using W&B:
```bash
python VGG6_main.py --activation=GELU --batch_size=256 --epochs=100 --learning_rate=0.01 --optimizer=Nadam --mode=train --weight_file_path=WEIGHT_FILE_PATH
```
This will:
- Train the model on CIFAR-10  with provided command line parameters 
- Display training progress in the console  
- Save the model weights in the WEIGHT_FILE_PATH provided

---

---

## 🚀 Running the Code

## How to Run the Project

Follow these steps to set up the environment and run the VGG6 model.

### 1. Clone the Repository

```bash
git clone https://github.com/cs25m521/VGG6-CIFAR10.git
cd VGG6-CIFAR10
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to Weights & Biases

```bash
wandb login
```

Enter your W&B API key when prompted.

### 5. Run the Model

```bash
python main.py
```

The program will train the VGG6 model on the CIFAR-10 dataset using GELU activation and the Adam optimizer. Training and validation metrics will be logged to Weights & Biases.

### 6. Output

After training, the program displays the best validation accuracy and test results. The trained model weights are saved as `vgg6_gelu_adam.pth`.

**Note:** The CIFAR-10 dataset is downloaded automatically when required. Make sure you have an active internet connection for the initial download.

## 🧪 Experiment Details

- **Dataset:** CIFAR-10  
- **Model:** Custom 6-layer VGG (VGG6)  
- **Optimizers Tested:** SGD, Nesterov-SGD, Adam, Adagrad, RMSprop, Nadam  
- **Activations Tested:** ReLU, Sigmoid, Tanh, SiLU, GELU  
- **Batch Sizes:** 64, 128,256
- **Learning Rates:** 0.01, 0.001  
- **Epochs:** 5,10,15,20

---

## 📊 Results Summary

| Configuration | Activation | Optimizer | LR | Batch Size | epochs|Best Val Accuracy |
|----------------|-------------|------------|----|-------------|-------------------|
| Best Model | GELU | adam | 0.01 | 128 |20| **85.6%** |

The GELU + NADAM configuration provided the best Results.

---

## ♻️ Reproducibility

- Random seeds are fixed in all scripts for deterministic results.  
- The training pipeline and data loading are fully reproducible.  
- The environment setup and dependency versions are provided for consistency.  

---

## 📁 Outputs
- W&B runs are automatically logged if enabled  
- Trained model weights are saved in the local directory 
- Charts are generated in W&B dashboard if running with W&B 


---



---

## 🧑‍💻 Author
**Name:** [– K V Niranjan Gupta]  
**Roll Number:** [– CS25M521]  
**Course:** CS6886W – System Engineering for Deep Learning  
**Institution:** IIT Madras  
