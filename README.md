# **Brain Tumor MRI Classification**

A **deep learning image classification project** built with **PyTorch** to classify brain MRI scans into four categories.

###  Live  **[Try the Brain Tumor MRI Classifier](https://dlbraintumormricnn-jomyzkazajbtvvkfrbkiaj.streamlit.app/)**

---

## **Overview**

This project uses a custom **Convolutional Neural Network (CNN)** to classify brain MRI images into:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

The trained model processes an MRI image and predicts its corresponding category.

## **Problem Statement**

Classifying brain MRI scans manually can be **time-consuming** and requires specialized medical expertise. This project explores how **deep learning and computer vision** can be used to automate MRI image classification and assist in identifying different brain tumor categories.

## **Dataset**

The project uses the **Brain Tumor MRI Dataset** by **Masoud Nickparvar**.

* **Total Images:** 7,200
* **Training Images:** 5,600
* **Validation Images:** 1,120
* **Testing Images:** 1,600
* **Number of Classes:** 4
* **Image Size:** 128 × 128 pixels
* **Input Format:** RGB

### **Class Distribution**

| **Class**      | **Training** | **Testing** |
| -------------- | -----------: | ----------: |
| **Glioma**     |        1,400 |         400 |
| **Meningioma** |        1,400 |         400 |
| **No Tumor**   |        1,400 |         400 |
| **Pituitary**  |        1,400 |         400 |
| **Total**      |    **5,600** |   **1,600** |

## **Model Architecture**

The project implements a custom CNN named **`BrainTumorCNN`** using **PyTorch**.

* **3 Convolutional Layers**
* **ReLU Activation**
* **Max Pooling**
* **Fully Connected Layers**
* **Dropout: 0.5**
* **4-Class Output Layer**

### **Training Configuration**

| **Parameter**         | **Value**              |
| --------------------- | ---------------------- |
| **Framework**         | PyTorch                |
| **Optimizer**         | Adam                   |
| **Learning Rate**     | 0.001                  |
| **Loss Function**     | Cross-Entropy Loss     |
| **Batch Size**        | 32                     |
| **Epochs**            | 8                      |
| **Image Size**        | 128 × 128              |
| **Data Augmentation** | Random Flip & Rotation |

## **Results**

The trained model achieved the following performance:

| **Metric**              | **Accuracy** |
| ----------------------- | -----------: |
| **Training Accuracy**   |   **96.52%** |
| **Validation Accuracy** |   **95.27%** |
| **Test Accuracy**       |   **92.56%** |

The final evaluation was performed on **1,600 previously unseen MRI images**.

## **Project Structure**

```text
brain-tumor-mri-classification/
│
├── Brain Tumor MRI.ipynb
├── Brain Tumor MRI.pth
├── README.md
└── dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

## **Technologies**

**Python · PyTorch · Torchvision · Scikit-learn · NumPy · Matplotlib · Seaborn · Jupyter Notebook · Streamlit**

## **Outcome**

The custom CNN achieved a **92.56% test accuracy** on **1,600 unseen MRI images**, demonstrating the potential of **deep learning-based image classification** for automated brain MRI category prediction.

> **Note:** This project is intended for educational and research purposes and is not a substitute for professional medical diagnosis.
