# **Brain Tumor MRI Classification**

## **Overview**

A **PyTorch CNN-based deep learning project** that classifies brain MRI images into four categories:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

## **Problem Statement**

Manual classification of brain MRI scans is time-consuming and requires medical expertise. This project uses **deep learning and image classification** to automatically identify the category of a brain MRI image.

## **Dataset**

**Brain Tumor MRI Dataset** by **Masoud Nickparvar**

* **Total Images:** 7,200
* **Training Images:** 5,600
* **Validation Images:** 1,120
* **Testing Images:** 1,600
* **Classes:** 4
* **Image Size:** 128 × 128
* **Input:** RGB

### **Classes**

| **Class**      | **Training** | **Testing** |
| -------------- | -----------: | ----------: |
| **Glioma**     |        1,400 |         400 |
| **Meningioma** |        1,400 |         400 |
| **No Tumor**   |        1,400 |         400 |
| **Pituitary**  |        1,400 |         400 |
| **Total**      |    **5,600** |   **1,600** |

## **Model**

**BrainTumorCNN** — a custom CNN developed using **PyTorch**.

* **3 Convolutional Layers**
* **ReLU Activation**
* **Max Pooling**
* **Fully Connected Layers**
* **Dropout: 0.5**
* **4-Class Output**

### **Training**

* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Loss Function:** Cross-Entropy Loss
* **Batch Size:** 32
* **Epochs:** 8
* **Data Augmentation:** Random Flip and Rotation

## **Results**

* **Training Accuracy:** 96.52%
* **Validation Accuracy:** 95.27%
* **Test Accuracy:** **92.56%**

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

**Python · PyTorch · Torchvision · Scikit-learn · NumPy · Matplotlib · Seaborn · Jupyter Notebook**

## **Outcome**

The trained CNN achieved **92.56% test accuracy** in classifying brain MRI images into **four categories**, demonstrating the effectiveness of deep learning for automated MRI image classification.
