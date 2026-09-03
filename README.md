https://dlbraintumormricnn-jomyzkazajbtvvkfrbkiaj.streamlit.app/

# **Brain Tumor MRI Classification**

## **Overview**

This project uses a **Convolutional Neural Network (CNN)** built with **PyTorch** to classify brain MRI images into **four categories**. The project addresses the problem of **automated brain tumor classification from MRI scans**, helping distinguish between different tumor types and scans with no tumor.

## **Problem Statement**

Manual analysis of brain MRI scans can be **time-consuming** and requires **medical expertise**. This project applies deep learning to classify MRI images into:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

The goal is to build an **image classification model** that can automatically identify the class of a given brain MRI image.

## **Dataset**

The project uses the **Brain Tumor MRI Dataset** by **Masoud Nickparvar**.

* **Total images:** **7,200**
* **Training images:** **5,600**
* **Validation images:** **1,120**
* **Testing images:** **1,600**
* **Classes:** **4**
* **Image size:** **128 × 128 pixels**
* **Input channels:** **3 (RGB)**

The original training set contains **5,600 images**, which were split into **80% training** and **20% validation** data.

### **Class Distribution**

| **Class**      | **Training** | **Testing** |
| -------------- | -----------: | ----------: |
| **Glioma**     |    **1,400** |     **400** |
| **Meningioma** |    **1,400** |     **400** |
| **No Tumor**   |    **1,400** |     **400** |
| **Pituitary**  |    **1,400** |     **400** |
| **Total**      |    **5,600** |   **1,600** |

## **Model**

A custom CNN named **`BrainTumorCNN`** was implemented using **PyTorch**. The architecture includes:

* **3 convolutional layers**
* **ReLU activation**
* **Max pooling**
* **Fully connected layers**
* **Dropout (0.5)**
* **4-class output layer**

### **Training Configuration**

* **Optimizer:** Adam
* **Learning rate:** **0.001**
* **Loss function:** Cross-Entropy Loss
* **Batch size:** **32**
* **Epochs:** **8**
* **Image size:** **128 × 128**

Data augmentation was applied to the training images using **random horizontal flipping** and **random rotation**.

## **Results**

The model achieved:

* **Training Accuracy:** **96.52%**
* **Validation Accuracy:** **95.27%**
* **Test Accuracy:** **92.56%**

The final test evaluation was performed on **1,600 previously unseen MRI images**.

### **Test Performance**

| **Class**            | **Precision** | **Recall** | **F1-Score** |
| -------------------- | ------------: | ---------: | -----------: |
| **Glioma**           |      **0.95** |   **0.79** |     **0.87** |
| **Meningioma**       |      **0.90** |   **0.92** |     **0.91** |
| **No Tumor**         |      **0.88** |   **0.99** |     **0.93** |
| **Pituitary**        |      **0.98** |   **1.00** |     **0.99** |
| **Overall Accuracy** |               |            |     **0.93** |

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

* **Python**
* **PyTorch**
* **Torchvision**
* **Scikit-learn**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

## **Outcome**

The trained CNN successfully classifies brain MRI images into **four categories** with a **92.56% test accuracy**, demonstrating the effectiveness of a **custom deep learning approach for automated brain tumor MRI classification**.
 approach for automated brain
tumor MRI classification.
