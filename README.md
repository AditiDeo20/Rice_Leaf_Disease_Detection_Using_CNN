# 🌾 Rice Leaf Disease Detection using Deep Learning

## **Project Overview**

This project detects **three rice leaf diseases** using image classification:

- **Brown Spot**
- **Leaf Smut**
- **Bacterial Leaf Blight**

We compare a **Custom CNN model** with **MobileNetV2 Transfer Learning** for classification.

---

## **Dataset**

- **Source:** [Rice Leaf Disease Dataset](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1001-RiceLeaf.zip)
- **Classes:**  
  - Brown Spot  
  - Leaf Smut  
  - Bacterial Leaf Blight

---

## **Model Comparison**

| Method                       | Train Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
|-----------------------------|----------------|---------------------|---------------|-----------|
| **Custom CNN**               | 72.63%         | 66.67%               | 73.68%        | 0.7235    |
| **Transfer Learning (MobileNetV2)** | 86.32%   | 83.33%               | 73.68%        | 0.7235    |

---

## **Key Observations**

- **Transfer Learning** improved **training and validation accuracy significantly.**
- **Test accuracy remained the same**, suggesting dataset size limitations or subtle class similarities.
- Further improvements can include:
  - Fine-tuning more layers of MobileNetV2
  - Collecting more data
  - Using ensemble models

---

## **Project Pipeline**

1. **Data Preparation**
   - Dataset split into **Train/Validation/Test (70/15/15)** using `split-folders`
   - Image size: **224×224**

2. **Models Built**
   - **Custom CNN**
