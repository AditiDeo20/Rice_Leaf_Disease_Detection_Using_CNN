# 🌾 PRCP-1001: Rice Leaf Disease Classification

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Precision%20Agriculture-darkgreen.svg)]()
[![Model](https://img.shields.io/badge/Architecture-MobileNetV2%20%2B%20Custom%20CNN-purple.svg)]()

Production-grade deep learning solution for automated diagnosis of major rice crop diseases (*Bacterial Leaf Blight*, *Brown Spot*, and *Leaf Smut*) built according to the **PRCP-1001** capstone specification. Features strict anti-leakage data preparation, classical ML baselines, a custom progressive CNN, transfer learning with **MobileNetV2** (fine-tuned top 30 layers), full comparative analytical reports, and a standalone diagnostic inference engine.

---

## 📌 1. Project Context & Problem Statement

Rice (*Oryza sativa*) is the primary dietary staple for more than 50% of the global population. Foliar diseases severely jeopardize agricultural productivity, frequently causing 20% to 50% yield losses during severe outbreaks. Manual inspection by agronomists is labor-intensive, geographically constrained, and error-prone during early manifestation stages.

### Target Disease Classes:
1. **Bacterial Leaf Blight** (*Xanthomonas oryzae pv. oryzae*):
   - Causes translucent, water-soaked longitudinal stripes that turn grayish-white along leaf margins, severely impeding photosynthesis.
2. **Brown Spot** (*Bipolaris oryzae* / *Cochliobolus miyabeanus*):
   - Manifests as small, circular-to-oval necrotic brown lesions with distinct yellow halos across the leaf blade.
3. **Leaf Smut** (*Entyloma oryzae*):
   - Appears as minute, angular, raised black pustules scattered across the leaf blade, often leading to premature foliar senescence.

---

## 📂 2. Repository Structure

```text
├── .gitignore                      # Git exclusion rules for large datasets and binaries
├── README.md                       # Comprehensive project documentation and benchmark report
├── requirements.txt                # Pinned production dependencies
├── PRCP-1001-RiceLeaf.ipynb        # Complete executed notebook containing all reports & models
├── src/                            # Modular production codebase
│   ├── __init__.py                 # Package initialization and exports
│   ├── data_loader.py              # Anti-leakage splitting and ImageDataGenerator pipelines
│   ├── model.py                    # Custom CNN and MobileNetV2 architecture definitions
│   └── inference.py                # Standalone single-image inference and visualizer
└── models/                         # Trained model artifacts and weights
    └── .gitkeep                    # Keeps directory tracked without storing binaries >100MB
```

---

## 🔬 3. Dataset Characteristics & Data Analysis Report

The raw dataset contains **119 disease-infected field images** evenly distributed across the 3 target classes:
- **Bacterial leaf blight**: 40 images
- **Brown spot**: 40 images
- **Leaf smut**: 39 images

### Morphological Analysis & Downscaling Justification:
- **Extreme Dimensional Variance**: Uncompressed images range from **250px to 3081px in width** and **71px to 900px in height** (mean dimensions: $\approx 2384 \times 708$ px).
- **Aspect Ratio Distortion**: Foliar specimens exhibit severe elongation with aspect ratios ranging from $0.43$ up to $8.86$.
- **Field Background Noise**: Images feature varied real-world backgrounds including soil, field water, human hands, and inconsistent outdoor sun glare.
- **128x128 Resizing Rationale**: Standardizing to $(128, 128, 3)$ using area/bilinear interpolation strikes the optimal trade-off between retaining critical lesion perimeter patterns (color gradient and halo margins) and maintaining low computational complexity required for resource-constrained edge/mobile deployment.

---

## 🛡️ 4. Anti-Leakage Partitioning & Augmentation Report

### Strict Post-Split Augmentation Protocol:
Data augmentation must **never** precede dataset splitting. When synthetic images (rotations, zooms, shears) are generated prior to partitioning, augmented variants of the identical leaf specimen inevitably spill into validation and test sets. This produces **catastrophic data leakage**, yielding artificially inflated validation scores that collapse in production.

Here, the raw data is strictly partitioned **before** augmentation into:
- **Training Set (70%)**: 83 images (28 Blight, 28 Brown Spot, 27 Smut)
- **Validation Set (15%)**: 17 images (6 Blight, 6 Brown Spot, 5 Smut)
- **Test Set (15%)**: 19 images (6 Blight, 6 Brown Spot, 7 Smut)

### Biological Validity of Transformations:
Applied strictly to the training fold via Keras `ImageDataGenerator`:
- **Rotation ($\pm 25^\circ$) & Shear Range ($0.15$)**: Reflects diverse leaf orientations within a canopy.
- **Horizontal Flip**: Preserves natural bilateral foliar symmetry while creating valid distinct views.
- **Vertical Inversion (Disabled)**: Excluded because gravity-driven lesion elongation and vascular sap flow patterns make inverted leaf lesions biologically unnatural.
- **Zoom ($0.2$) & Shifts ($0.15$)**: Simulates variable focal distances of mobile camera inspections in the field.

---

## 🧠 5. Model Architectures

### 1. Classical Machine Learning Baselines (Feature Flattening):
- **Input**: $64 \times 64 \times 3$ flattened to 12,288 normalized 1D features.
- **Support Vector Machine (SVM)**: Radial Basis Function (RBF) kernel, $C=1.0$.
- **Random Forest**: 100 decision trees, Gini impurity criterion.

### 2. Custom Progressive Convolutional Neural Network (CNN):
- Progressive feature abstraction:
  - **Conv Block 1**: $\text{Conv2D}(32, 3\times 3) \rightarrow \text{BatchNorm} \rightarrow \text{MaxPool2D}(2\times 2)$
  - **Conv Block 2**: $\text{Conv2D}(64, 3\times 3) \rightarrow \text{BatchNorm} \rightarrow \text{MaxPool2D}(2\times 2)$
  - **Conv Block 3**: $\text{Conv2D}(128, 3\times 3) \rightarrow \text{BatchNorm} \rightarrow \text{MaxPool2D}(2\times 2)$
  - **Classification Head**: $\text{Flatten} \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dropout}(0.5) \rightarrow \text{Dense}(3, \text{Softmax})$
- Trained with `Adam(1e-4)`, `ReduceLROnPlateau(factor=0.2, patience=5)`, and `EarlyStopping(patience=10)`.

### 3. Transfer Learning & Fine-Tuning with MobileNetV2:
- **Base**: Pre-trained on ImageNet with inverted residual bottlenecks and depthwise separable convolutions.
- **Stage 1 (Feature Extractor)**: Base frozen; training custom classification head:
  $$\text{GlobalAveragePooling2D} \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{BatchNorm} \rightarrow \text{Dropout}(0.4) \rightarrow \text{Dense}(3, \text{Softmax})$$
- **Stage 2 (Fine-Tuning)**: Top 30 layers unfrozen and fine-tuned with a low learning rate `Adam(1e-5)` to adapt high-level visual features to foliar pathologies.

---

## 📊 6. Empirical Benchmark & Comparison Report

| Model | Input Dimensions | Train Accuracy | Test Accuracy | Macro F1-Score | Parameter Count | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF Kernel)** | $64 \times 64 \times 3$ (12.2k 1D) | $63.8\%$ | $47.37\%$ | $0.45$ | Non-parametric | $\approx 2.1 \text{ ms}$ |
| **Random Forest (100 Trees)** | $64 \times 64 \times 3$ (12.2k 1D) | $100.0\%$ | $52.63\%$ | $0.51$ | N/A (Ensemble) | $\approx 4.8 \text{ ms}$ |
| **Custom CNN (From Scratch)** | $128 \times 128 \times 3$ | $74.7\%$ | $57.89\%$ | $0.56$ | $4,228,867$ | $\approx 9.4 \text{ ms}$ |
| **MobileNetV2 (Transfer Learning)** | **$128 \times 128 \times 3$** | **$93.9\%$** | **$84.21\%$** | **$0.85$** | **$2,423,491$** | **$\approx 14.2 \text{ ms}$** |

### MobileNetV2 Detailed Test Classification Report:
```text
                       precision    recall  f1-score   support

Bacterial leaf blight       1.00      0.83      0.91         6
           Brown spot       0.75      1.00      0.86         6
            Leaf smut       0.83      0.71      0.77         7

             accuracy                           0.84        19
            macro avg       0.86      0.85      0.85        19
         weighted avg       0.86      0.84      0.84        19
```

---

## 🚀 7. Production Deployment Recommendation

1. **Edge / Mobile Agricultural Deployment (Recommended: MobileNetV2)**:
   - **Rationale**: MobileNetV2 utilizes depthwise separable convolutions that dramatically reduce multiply-accumulate (MAC) operations. With only 2.4M parameters and a memory footprint under 12 MB (quantizable to <3 MB via TensorFlow Lite `INT8`), it delivers instant on-device inference without cellular connectivity in remote farm environments.
2. **Cloud API Inference**:
   - For centralized agricultural advisory portals, MobileNetV2 can be deployed via FastAPI or TensorFlow Serving containerized on Google Cloud Run or AWS ECS, sustaining over 150 requests/sec with minimal GPU/CPU utilization.

---

## ⚡ 8. Getting Started & Installation

### Prerequisites:
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository:
```bash
git clone https://github.com/AditiDeo20/RICE-LEAF-CNN.git
cd RICE-LEAF-CNN
```

### 2. Set Up Virtual Environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run CLI Inference:
```bash
python src/inference.py --image "dataset_split/test/Brown spot/DSC_0119.jpg" --model "models/best_mobilenetv2.keras"
```

### 5. Launch the Complete Notebook:
```bash
jupyter notebook PRCP-1001-RiceLeaf.ipynb
```

---

## 👥 Author
- **Candidate / Engineer**: Aditi Deo  
- **GitHub**: [@AditiDeo20](https://github.com/AditiDeo20)
