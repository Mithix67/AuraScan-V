# LungScan Pro - Technical Architecture Overview

## 1. Core AI Model: 3D Convolutional Neural Network (3D CNN)

The heart of the system is a custom **3D CNN** designed to analyze volumetric medical data.

- **Why 3D?**: Unlike standard 2D image classifiers (like for cats/dogs), lung nodules exist in 3D space. A 2D slice might look benign, but the 3D volume reveals its true irregular shape.
- **Architecture**:
  - **Input**: `(1, 32, 32, 32)` - A single-channel 3D volume of 32x32x32 voxels.
  - **Layers**:
    - 4x Convolutional Blocks (Conv3D + BatchNorm + ReLU + MaxPool3D).
    - 2x Fully Connected Layers for final classification.
    - **Sigmoid Activation**: Outputs a probability score (0.0 to 1.0) for malignancy.
- **File**: `src/models.py` (Class: `NoduleClassifier3D`)

## 2. Clinical Risk Engine (Rule-Based Logic)

Before AI analysis, a deterministic logic engine assesses patient risk based on clinical guidelines.

- **Inspiration**: Modified version of the **PLCOm2012** lung cancer risk model.
- **Factors**: Age, Pack-Years (Smoking), BMI, COPD History, Personal/Family Cancer History, and Symptoms.
- **Logic**: Assigns weighted points to factors -> Aggregates Score -> Maps to Low/Moderate/High risk tiers.
- **File**: `src/logic.py` (Function: `calculate_risk_score`)

## 3. Data Pipeline & Processing

The system handles raw medical data and converts it into a format the AI can understand.

- **DICOM Handling** (`pydicom`): Reads raw `.dcm` files from CT scanners, extracting pixel data and metadata.
- **Image Normalization**:
  - **Resizing**: All inputs are standardized to 32x32 pixels per slice.
  - **Stacking**: 2D slices are stacked 32 times (or interpolated) to create the required 3D volume.
  - **Intensity Normalization**: Pixels are scaled to a [0, 1] range to ensure consistent model behavior across different scanners.
- **File**: `src/app.py` (Upload handling logic)

## 4. Validation & Reliability

To ensure the system is scientifically valid and safe:

- **5-Fold Cross-Validation**: The model was trained 5 separate times on different subsets of data to ensure it generalizes well to new patients.
- **Metrics**: We track AUC (Area Under Curve), Sensitivity, Specificity, and Accuracy.
- **Safety Gating**: The UI blocks the AI analysis if the patient is "Low Risk" to prevent over-screening (false positives).

## 5. Software Stack

- **Frontend**: Streamlit (Python-based web frameowork).
- **Deep Learning**: PyTorch (Torch).
- **Data Science**: NumPy, Pandas, Scikit-Learn.
- **Medical Imaging**: Pydicom, OpenCV, SimpleITK.
- **Visualization**: Plotly (Interactive Charts), Matplotlib (Slice viewing).
