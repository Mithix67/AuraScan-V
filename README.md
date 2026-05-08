# 🔬 AuraScan — AI Lung Cancer Screening System

> **Showcase / Portfolio Version**
> This repository is a read-only architectural showcase of the AuraScan system. The core AI model weights and proprietary inference logic have been omitted. This version demonstrates the system's design, UI, and pipeline structure for educational and portfolio purposes.

---

## ⚠️ Medical Disclaimer

> **This tool is for Research & Educational Use Only.**
> It is **NOT** a certified medical device and should **NOT** be used for clinical diagnosis. All AI predictions must be verified by a qualified radiologist.

---

## 📌 Project Overview

AuraScan is an end-to-end AI system for lung cancer risk assessment and nodule classification using 3D CNNs. This project demonstrates:

- **3D Deep Learning**: Processing volumetric CT data with a custom 3D CNN.
- **Robust Validation**: 5-Fold Cross-Validation to ensure scientific reliability.
- **Interactive Clinical UI**: Streamlit dashboard for real-time multi-cancer risk assessment.
- **Grad-CAM Explainability**: Visual heatmaps showing which region of the scan influenced the AI's decision.

---

## 🚀 Key Features

- **Patient Risk Calculator**: Estimates lung cancer risk based on age, smoking history, and clinical symptoms (PLCOm2012-inspired).
- **3D Nodule Classification**:
  - Input: 3D crops of lung nodules (32×32×32 voxels)
  - Architecture: Custom 3D CNN (`NoduleClassifier3D`)
  - Output: Malignancy Probability + Risk Level
- **Scan Validation Engine**: Automatically rejects non-CT images (color photos, screenshots, UI captures) before inference.
- **Multi-Cancer Modules**: Lung, Skin, Prostate, and Colorectal risk assessment in one dashboard.

---

## 📂 Project Structure

```
AuraScan-Showcase/
├── src/
│   ├── ai_model_architecture.py   # U-Net & 3D CNN architecture definitions
│   ├── clinical_risk_calculator.py # Risk scoring engine (interface only)
│   ├── preprocessing.py            # CT scan pipeline (interface only)
│   ├── calculate_metrics.py        # Evaluation framework (interface only)
│   ├── mock_data.py                # Synthetic data generator for testing
│   └── frontend_dashboard.py       # Full Streamlit UI (demo mode)
├── TECHNICAL_OVERVIEW.md           # Architecture deep-dive
├── system_design_diagrams.md       # Mermaid diagrams (UML, DFD, Sequence)
├── database_design.md              # Data schema design
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🧠 Model Architecture

### U-Net (Segmentation)
Standard encoder-decoder with skip connections. Used to isolate lung parenchyma from chest wall, heart, and background tissue.

### NoduleClassifier3D (Classification)
- **Input**: 32×32×32 voxel patches centered on candidate nodules.
- **Layers**: 3× Conv3D blocks with BatchNorm + ReLU + MaxPool3D.
- **Classifier**: Fully connected layers with 0.5 Dropout for regularization.
- **Output**: Sigmoid probability (0.0 = Benign, 1.0 = Malignant).

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| Frontend | Streamlit |
| Deep Learning | PyTorch |
| Data Science | NumPy, Pandas, Scikit-Learn |
| Medical Imaging | Pydicom, OpenCV, SimpleITK |
| Visualization | Plotly, Matplotlib, Seaborn |

---

## 📊 Scientific Methodology

### Data Leakage Prevention
Early prototypes achieved unrealistic 1.0 AUC due to random splitting of slices from the same patient. **Fix**: Patient-wise splitting with **5-Fold Cross-Validation** ensures the model generalizes to unseen patients.

### Evaluation Metrics
- **AUC-ROC**: Overall discriminative ability.
- **Sensitivity (Recall)**: Critical for a screening tool — must catch malignant cases.
- **Specificity**: Reduces false alarms and unnecessary follow-up procedures.
- **Youden's J Optimal Threshold**: Automatically balances sensitivity and specificity.

---

## 📧 Contact

Built by **Mithix67** | AuraScan V2 — For portfolio and academic review.
