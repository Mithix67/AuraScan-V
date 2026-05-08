# 4. Overall Description

## 4.1 Product Perspective
AuraScan (Lung Cancer V2) is an advanced clinical decision support tool designed to operate as a high-intelligence diagnostic layer within a medical institution's digital infrastructure. It represents the next generation of the AuraScan series, specifically evolving to handle 3D volumetric data rather than static 2D slices.

### 4.1.1 System Interfaces and Integration
The product is designed with a "plug-and-play" philosophy regarding hospital IT ecosystems. It interfaces with the following core components:
*   **Imaging Modalities (CT Scanners)**: Directly consumes high-resolution CT scans. The system is agnostic to scanner manufacturers (e.g., GE, Siemens, Philips) as long as the output adheres to the DICOM Standard.
*   **PACS (Picture Archiving and Communication Systems)**: AuraScan acts as an edge-processing node. It can be configured to "listen" for new CT studies on the PACS network, automatically triggering analysis when a relevant chest scan is archived.
*   **Electronic Health Records (EHR)**: The system utilizes standard API protocols (such as HL7 or FHIR) to retrieve structured patient history, including age, gender, smoking status, and clinical symptoms, ensuring a holistic risk calculation.

### 4.1.2 Hardware and Software Constraints
To ensure scalability across various medical settings, the product is built on a containerized architecture (Docker). 
*   **Deployment**: Can be hosted locally on a hospital "Mini-Server" or on a secure private cloud. 
*   **Hardware Requirements**: While the deep learning model inference is optimized for CPU-only environments (minimum 16GB RAM, i7 Processor), the system supports NVIDIA GPU acceleration (CUDA) to reduce processing time from ~120ms to <20ms per volume in high-volume environments.
*   **Operating Environment**: A cross-platform web interface ensures accessibility via any standard browser (Chrome, Edge, Safari) on hospital tablets or workstations.

### 4.1.3 Regulatory and Compliance Context
AuraScan is developed with strict adherence to data privacy regulations. In its production-ready state, it implements:
*   **HIPAA/GDPR Compliance**: All Patient Health Information (PHI) is processed in-memory and can be anonymized locally before any data is logged or used for further model fine-tuning.
*   **Clinical Decision Support (CDS) Guidelines**: The system is designed to meet the criteria for a "Class II Medical Device" software, focusing on augmenting—not replacing—clinician judgment.

---

## 4.2 Product Functions
The AuraScan system is composed of several high-level functional modules that work in a synchronized pipeline to deliver a final diagnostic risk score.

### 4.2.1 Clinical Intake and Risk Engine
This module serves as the initial gatekeeper. Its functions include:
*   **Dynamic Data Collection**: An interactive form that intelligently filters questions based on prior answers (e.g., if "Non-Smoker" is selected, smoking-specific fields are hidden).
*   **Weighted Scoring Calculation**: A proprietary algorithm that assigns weights to clinical markers. For instance, **Hemoptysis** (coughing blood) is assigned a high critical weight (+4 points) due to its strong correlation with advanced malignancy.
*   **Risk Tiering**: Automatically categorizes patients into "Low", "Moderate", or "High" clinical risk buckets even before the image is processed.

### 4.2.2 3D Neural Processing Pipeline
This is the "Brain" of the system, responsible for the deep analysis of CT volumes.
*   **Automated Volume Stacking**: The system automatically identifies, sorts, and stacks individual 2D slices based on their `SliceLocation` and `InstanceNumber` metadata.
*   **Intelligent Voxel Normalization**: Unlike standard image resizing, this function performs 3D interpolation to ensure that each "voxel" (3D pixel) represents a uniform physical dimension (e.g., 1mm x 1mm x 1mm), regardless of the original scanner settings.
*   **Deep Nodule Classification**: The 3D CNN scans the volume to predict the probability of malignancy. It doesn't just look for brightness; it analyzes spatial features like **spiculation** (jagged edges) and **calcification patterns**.

### 4.2.3 Explainability and Visualization (XAI)
To gain clinical trust, the AI must explain its "reasoning."
*   **3D Heatmap Generation**: The system uses Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the specific 32x32x32 regions that influenced the AI’s decision.
*   **Interactive Slice Viewer**: Clinicians can scroll through the original CT stack with the AI's "regions of interest" overlaid in real-time.
*   **Risk Synthesis Dashboard**: A final unified view that presents the Clinical Score and the AI Probability side-by-side, along with a "Confidence Metric" indicating how certain the AI is about its prediction.

---

## 4.3 User Characteristics
The system’s user interface and output complexity are tailored to the specific professional profiles found in a clinical setting.

### 4.3.1 Oncologists and Pulmonologists (The Decision Makers)
*   **Clinical Background**: These users possess 10+ years of specialized training. They are skeptical of "black-box" AI and require evidence.
*   **System Usage**: They use AuraScan for high-stakes decisions (e.g., whether to proceed with a lung biopsy). They rely heavily on the **Integrated Risk Summary** and the **XAI Heatmaps** to correlate AI findings with their own observations of the patient’s clinical symptoms.
*   **Requirement**: High transparency and detailed clinical reporting.

### 4.3.2 Radiologists (The Image Experts)
*   **Clinical Background**: Experts in spatial pattern recognition. They process hundreds of scans daily and are prone to "search pattern fatigue."
*   **System Usage**: They use the tool as a "CADe" (Computer-Aided Detection) system. The AI helps them quickly "triage" normal scans from suspicious ones, allowing them to focus their limited time on the most complex cases.
*   **Requirement**: Low false-positive rates and high-speed image rendering.

### 4.3.3 General Practitioners and Screening Coordinators (The Gatekeepers)
*   **Clinical Background**: Broad medical knowledge but not specialized in oncology. 
*   **System Usage**: They primarily interact with the **Clinical Risk Engine**. They use the tool to determine if a patient qualifies for a referral to a specialist. The system provides them with a "Referral Recommendation" that they can print and include in the patient's file.
*   **Requirement**: Simplicity, ease of data entry, and clear "Next Step" instructions.

### 4.3.4 Medical Residents and Academic Researchers
*   **Clinical Background**: Currently in training or focusing on clinical trials.
*   **System Usage**: They use the system for retrospective studies or for learning. They are interested in the **raw data outputs** (e.g., raw probability scores and BCE loss metrics) to understand the "edge cases" where the AI and clinical history might disagree.
*   **Requirement**: Access to raw metrics and the ability to export diagnostic data for research purposes.

### 4.3.5 Hospital IT and Cybersecurity Personnel
*   **Technical Background**: Non-medical; focused on network integrity and data privacy.
*   **System Usage**: They do not interact with the clinical UI. Their focus is on the **Backend Management Console**, where they monitor API uptimes, data encryption status, and user access logs.
*   **Requirement**: Robust audit trails and low system overhead.

---
*Technical specifications prepared for AuraScan V2 - Full Product Documentation.*
