# 5. System Design and Architecture

This section provides the structural and behavioral diagrams for the AuraScan (Lung Cancer V2) system. To view or edit these diagrams, copy the code blocks below into the [Mermaid Live Editor](https://mermaid.live/).

---

### 5.1.1 Proposed System
The proposed system integrates clinical risk scoring with 3D Deep Learning image analysis. It transitions from a manual, subjective interpretation to an augmented, data-driven diagnostic workflow.

```mermaid
graph TD
    A[Patient Entry] --> B[Clinical Risk Intake]
    A --> C[CT Scan Upload]
    B --> D[Clinical Logic Engine]
    C --> E[3D CNN Analysis]
    D --> F[Integrated Risk Synthesis]
    E --> F
    F --> G[Explainable Heatmaps]
    F --> H[Final Diagnostic Report]
```

---

### 5.1.2 Block Diagram
A high-level view of the functional blocks within the system.

```mermaid
graph LR
    subgraph Frontend
    UI[Web Dashboard]
    end
    subgraph Processing_Layer
    CR[Clinical Risk Logic]
    DL[3D CNN Inference]
    PP[Image Preprocessing]
    end
    subgraph Data_Layer
    DB[(Patient Records)]
    MOD[Model Checkpoints]
    end
    UI <--> CR
    UI <--> PP
    PP --> DL
    DL <--> MOD
    CR <--> DB
```

---

### 5.1.3 Component Diagram
Detailed software components and their interfaces.

```mermaid
component "AuraScan Core" {
  [UI Controller] -- "Data Objects" --> [Logic Hub]
  [Logic Hub] --> [Risk Scorer]
  [Logic Hub] --> [AI Pipeline]
  [AI Pipeline] --> [DICOM Handler]
  [AI Pipeline] --> [3D CNN Model]
  [3D CNN Model] --> [XAI Engine]
}
```

---

### 5.1.4 Use Case Diagram
Interaction between clinical actors and the system.

```mermaid
useCaseDiagram
    actor Clinician
    actor "Hospital Admin" as Admin
    actor "AI Model" as AI
    
    package "AuraScan System" {
        usecase "Input Patient Data" as UC1
        usecase "Upload CT Scan" as UC2
        usecase "View Risk Analysis" as UC3
        usecase "Generate Report" as UC4
        usecase "Audit Logs" as UC5
    }
    
    Clinician --> UC1
    Clinician --> UC2
    Clinician --> UC3
    Clinician --> UC4
    UC2 --> AI
    AI --> UC3
    Admin --> UC5
```

---

### 5.1.5 Data Flow Diagram (DFD)
The flow of information from raw input to diagnostic output.

```mermaid
graph LR
    P((Patient)) -- Clinical Data --> RE[Risk Engine]
    S((CT Scanner)) -- DICOM Volume --> IP[Image Processor]
    RE -- Weighted Score --> RS[Risk Synthesis]
    IP -- Resampled Voxels --> AI[AI Classifier]
    AI -- Probability Score --> RS
    RS -- Report --> DR((Doctor))
```

---

### 5.1.6 Class Diagram
The underlying data structures and class relationships in the Python implementation.

```mermaid
classDiagram
    class Patient {
        +String name
        +int age
        +float packYears
        +calculateClinicalScore()
    }
    class NoduleClassifier3D {
        +Sequential conv_layers
        +Linear fc_layers
        +forward(tensor)
    }
    class DICOMHandler {
        +load_volume(path)
        +normalize_hu()
        +resample_3d()
    }
    class DiagnosticReport {
        +float combined_risk
        +Image heatmap
        +generate_pdf()
    }
    Patient --> DiagnosticReport
    DICOMHandler --> NoduleClassifier3D
    NoduleClassifier3D --> DiagnosticReport
```

---

### 5.1.7 Sequence Diagram
The step-by-step diagnostic workflow for a single patient session.

```mermaid
sequenceDiagram
    participant D as Doctor
    participant UI as Web Dashboard
    participant LE as Logic Engine
    participant PP as Image Processor
    participant AI as 3D CNN Model
    
    D->>UI: Input Patient Clinical Data
    UI->>LE: Calculate Base Risk
    LE-->>UI: Return Clinical Score
    D->>UI: Upload CT Volume (DICOM)
    UI->>PP: Process & Resample Volume
    PP->>AI: Predict Malignancy
    AI-->>UI: Return Probability & Heatmap
    UI->>D: Show Integrated Risk Dashboard
```

---
*Diagrams generated for the AuraScan (Lung Cancer V2) technical specification.*
