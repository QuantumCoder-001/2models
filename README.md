# 🏥 AI-Healthcare Integration Suite
> **A Professional System Integration Layer for Predictive Diagnostics**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)

---

## 💎 Project Essence
This repository is a sophisticated **System Integration** project that bridges the gap between raw medical data and clinical intelligence. It features a dual-engine API that transforms complex symptoms and blood biomarkers into accurate disease predictions and personalized recovery protocols.

## 🚀 Core Competencies Demonstrated

### 🔗 System Integration (The "Glue")
The core of this project is the **Integration Layer** in `app.py`, which harmonizes disparate components into a unified service:
* **Multi-Model Orchestration:** Simultaneously manages a Symptom-based Random Forest model and a Blood-Report XGBoost/Classifier.
* **Cross-Origin Synergy:** Implements `Flask-CORS` to allow secure communication between the Python backend and any modern web/mobile frontend.
* **Recommendation Engine:** A hard-coded logic layer that maps 42+ diseases to specific diet and lifestyle "prescriptions".

### 🛠️ Data Engineering Pipeline
Built a robust pipeline to ensure data integrity before it reaches the AI:
* **Feature Vectorization:** Dynamically maps user-selected symptoms to a fixed 133-point binary vector using `symptom_features.json`.
* **Automated Normalization:** Processes 24 distinct blood report variables (Glucose, CRP, Troponin, etc.) into high-dimensional arrays for real-time inference.
* **Persistence Management:** Uses `joblib` for efficient loading and serving of serialized ML models and label encoders.

### ☁️ Cloud & DevOps Readiness
* **Gunicorn Integration:** Optimized for high-concurrency cloud environments (AWS EC2, Google App Engine).
* **Dynamic Infrastructure:** Built-in environment variable support for flexible port binding during cloud deployment.

---

## 🏗️ Architecture Visualization
```text
[ USER INPUT ] ──→ [ API ENDPOINT ] ──→ [ DATA TRANSFORMER ] ──→ [ ML INFERENCE ]
      │                  │                      │                      │
(JSON Symptoms)     (Flask /app.py)      (Feature Mapping)       (Random Forest)
      │                  │                      │                      │
      └──────────────────┴───────────┬──────────┴──────────────────────┘
                                     ▼
                        [ HEALTH RECOMMENDATION ENGINE ]
                        (Diet, Lifestyle, Precautions)
