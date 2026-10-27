# ClinXAI: Transparent Clinical Risk Prediction Using LIME and SHAP
### Third-Year B.Tech Artificial Intelligence and Data Science (Semester V)
**Subject**: Trustworthy and Responsible AI  
**Department**: Artificial Intelligence and Data Science  
**Institution**: Shah & Anchor Kutchhi Engineering College (SAKEC), Chembur, Mumbai  
**Academic Year**: 2026–2027  

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?style=flat&logo=React&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6.svg?style=flat&logo=TypeScript&logoColor=white)](https://www.typescriptlang.org)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-FF6F00.svg?style=flat)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-TabularExplainer-4CAF50.svg?style=flat)](https://github.com/marcotcr/lime)

---

## 1. Project Overview & Clinical Objective

Medical machine learning models often suffer from the "black-box" dilemma: despite demonstrating high predictive performance, their decision mechanisms remain opaque to clinical stakeholders. In healthcare, deploying unexplainable models risks unmonitored bias, algorithmic harm, and diminished clinician trust.

**ClinXAI** is an academic, production-ready educational research platform that predicts diabetes risk from structured patient features and demystifies model reasoning through dual explainable AI (XAI) paradigms:
- **LIME** (Local Interpretable Model-agnostic Explanations): Approximates non-linear decision boundaries locally using perturbed neighbourhood sampling and weighted Ridge regression.
- **SHAP** (SHapley Additive exPlanations): Computes mathematically consistent Shapley values derived from cooperative game theory via TreeSHAP, ensuring the Efficiency Axiom ($\sum \phi_i = f(x) - E[f(x)]$).
- **Comparative XAI Analysis**: Juxtaposes LIME and SHAP to analyze directional agreements, rank variations, and explainability trade-offs.

> **Prominent Clinical Disclaimer**: ClinXAI is developed strictly for educational and academic research purposes within the *Trustworthy and Responsible AI* curriculum. It does not furnish medical diagnoses or therapeutic prescriptions.

---

## 2. System Architecture

ClinXAI is engineered with a decoupled, professional architecture:

```
ClinXAI/
├── frontend/                     # React 19 + TypeScript + Vite + Tailwind CSS + Recharts
│   ├── src/
│   │   ├── components/           # Reusable UI & XAI Visualizers (Waterfall, Diverging Bar, ROC, etc.)
│   │   ├── pages/                # 7 Full Pages (Overview, Predict, LIME, SHAP, Compare, Metrics, About)
│   │   ├── services/             # Centralized API service with live connection telemetry
│   │   ├── data/                 # Authentic held-out test data & case-study profiles
│   │   └── types/                # Strict TypeScript clinical interfaces
├── backend/                      # Python FastAPI ML & XAI Engine
│   ├── main.py                   # FastAPI REST application with CORS & Swagger docs
│   ├── train_model.py            # Reproducible training pipeline (median imputation, 80/20 split)
│   ├── model_service.py          # Random Forest inference & held-out test metrics evaluation
│   ├── explainability.py         # Real TreeSHAP & LIME local surrogate algorithms
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── requirements.txt          # Python dependencies
│   ├── model_artifacts.json      # Evaluated test metrics & global SHAP distribution points
│   └── data/                     # Pima Indians Diabetes dataset
├── notebooks/
│   └── ClinXAI.ipynb             # 14-chapter comprehensive Jupyter Notebook deliverable
├── reports/
│   └── case_study_report.md      # Detailed empirical evaluation across 4 clinical archetypes
└── README.md                     # Academic project documentation and execution guide
```

---

## 3. Quick Start & Execution Guide

### Prerequisites
- **Node.js** (v18 or v20+)
- **Python** (3.10+ recommended)

### Step 1: Clone or Navigate to the Repository
```bash
git clone <repository-url>
cd clinXAI
```

### Step 2: Run the Python FastAPI Backend
Open a terminal and execute:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 train_model.py         # Trains Random Forest and evaluates held-out metrics
uvicorn main:app --reload --port 8000
```
The FastAPI backend will start at `http://localhost:8000`.  
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

### Step 3: Run the Frontend Application
Open a second terminal in the project root:
```bash
npm install
npm run dev
```
The application will launch at `http://localhost:3000`.  
The top navigation header automatically displays real-time connection telemetry to the Python FastAPI backend.

### Step 4: Run the Jupyter Notebook
```bash
cd notebooks
jupyter notebook ClinXAI.ipynb
```

---

## 4. Key Pages & Features

1. **Overview Dashboard**:
   - Four real held-out metrics: Model Accuracy (77.3%), Precision (70.2%), Recall (61.1%), ROC-AUC (0.836).
   - Global SHAP feature importance chart.
   - Interactive Held-Out Confusion Matrix.
   - Clinical workflow overview.

2. **Patient Prediction**:
   - 8 validated clinical input fields (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Pedigree, Age).
   - 4 Instant Case-Study Presets (Typical High-Risk, Decision Threshold Borderline, Discordant Phenotype, Typical Low-Risk).
   - Probability output, risk tier, confidence certainty indicator, and quick links to LIME/SHAP.

3. **LIME Explanation**:
   - Local interpretable surrogate linear regression ($R^2$ fidelity score, intercept).
   - Diverging horizontal bar chart (supporting vs opposing High Risk).
   - Configurable perturbation sample count ($N = 100$ to $2000$).
   - Educational breakdown of local surrogates, kernel bandwidth, and sampling variance.

4. **SHAP Analysis**:
   - TreeSHAP game-theoretic Shapley value attribution.
   - Interactive Waterfall progression: $E[f(x)] \to f(x)$.
   - Global Mean Absolute SHAP feature importance.
   - Summary beeswarm distribution plot showing feature values vs impact.

5. **LIME vs SHAP Comparison**:
   - Side-by-side synchronized contribution charts for the identical patient.
   - Dual-method ranking comparison table with direction agreement flags.
   - Discrepancy analysis: explains mathematical causes for disagreement.
   - LIME stability test across random seeds.

6. **Model Performance**:
   - Held-out test set metrics (Sensitivity, Specificity, F1, ROC-AUC).
   - Interactive Receiver Operating Characteristic (ROC) curve.
   - Medical confusion matrix with clinical cost analysis (False Negatives vs False Positives).

7. **About the Project**:
   - Institutional attribution: Shah & Anchor Kutchhi Engineering College (SAKEC), Mumbai.
   - Course: Trustworthy and Responsible AI, Semester V.
   - Dataset demographic limitations and Pima cohort bias audit.
   - Responsible AI principles and academic references.

---

## 5. REST API Specifications

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend status, Python version, SHAP & LIME availability |
| `GET` | `/api/metrics` | Held-out test set metrics, ROC points, confusion matrix |
| `POST` | `/api/predict` | Computes diabetes risk class and probability for patient input |
| `POST` | `/api/explain/lime` | Calculates LIME local surrogate weights, intercept, and fidelity |
| `POST` | `/api/explain/shap` | Calculates TreeSHAP Shapley values and waterfall steps |
| `GET` | `/api/explain/global` | Global SHAP feature importance and beeswarm summary points |
| `POST` | `/api/explain/compare` | Side-by-side comparison with direction and rank divergence |
| `GET` | `/api/model-info` | Dataset provenance, hyperparameters, and clinical limitations |
| `GET` | `/api/cases` | 4 standardized case-study patient profiles |

---

## 6. Academic References & Citations

1. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 1135–1144.
2. **Lundberg, S. M., & Lee, S.-I.** (2017). *A Unified Approach to Interpreting Model Predictions*. In Advances in Neural Information Processing Systems (NeurIPS 30), pp. 4765–4774.
3. **Lundberg, S. M., et al.** (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2(1), pp. 56–67.
4. **Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S.** (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. In Proceedings of the Symposium on Computer Applications and Medical Care, pp. 261–265.
5. **Doshi-Velez, F., & Kim, B.** (2017). *Towards a rigorous science of interpretable machine learning*. arXiv preprint arXiv:1702.08608.
