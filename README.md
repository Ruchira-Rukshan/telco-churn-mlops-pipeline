<div align="center">
  <h1>📡 Telco Customer Churn Analytics</h1>
  <p>An End-to-End MLOps Pipeline and Prediction Dashboard</p>

  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-191A1C?style=for-the-badge)](#)
  [![Evidently_AI](https://img.shields.io/badge/Evidently_AI-FF6F00?style=for-the-badge)](#)
</div>

---

## 🌟 Overview

**Telco Customer Churn Analytics** is a comprehensive, production-ready Data Science application designed to predict and analyze customer churn. By bridging state-of-the-art machine learning models with a robust backend architecture and a highly interactive frontend, this platform empowers retention teams to make data-driven decisions.

It natively integrates advanced MLOps practices, utilizing Artificial Intelligence (XGBoost + SMOTE) for churn predictions, **MLflow** for experiment tracking, and **Evidently AI** for automated Data Drift detection to ensure the model remains reliable in production.

## 🚀 Key Modules & Features

The platform is meticulously designed to serve both technical data scientists and business analysts. The following key modules constitute the ecosystem:

| Module | Core Functionalities |
| ------ | -------------------- |
| 📈 **Analytics (EDA)** | Interactive dashboards visualizing customer demographics, contract types, and financial distributions. |
| 🎯 **Model Performance** | Real-time evaluation metrics (Accuracy, ROC AUC, F1 Score) and confusion matrices pulled directly from the backend. |
| 👤 **Single Prediction** | Customer service interface to input individual customer details and instantly calculate their churn probability. |
| 📁 **Bulk Prediction** | Batch processing module for marketing teams to upload CSVs, returning grouped risk assessments and downloadable reports. |
| 🧠 **Explainability** | SHAP-powered Global and Local explanations to demystify "black-box" model predictions and highlight driving features. |
| 🛡️ **Data Drift Detection** | Automated monitoring using Evidently AI to flag statistically anomalous data during bulk predictions and generate HTML reports. |
| 🧪 **MLflow Tracking** | Professional experiment tracking for model versioning, hyperparameter logging, and metric comparisons. |

## 🎯 Model Performance & Metrics

Our core predictive engine is powered by an **XGBClassifier** trained on data that was synthetically balanced using **SMOTE** (Synthetic Minority Over-sampling Technique).

> [!IMPORTANT]
> **Business Optimization:** We heavily prioritized **Recall** over Accuracy when tuning this model. In customer retention strategies, minimizing false negatives (missing actual churners) is the most expensive business risk. Maximizing Recall ensures we capture and flag the highest possible number of true churn risks for intervention.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Recall (Churners)** | **~70%** | Percentage of actual churners successfully identified. |
| **ROC-AUC** | **~0.86** | Overall ability of the model to distinguish between classes. |
| **F1-Score** | **~0.64** | Harmonic mean of Precision and Recall. |
| **Accuracy** | **~79%** | Overall correct predictions across both classes. |
## 🛠️ Technology Stack

Our platform is constructed using modern, scalable, and robust data science and web technologies:
* **Frontend:** Streamlit, Plotly Express, Pandas.
* **Backend:** FastAPI, Uvicorn, Pydantic, Python 3.
* **Machine Learning:** XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE), SHAP.
* **MLOps & Monitoring:** MLflow, Evidently AI.
* **Data Processing:** NumPy, Pandas.

## 🗂️ Project Structure

The repository is modularized into feature-specific components. Each fundamental service is neatly decoupled:
```text
telco-churn-mlops-pipeline/
├── 📁 data/           # Place WA_Fn-UseC_-Telco-Customer-Churn.csv here
├── 📁 models/         # Saved Machine Learning pipelines (churn_model.pkl)
├── 📁 backend/        # FastAPI Core REST API Servers & Drift Detection
├── 📁 dashboard/      # Streamlit User Interfaces (Analytics & Portals)
├── 📁 reports/        # Generated HTML Drift reports and metrics
└── 📄 README.md       # Project documentation
```

## ⚙️ Getting Started

To run the full suite locally, you will need to construct the environment and spin up all the services.

### 1️⃣ Dependencies & Dataset Setup
1. Clone the repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
3. Place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file inside the `data/` folder.

### 2️⃣ Backend Execution (FastAPI)
Open a terminal and start the REST API server to serve the model and drift detection.
```bash
cd backend
py -m uvicorn main:app --reload
```
The server will boot up and handle API traffic via `http://127.0.0.1:8000`.

### 3️⃣ Frontend Execution (Streamlit)
Open a **new** terminal and fire up the dynamic UI interfaces.
```bash
cd dashboard
py -m streamlit run app.py
```
The application will be accessible via your browser at `http://localhost:8501/`.

### 4️⃣ MLOps Execution (MLflow UI)
Open a **new** terminal in the root directory to launch the experiment tracking dashboard (Optional but recommended).
```bash
py -m mlflow ui --port 5000
```
The MLflow dashboard will run seamlessly on `http://127.0.0.1:5000/`.

---

<div align="center">
  <i>Developed and Maintained by Ruchira Rukshan.</i>
</div>
