# Customer Churn Analytics & Prediction Dashboard

This repository contains an end-to-end Data Science web application for analyzing and predicting customer churn, based on the Telco Customer Churn dataset.

## Project Architecture

The project consists of two main components:
1. **FastAPI Backend:** Provides REST API endpoints for model inference (Single Prediction & Bulk Prediction).
2. **Streamlit Frontend:** An interactive dashboard providing EDA, Model Evaluation metrics, and Explainable AI (SHAP) visualizations.

## Directory Structure

```text
customer-churn-dashboard/
├── data/                      # Place WA_Fn-UseC_-Telco-Customer-Churn.csv here
├── notebooks/                 # Jupyter notebooks for EDA and model training
├── models/                    # Saved Machine Learning models (churn_model.pkl, preprocessor.pkl)
├── backend/
│   └── main.py                # FastAPI application
├── dashboard/
│   ├── app.py                 # Streamlit main entry point
│   └── pages/                 # Streamlit multi-page structure
├── reports/                   # Generated reports and metrics
├── screenshots/               # Dashboard screenshots
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup & Installation

### 1. Prerequisites
- Python 3.8 or higher.
- `pip` package manager.

### 2. Install Dependencies
Run the following command to install all required packages:
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
For the Analytics, Model Performance, and Explainability pages to work fully, download the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file inside the `data/` folder.

## Running the Application

To run the full stack, you need to start both the FastAPI backend and the Streamlit frontend.

### 1. Start the FastAPI Backend
Open a terminal, navigate to the `backend/` directory, and start the server:
```bash
cd backend
py -m uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can view the API documentation at `http://127.0.0.1:8000/docs`.

### 2. Start the Streamlit Dashboard
Open a **new** terminal, navigate to the `dashboard/` directory, and start the dashboard:
```bash
cd dashboard
py -m streamlit run app.py
```
The dashboard will open in your default web browser at `http://localhost:8501`.

### 3. Start the MLflow UI (Optional)
To view experiment tracking and model metrics, open a **new** terminal in the root `D:\Churn Prediction` folder:
```bash
py -m mlflow ui --port 5000
```
Then navigate to `http://127.0.0.1:5000` in your browser.

## Features
- **Analytics:** Interactive Plotly charts showing customer distributions, churn rates, and feature correlations.
- **Model Performance:** Detailed metrics (Accuracy, ROC AUC, F1 Score) evaluated on the dataset.
- **Single Prediction:** A comprehensive form to predict churn probability for an individual customer.
- **Bulk Prediction:** Upload a CSV of customers to receive batch predictions and a downloadable summary report.
- **Explainability:** SHAP summary plots and local waterfall plots explaining the feature impacts on predictions.
