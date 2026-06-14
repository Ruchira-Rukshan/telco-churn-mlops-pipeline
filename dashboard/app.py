import streamlit as st

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Customer Churn Analytics & Prediction Dashboard")

st.markdown("""
### Welcome to the Telco Customer Churn Dashboard!

This application allows business analysts, customer retention teams, and data scientists to:
- **Analyze** customer churn behavior through interactive EDA (Analytics).
- **Evaluate** machine learning model performance (Model Performance).
- **Predict** churn for a single customer in real-time (Single Prediction).
- **Predict** churn for multiple customers via CSV upload (Bulk Prediction).
- **Understand** why a customer is likely to churn using explainable AI (Explainability).

Please navigate through the pages on the left sidebar to explore the features.
""")

st.info("Ensure the FastAPI backend is running before making predictions.")
