import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Model Performance", page_icon="⚙️", layout="wide")

st.title("Model Performance & Evaluation")

# API URL
import os
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.subheader("Model Information")
try:
    response = requests.get(f"{API_URL}/model-info")
    if response.status_code == 200:
        info = response.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Algorithm", info.get("model_type", "Unknown"))
        col2.metric("Features Expected", info.get("features_expected", "Unknown"))
        col3.metric("Backend Status", info.get("status", "Unknown"))
    else:
        st.warning("Backend API is running but returned an error.")
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to the FastAPI backend. Please ensure it is running on port 8000.")

st.markdown("---")

@st.cache_data
def load_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model_path = os.path.join(base_dir, "models", "churn_model.pkl")
    preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    
    if os.path.exists(data_path) and os.path.exists(model_path) and os.path.exists(preprocessor_path):
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        
        # Split features and target
        X = df.drop(columns=['customerID', 'Churn'])
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        model = joblib.load(model_path)
        expected_features = joblib.load(preprocessor_path)
        
        X_processed = pd.get_dummies(X, dtype=float)
        for col in expected_features:
            if col not in X_processed.columns:
                X_processed[col] = 0.0
        X_processed = X_processed[expected_features]
        
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1]
        
        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "ROC AUC": roc_auc_score(y, y_prob)
        }
        
        return metrics, y, y_pred, y_prob
    return None

results = load_and_evaluate()

if results:
    metrics, y_true, y_pred, y_prob = results
    
    st.subheader("Evaluation Metrics (On Full Dataset)")
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    m_col2.metric("Precision", f"{metrics['Precision']:.3f}")
    m_col3.metric("Recall", f"{metrics['Recall']:.3f}")
    m_col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
    m_col5.metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
    
    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted Label", y="True Label"),
                           x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_c2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        df_roc = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
        fig_roc = px.line(df_roc, x='False Positive Rate', y='True Positive Rate',
                          title=f'ROC Curve (AUC={metrics["ROC AUC"]:.3f})')
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

else:
    st.info("Dataset or models not found locally. Evaluation metrics cannot be generated. Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the 'data/' folder to view actual metrics.")
