import streamlit as st
import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")

st.title("Model Explainability (SHAP)")
st.markdown("Understand how the machine learning model makes predictions using Shapley Additive exPlanations (SHAP).")

@st.cache_data
def load_data_and_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model_path = os.path.join(base_dir, "models", "churn_model.pkl")
    preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    
    if os.path.exists(data_path) and os.path.exists(model_path) and os.path.exists(preprocessor_path):
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        
        # Sample data for SHAP to avoid long computation times
        df_sample = df.sample(min(500, len(df)), random_state=42)
        X = df_sample.drop(columns=['customerID', 'Churn'])
        
        model = joblib.load(model_path)
        expected_features = joblib.load(preprocessor_path)
        
        X_processed = pd.get_dummies(X, dtype=float)
        for col in expected_features:
            if col not in X_processed.columns:
                X_processed[col] = 0.0
        X_processed = X_processed[expected_features]
        
        # Features are already aligned, get names
        feature_names = expected_features
            
        # Create Explainer
        if hasattr(model, 'predict_proba'):
            # For tree-based models (XGBoost, Random Forest, etc.)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_processed)
            except:
                # Fallback to KernelExplainer
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_processed, 50))
                shap_values = explainer.shap_values(X_processed)[1] # For class 1 (Churn)
        else:
            explainer = shap.Explainer(model, X_processed)
            shap_values = explainer(X_processed)
            
        return df_sample, X_processed, feature_names, explainer, shap_values
    return None

results = load_data_and_model()

if results:
    df_sample, X_processed, feature_names, explainer, shap_values = results
    
    st.subheader("Global Explanations")
    st.markdown("These plots show the overall importance of each feature across all customers.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**SHAP Summary Plot (Beeswarm)**")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Adjust depending on shap_values format (binary classification usually returns list or single array)
        if isinstance(shap_values, list):
            sv = shap_values[1] # Usually index 1 is positive class
        else:
            sv = shap_values
            
        shap.summary_plot(sv, X_processed, feature_names=feature_names, show=False)
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Feature Importance (Bar)**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, X_processed, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig2)
        
    st.markdown("---")
    
    st.subheader("Local Explanations")
    st.markdown("Select a customer from the sample to see how features influenced their specific prediction.")
    
    customer_indices = df_sample.index.tolist()
    selected_idx = st.selectbox("Select Customer Index", customer_indices)
    
    if st.button("Generate Explanation"):
        idx_pos = customer_indices.index(selected_idx)
        
        st.markdown(f"**Explanation for Customer Index: {selected_idx}**")
        
        # Waterfall Plot
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # Waterfall plot requires an Explanation object
        if not isinstance(shap_values, shap.Explanation):
            # Create a mock Explanation object for plotting
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
            explanation = shap.Explanation(values=sv[idx_pos], 
                                           base_values=expected_value, 
                                           data=X_processed.iloc[idx_pos], 
                                           feature_names=feature_names)
            shap.waterfall_plot(explanation, show=False)
        else:
            shap.waterfall_plot(shap_values[idx_pos], show=False)
            
        st.pyplot(fig3)
        
        st.info("Red bars push the prediction higher (more likely to churn). Blue bars push the prediction lower (less likely to churn).")

else:
    st.warning("Dataset or models not found. Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the 'data/' folder to generate SHAP explanations.")
