import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import io

st.set_page_config(page_title="Bulk Prediction", page_icon="📁", layout="wide")

st.title("Bulk Customer Churn Prediction")
st.markdown("Upload a CSV file containing multiple customer records to predict their churn probabilities simultaneously.")

API_URL = "http://127.0.0.1:8000"

st.info("The CSV must contain the standard Telco Customer Churn dataset columns.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the file to display a preview
    df_preview = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df_preview.head())
    
    if st.button("Run Bulk Prediction"):
        with st.spinner("Processing file..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                response = requests.post(f"{API_URL}/bulk-predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    drift_data = data["data_drift_analysis"]
                    summary = data["prediction_summary"]
                    results_df = pd.DataFrame(data["predictions"])
                    
                    st.success("Prediction complete!")
                    
                    st.subheader("Data Drift Analysis (Evidently AI)")
                    drift_col1, drift_col2, drift_col3 = st.columns(3)
                    
                    drift_status = "⚠️ Detected" if drift_data["drift_detected"] else "✅ No Drift"
                    drift_col1.metric("Dataset Drift Status", drift_status)
                    drift_col2.metric("Drifted Features Share", f"{drift_data['drifted_features_share_percentage']}%")
                    if drift_data["status"] == "success":
                        drift_col3.info(f"Report saved locally to: \n`{drift_data['html_report_saved_at']}`")
                    else:
                        drift_col3.warning(drift_data.get("message", "Drift detection failed."))
                        
                    st.markdown("---")
                    
                    st.subheader("Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Records Analyzed", summary["total_records"])
                    col2.metric("Predicted to Churn", summary["predicted_churn_customers"])
                    col3.metric("Churn Rate", f"{summary['churn_percentage']}%")
                    
                    st.markdown("---")
                    
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        fig_risk = px.pie(results_df, names="Risk_Category", title="Risk Distribution",
                                          color="Risk_Category", 
                                          color_discrete_map={"High":"red", "Medium":"orange", "Low":"green"})
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                    with col_chart2:
                        fig_prob = px.histogram(results_df, x="Churn_Probability", title="Churn Probability Distribution", nbins=20)
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    st.subheader("Detailed Results")
                    # Show important columns
                    display_cols = []
                    if 'customerID' in results_df.columns:
                        display_cols.append('customerID')
                    display_cols.extend(['Churn_Prediction', 'Churn_Probability', 'Risk_Category'])
                    st.dataframe(results_df[display_cols].head(100))
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Predictions CSV",
                        data=csv,
                        file_name="churn_predictions_results.csv",
                        mime="text/csv",
                    )
                    
                else:
                    st.error(f"Error from backend: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the FastAPI backend. Please ensure it is running on port 8000.")
