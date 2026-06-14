import streamlit as st
import requests

st.set_page_config(page_title="Single Prediction", page_icon="👤", layout="wide")

st.title("Single Customer Churn Prediction")
st.markdown("Enter the customer details below to predict their likelihood of churning.")

API_URL = "http://127.0.0.1:8000"

with st.form("prediction_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    gender = col1.selectbox("Gender", ["Male", "Female"])
    senior_citizen = col2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = col1.selectbox("Partner", ["Yes", "No"])
    dependents = col2.selectbox("Dependents", ["Yes", "No"])
    
    st.subheader("Account Information")
    col3, col4, col5 = st.columns(3)
    tenure = col3.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    contract = col4.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = col5.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = col3.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = col4.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = col5.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    
    st.subheader("Services")
    col6, col7, col8 = st.columns(3)
    phone_service = col6.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = col7.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = col8.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    online_security = col6.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = col7.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = col8.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    
    tech_support = col6.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = col7.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = col8.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                # Determine color based on risk
                risk = result['risk_category']
                color = "green" if risk == "Low" else "orange" if risk == "Medium" else "red"
                
                res_col1.metric("Will Churn?", result['prediction'])
                res_col2.metric("Churn Probability", f"{result['churn_probability']}%")
                
                st.markdown(f"### Risk Category: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
                
            else:
                st.error(f"Error from backend: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the FastAPI backend. Please ensure it is running on port 8000.")
