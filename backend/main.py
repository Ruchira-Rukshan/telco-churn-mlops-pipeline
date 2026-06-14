from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import os
from datetime import datetime
import logging
from drift_detector import check_data_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Churn Prediction API")

# Path to models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# Load models
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    preprocessor = None

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "preprocessor_loaded": preprocessor is not None}

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = []
    if isinstance(preprocessor, list):
        features = preprocessor
            
    return {
        "model_type": str(type(model).__name__),
        "features_expected": len(features) if features else "Unknown",
        "status": "Ready for inference"
    }

def process_and_predict(df: pd.DataFrame):
    # Ensure TotalCharges is numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])
    
    # Preprocess (One-Hot Encoding)
    try:
        X_processed = pd.get_dummies(df, dtype=float)
        
        # Align with expected features from preprocessor
        expected_features = preprocessor
        for col in expected_features:
            if col not in X_processed.columns:
                X_processed[col] = 0.0
                
        # Reorder and filter columns
        X_processed = X_processed[expected_features]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
    # Predict
    try:
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    return predictions, probabilities

@app.post("/predict")
def predict_single(customer: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # Convert input to DataFrame
    df = pd.DataFrame([customer.dict()])
    
    preds, probs = process_and_predict(df)
    
    prob = float(probs[0])
    pred = int(preds[0])
    
    risk_category = "Low"
    if prob > 0.7:
        risk_category = "High"
    elif prob > 0.4:
        risk_category = "Medium"
        
    return {
        "prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(prob * 100, 2),
        "risk_category": risk_category
    }

@app.post("/bulk-predict")
async def predict_bulk(file: UploadFile = File(...)):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        df = pd.read_csv(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
        
    # Data Drift Detection (Non-blocking)
    drift_metrics = {
        "drift_detected": False,
        "drifted_features_share_percentage": 0.0,
        "html_report_saved_at": "Not Generated",
        "status": "failed",
        "message": "Drift detection failed or was skipped."
    }
    
    try:
        reference_path = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        logger.info("Running Data Drift Detection...")
        drift_result = check_data_drift(reference_path=reference_path, current_df=df.copy())
        
        drift_metrics = {
            "drift_detected": drift_result["drift_detected"],
            "drifted_features_share_percentage": round(drift_result["drifted_features_share"] * 100, 2),
            "html_report_saved_at": drift_result["html_report_path"],
            "status": "success"
        }
        logger.info("Data Drift Detection completed successfully.")
    except Exception as e:
        logger.error(f"Data Drift Detection Failed: {e}")
        drift_metrics["message"] = f"Drift detection failed: {str(e)}"
        
    # Process & Predict
    try:
        # Keep original df for returning results
        result_df = df.copy()
        
        # Predict
        preds, probs = process_and_predict(df)
        
        result_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in preds]
        result_df['Churn_Probability'] = probs
        
        def get_risk(prob):
            if prob > 0.7: return "High"
            if prob > 0.3: return "Medium"
            return "Low"
            
        result_df['Risk_Category'] = [get_risk(p) for p in probs]
        
        # Generate summary
        total_records = len(result_df)
        churn_count = sum(preds == 1)
        
        prediction_summary = {
            "total_records": total_records,
            "predicted_churn_customers": int(churn_count),
            "churn_percentage": round((churn_count / total_records) * 100, 2) if total_records > 0 else 0,
        }
        
        return {
            "status": "success",
            "data_drift_analysis": drift_metrics,
            "prediction_summary": prediction_summary,
            "predictions": result_df.to_dict(orient="records")
        }
        
    except Exception as e:
        logger.error(f"Bulk Prediction Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing bulk prediction: {str(e)}")
