import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

def check_data_drift(reference_path: str, current_df: pd.DataFrame) -> dict:
    """
    Compares the incoming data against the reference baseline to detect data drift.
    
    Args:
        reference_path (str): File path to the baseline training CSV dataset.
        current_df (pd.DataFrame): Incoming production DataFrame for bulk prediction.
        
    Returns:
        dict: A dictionary containing drift metrics and the path to the HTML report.
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found at {reference_path}")
        
    # Load baseline data
    ref_df = pd.read_csv(reference_path)
    
    # Drop targets and identifiers
    for col in ['Churn', 'customerID']:
        if col in ref_df.columns:
            ref_df = ref_df.drop(columns=[col])
        if col in current_df.columns:
            current_df = current_df.drop(columns=[col])
            
    # Handle TotalCharges if it's stored as strings with spaces
    if 'TotalCharges' in ref_df.columns:
        ref_df['TotalCharges'] = pd.to_numeric(ref_df['TotalCharges'], errors='coerce').fillna(0.0)
    if 'TotalCharges' in current_df.columns:
        current_df['TotalCharges'] = pd.to_numeric(current_df['TotalCharges'], errors='coerce').fillna(0.0)
        
    # Sync columns dynamically: Evaluate only common columns
    common_columns = list(set(ref_df.columns).intersection(set(current_df.columns)))
    
    if not common_columns:
        raise ValueError("No common columns found between reference and current dataframes.")
        
    ref_df = ref_df[common_columns]
    current_df = current_df[common_columns]
    
    # Initialize Evidently Report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    
    # Run the drift calculation
    snapshot = data_drift_report.run(reference_data=ref_df, current_data=current_df)
    
    # Extract metrics from the snapshot dictionary
    report_dict = snapshot.dict()
    
    drift_share = 0.0
    drift_detected = False
    
    # The structure of the dictionary varies depending on evidently versions, 
    # but generally DataDriftPreset populates the first metric with DriftedColumnsCount
    # Find the metric with metric_name starting with 'DriftedColumnsCount'
    drift_metric = next((m for m in report_dict['metrics'] if m['metric_name'].startswith('DriftedColumnsCount')), None)
    
    if drift_metric:
        drift_share = drift_metric['value'].get('share', 0.0)
        drift_threshold = drift_metric['config'].get('drift_share', 0.5)
        drift_detected = drift_share >= drift_threshold
    
    # Save the HTML report
    # Assumes the script runs from the backend/ directory or root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, "data_drift_report.html")
    snapshot.save_html(report_path)
    
    return {
        "drift_detected": bool(drift_detected),
        "drifted_features_share": float(drift_share),
        "html_report_path": report_path
    }
