import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings("ignore")

def load_and_preprocess_data(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Loads the dataset and performs basic preprocessing.
    In a real scenario, this would use a more robust pipeline.
    """
    df = pd.read_csv(data_path)
    
    # Drop irrelevant columns
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Handle TotalCharges missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, dtype=float)
    
    return X_encoded, y

def train_with_mlflow():
    """
    Trains an XGBoost model using SMOTE and tracks experiments using MLflow.
    """
    print("Loading data...")
    X, y = load_and_preprocess_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Set MLflow Experiment
    experiment_name = "Telco_Customer_Churn_Analytics"
    mlflow.set_experiment(experiment_name)
    
    # Define XGBoost hyperparameters
    xgb_params = {
        'learning_rate': 0.05,
        'max_depth': 4,
        'n_estimators': 150,
        'subsample': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    print("Starting MLflow Run...")
    # Start an MLflow run
    with mlflow.start_run(run_name="XGBoost_SMOTE_Tuned"):
        
        # 1. Log Hyperparameters
        mlflow.log_params(xgb_params)
        
        # Initialize and train the model
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # 2. Calculate Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Note: In customer churn, failing to identify a churner (False Negative) 
        # is usually more expensive than misclassifying a loyal customer as a churner (False Positive).
        # Therefore, maximizing 'Recall' is critical to ensure retention teams capture as many 
        # at-risk customers as possible.
        
        # 3. Log Metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        print(f"Metrics - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
        
        # 4. Generate and Log Feature Importance Plot Artifact
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, max_num_features=15, ax=ax, title="Top 15 Feature Importances")
        plt.tight_layout()
        
        artifact_dir = "reports"
        os.makedirs(artifact_dir, exist_ok=True)
        plot_path = os.path.join(artifact_dir, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Log the plot as an artifact
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # Save the expected feature names to align with our backend pipeline
        expected_features = list(X_train.columns)
        features_path = os.path.join(artifact_dir, "expected_features.txt")
        with open(features_path, 'w') as f:
            for feature in expected_features:
                f.write(f"{feature}\n")
        mlflow.log_artifact(features_path, artifact_path="features")
        
        # 5. Log the trained XGBoost model natively
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")
        
        print("MLflow run completed successfully. Model and artifacts logged.")

if __name__ == "__main__":
    train_with_mlflow()
