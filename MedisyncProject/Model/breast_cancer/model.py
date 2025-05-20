import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC
from .Breast_cancer_prediction_final import predict as predict_internal
import os

# Get the absolute path of the current directory and model files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = CURRENT_DIR
DATA_PATH = os.path.join(CURRENT_DIR, 'data')

# Ensure the directories exist
os.makedirs(DATA_PATH, exist_ok=True)

# Define model file paths
MODEL_FILES = {
    'svm_rbf': os.path.join(MODEL_PATH, 'svm_rbf_model.joblib'),
    'svm_linear': os.path.join(MODEL_PATH, 'svm_linear_model.joblib'),
    'knn': os.path.join(MODEL_PATH, 'knn_model.joblib'),
    'decision_tree': os.path.join(MODEL_PATH, 'decision_tree_model.joblib'),
    'random_forest': os.path.join(MODEL_PATH, 'random_forest_model.joblib'),
    'logistic': os.path.join(MODEL_PATH, 'logistic_model.joblib')
}

# Data file paths
DATASET_PATH = os.path.join(DATA_PATH, 'breast-cancer-wisconsin-data.csv')
SCALER_PATH = os.path.join(MODEL_PATH, 'scaler.joblib')

def load_model():
    """Load the trained breast cancer prediction model."""
    try:
        model = joblib.load(os.path.join(CURRENT_DIR, 'breast_cancer_model.joblib'))
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except:
        print("Error loading model files")
        return None, None

def predict(features, algorithm='svm_rbf'):
    """
    Make a breast cancer prediction using the specified algorithm.
    
    Args:
        features (list): List of feature values
        algorithm (str): Algorithm to use for prediction
        
    Returns:
        dict: Prediction results including prediction and probability
    """
    try:
        # Convert features to numpy array
        features = np.array(features).reshape(1, -1)
        
        # Load the scaler and transform features
        scaler = joblib.load(SCALER_PATH)
        features_scaled = scaler.transform(features)
        
        # Load the appropriate model
        if algorithm not in MODEL_FILES:
            raise ValueError(f"Unsupported algorithm. Use one of: {', '.join(MODEL_FILES.keys())}")
            
        model = joblib.load(MODEL_FILES[algorithm])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probability scores if the model supports it
        try:
            probability = model.predict_proba(features_scaled)[0]
            prob_malignant = probability[1] if prediction == 1 else probability[0]
        except:
            prob_malignant = None
        
        return {
            'prediction': prediction,
            'probability': prob_malignant,
            'error': None
        }
        
    except Exception as e:
        return {
            'prediction': None,
            'probability': None,
            'error': str(e)
        }

def predict_internal(features, algorithm='svm'):
    """
    Wrapper function for breast cancer prediction.
    Args:
        features: List of numerical features for prediction
        algorithm: String indicating which algorithm to use ('svm', 'random_forest', or 'logistic')
    Returns:
        Dictionary containing prediction results
    """
    return predict_internal(features, algorithm) 