import numpy as np
import pandas as pd
import joblib
import os

# Get the absolute path of the current directory and model files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = CURRENT_DIR

# Define model file paths
MODEL_FILES = {
    'logistic': os.path.join(MODEL_PATH, 'logistic_model.joblib'),
    'random_forest': os.path.join(MODEL_PATH, 'random_forest_model.joblib')
}

# Column names in the correct order (with spaces removed)
FEATURE_COLUMNS = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY',
    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

def calculate_age_risk(age):
    """Calculate age-based risk factor"""
    if age < 30:
        return 0.0  # Very low risk for young ages
    elif age < 40:
        return 0.2  # Low risk
    elif age < 50:
        return 0.4  # Moderate risk
    elif age < 60:
        return 0.6  # Increased risk
    elif age < 70:
        return 0.8  # High risk
    else:
        return 1.0  # Very high risk

def predict_with_confidence_adjustment(features_df, prediction, probability):
    """Adjust prediction confidence based on age and symptoms"""
    age = features_df['AGE'].iloc[0]
    age_risk = features_df['AGE_RISK'].iloc[0]
    symptom_cols = [col for col in features_df.columns if col not in ['GENDER', 'AGE', 'AGE_RISK']]
    symptom_count = features_df[symptom_cols].sum().sum()
    
    print("\nConfidence adjustment factors:")
    print(f"Age: {age}")
    print(f"Age risk factor: {age_risk}")
    print(f"Number of symptoms present: {symptom_count}")
    
    # For young ages with no symptoms, reduce the confidence of high-risk predictions
    if age_risk < 0.2 and symptom_count == 0 and prediction == 1:
        print("Adjusting prediction: Young age with no symptoms -> Low Risk")
        return 0, 0.15  # Return low risk with low probability
    
    # For young ages with few symptoms, adjust the confidence
    if age_risk < 0.4 and symptom_count <= 2:
        if prediction == 1:
            print("Adjusting confidence: Young age with few symptoms -> Capped probability")
            probability = min(probability, 0.6)  # Cap high risk probability
    
    return prediction, probability

def preprocess_features(features):
    """
    Preprocess features to match training data format.
    Input features are:
    - Gender: 1=Male, 2=Female
    - Age: Original value (will be scaled)
    - Symptoms: 1=No, 2=Yes (will be converted to 0=No, 1=Yes)
    """
    try:
        # Convert to DataFrame with correct column names
        features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        
        # Add age risk factor
        features_df['AGE_RISK'] = features_df['AGE'].apply(calculate_age_risk)
        
        # Convert symptoms from 1/2 to 0/1 (except GENDER and AGE)
        for col in features_df.columns[2:]:  # Skip GENDER and AGE
            if col != 'AGE_RISK':
                features_df[col] = features_df[col].apply(lambda x: 1 if x == 2 else 0)
        
        # Handle missing values if any
        imputer = joblib.load(os.path.join(CURRENT_DIR, 'imputer.pkl'))
        features_df = pd.DataFrame(imputer.transform(features_df), columns=features_df.columns)
        
        # Load and apply scaler to AGE column
        scaler = joblib.load(os.path.join(CURRENT_DIR, 'scaler.pkl'))
        features_df['AGE'] = scaler.transform(features_df[['AGE']])
        
        return features_df

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

def predict(features, algorithm='logistic'):
    """
    Predict lung cancer risk based on input features.
    
    Parameters:
    features (list): List of input features
    algorithm (str): Algorithm to use for prediction ('logistic', 'random_forest')
    
    Returns:
    tuple: (prediction, probability)
    """
    try:
        # Convert features to numpy array and reshape to 2D
        features = np.array(features).reshape(1, -1)
        
        # Preprocess features
        features_df = preprocess_features(features)
        
        # Load the appropriate model based on the algorithm
        if algorithm == 'logistic':
            model_path = MODEL_FILES['logistic']
        elif algorithm == 'random_forest':
            model_path = MODEL_FILES['random_forest']
        else:
            raise ValueError("Unsupported algorithm. Use 'logistic' or 'random_forest'.")
        
        model = joblib.load(model_path)
        prediction = model.predict(features_df)[0]  # Get the first prediction since we have only one sample
        probability = model.predict_proba(features_df)[0][1]  # Get probability of class 1 (High Risk)
        
        # Apply confidence adjustment
        adjusted_prediction, adjusted_probability = predict_with_confidence_adjustment(
            features_df, prediction, probability
        )
        
        return {
            'prediction': adjusted_prediction,
            'probability': adjusted_probability
        }
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            'error': str(e),
            'prediction': None,
            'probability': None
        }
