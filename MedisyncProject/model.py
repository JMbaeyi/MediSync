import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class LungCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            # Get the current directory and construct the correct model path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'lung_cancer', 'model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = joblib.load(model_path)
            print("\nModel loaded successfully")
            print("Model type:", type(self.model))
            return True
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            raise

    def preprocess_features(self, features):
        """Preprocess features to match training data"""
        processed = features.copy()
        
        # No need to convert gender or symptoms since they already match training data
        # Gender is already: 1 (Male), 2 (Female)
        # Symptoms are already: 1 (No), 2 (Yes)
        
        # Only scale age to match training data range
        processed[1] = (processed[1] - 0) / (120 - 0)
            
        return processed

    def predict(self, symptoms):
        """Make a prediction"""
        try:
            # Print input symptoms for debugging
            print("\nOriginal input symptoms:")
            feature_names = ['Gender', 'Age', 'Smoking', 'Yellow Fingers', 'Anxiety', 
                           'Peer Pressure', 'Chronic Disease', 'Fatigue', 'Allergy',
                           'Wheezing', 'Alcohol Consuming', 'Coughing', 
                           'Shortness of Breath', 'Swallowing Difficulty', 'Chest Pain']
            for name, value in zip(feature_names, symptoms):
                print(f"{name}: {value}")
            
            # Validate number of features
            if len(symptoms) != 15:
                raise ValueError(f"Expected 15 features, got {len(symptoms)}")
            
            # Preprocess features
            processed_features = self.preprocess_features(symptoms)
            print("\nProcessed features:")
            for name, value in zip(feature_names, processed_features):
                print(f"{name}: {value}")
            
            # Reshape for prediction
            features = np.array(processed_features, dtype=float).reshape(1, -1)
            print("\nReshaped features:", features)
            print("Features shape:", features.shape)
            print("Features dtype:", features.dtype)
            
            # Make prediction
            prediction = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            print("\nRaw prediction:", prediction)
            print("Probabilities:", probabilities)
            
            # Return prediction and probability
            return {
                'prediction': int(prediction[0]),
                'probability': float(probabilities[0][1])  # Probability of high risk
            }
            
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
            raise

# Create a singleton instance
predictor = LungCancerPredictor()

def predict(symptoms):
    """Wrapper function for prediction"""
    return predictor.predict(symptoms)
