from .lung_cancer.model import predict as predict_lung_cancer
from .breast_cancer.model import predict as predict_breast_cancer

def predict(features, model_type, algorithm='svm'):
    """
    Wrapper function to make predictions using either lung or breast cancer models.
    
    Args:
        features (list): List of features for prediction
        model_type (str): Either 'lung' or 'breast'
        algorithm (str): Algorithm to use for prediction (default: 'svm')
        
    Returns:
        dict: Prediction results including prediction, probability, and algorithm info
    """
    if model_type.lower() == 'lung':
        return predict_lung_cancer(features, algorithm)
    elif model_type.lower() == 'breast':
        return predict_breast_cancer(features, algorithm)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 