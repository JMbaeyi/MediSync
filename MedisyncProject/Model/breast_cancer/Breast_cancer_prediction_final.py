import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess_data():
    """Load and preprocess the breast cancer dataset."""
    # Load the dataset
    data_path = os.path.join(current_dir, 'breast-cancer-wisconsin-data.csv')
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    df = df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    
    # Encode diagnosis column (M = 1, B = 0)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df['diagnosis'] = labelencoder.fit_transform(df['diagnosis'])
    
    # Select all features except diagnosis
    feature_columns = [col for col in df.columns if col != 'diagnosis']
    X = df[feature_columns]
    y = df['diagnosis']
    
    return X, y, feature_columns

def train_models():
    """Train and save the machine learning models."""
    # Load and preprocess data
    X, y, feature_columns = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(current_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Train and save SVM RBF model
    svm_rbf_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_rbf_model.fit(X_train_scaled, y_train)
    svm_rbf_path = os.path.join(current_dir, 'svm_rbf_model.joblib')
    joblib.dump(svm_rbf_model, svm_rbf_path)
    print(f"SVM RBF Accuracy: {svm_rbf_model.score(X_test_scaled, y_test):.4f}")
    
    # Train and save SVM Linear model
    svm_linear_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_linear_model.fit(X_train_scaled, y_train)
    svm_linear_path = os.path.join(current_dir, 'svm_linear_model.joblib')
    joblib.dump(svm_linear_model, svm_linear_path)
    print(f"SVM Linear Accuracy: {svm_linear_model.score(X_test_scaled, y_test):.4f}")
    
    # Train and save KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    knn_path = os.path.join(current_dir, 'knn_model.joblib')
    joblib.dump(knn_model, knn_path)
    print(f"KNN Accuracy: {knn_model.score(X_test_scaled, y_test):.4f}")
    
    # Train and save Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_path = os.path.join(current_dir, 'decision_tree_model.joblib')
    joblib.dump(dt_model, dt_path)
    print(f"Decision Tree Accuracy: {dt_model.score(X_test_scaled, y_test):.4f}")
    
    # Train and save Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_path = os.path.join(current_dir, 'random_forest_model.joblib')
    joblib.dump(rf_model, rf_path)
    print(f"Random Forest Accuracy: {rf_model.score(X_test_scaled, y_test):.4f}")
    
    # Train and save Logistic Regression model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_path = os.path.join(current_dir, 'logistic_model.joblib')
    joblib.dump(lr_model, lr_path)
    print(f"Logistic Regression Accuracy: {lr_model.score(X_test_scaled, y_test):.4f}")

def load_model(algorithm='svm_rbf'):
    """Load the specified model and scaler."""
    try:
        # Load scaler
        scaler_path = os.path.join(current_dir, 'scaler.joblib')
        scaler = joblib.load(scaler_path)
        
        # Load the specified model
        if algorithm == 'svm_rbf':
            model_path = os.path.join(current_dir, 'svm_rbf_model.joblib')
        elif algorithm == 'svm_linear':
            model_path = os.path.join(current_dir, 'svm_linear_model.joblib')
        elif algorithm == 'knn':
            model_path = os.path.join(current_dir, 'knn_model.joblib')
        elif algorithm == 'decision_tree':
            model_path = os.path.join(current_dir, 'decision_tree_model.joblib')
        elif algorithm == 'random_forest':
            model_path = os.path.join(current_dir, 'random_forest_model.joblib')
        elif algorithm == 'logistic':
            model_path = os.path.join(current_dir, 'logistic_model.joblib')
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        model = joblib.load(model_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def predict(features, algorithm='svm_rbf'):
    """
    Make a breast cancer prediction using the specified algorithm.
    Returns a dictionary containing the prediction and probability.
    """
    try:
        # Load the model and scaler
        model, scaler = load_model(algorithm)
        if model is None or scaler is None:
            return {
                'error': 'Model could not be loaded',
                'prediction': None,
                'probability': None
            }
        
        # Convert features to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get probability scores
        try:
            probability = model.predict_proba(features_scaled)[0][1]
        except:
            probability = None
        
        return {
            'prediction': int(prediction[0]),
            'probability': probability
        }
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            'error': str(e),
            'prediction': None,
            'probability': None
        }

if __name__ == '__main__':
    # Train and save the models
    print("Training models...")
    train_models()
    print("\nModel training completed and saved successfully!") 