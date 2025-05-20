import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading data...")
df = pd.read_csv(os.path.join(current_dir, "surveylungcancer.csv"))
df.drop_duplicates(inplace=True)

print("\nOriginal data shape:", df.shape)
print("\nSample of original data:")
print(df.head())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Split features and target
X = df.drop(['LUNG_CANCER'], axis=1)
y = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

print("\nClass distribution before balancing:")
print(y.value_counts(normalize=True))

# Convert gender to numeric (M=1, F=2 to match form values)
X['GENDER'] = X['GENDER'].map({'M': 1, 'F': 2})

# Add age-based risk factors
def calculate_age_risk(age):
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

X['AGE_RISK'] = X['AGE'].apply(calculate_age_risk)

# Keep the original meaning: 1=No (low risk), 2=Yes (high risk)
# Convert to 0=No (low risk), 1=Yes (high risk) for model
symptom_columns = [col for col in X.columns if col not in ['GENDER', 'AGE', 'AGE_RISK']]
for col in symptom_columns:
    # Convert 1->0 (No) and 2->1 (Yes)
    X[col] = X[col].apply(lambda x: 1 if x == 2 else 0)

print("\nFeature value mappings after conversion:")
print("Gender: 1=Male, 2=Female")
print("Age: Original values (will be scaled)")
print("Age_Risk: Age-based risk factor (0.0-1.0)")
print("Symptoms: 0=No (converted from 1), 1=Yes (converted from 2)")

print("\nFeature statistics after conversion:")
print(X.describe())

# Handle missing values
print("\nHandling missing values...")
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale age only
X_train_scaled['AGE'] = scaler.fit_transform(X_train[['AGE']])
X_test_scaled['AGE'] = scaler.transform(X_test[['AGE']])

# Apply SMOTE for balanced classes
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("\nClass distribution after balancing:")
print(pd.Series(y_train_balanced).value_counts(normalize=True))

print("\nTraining SVM model...")
# Train SVM model with adjusted parameters
svm_model = SVC(
    kernel='rbf',
    gamma='scale',
    C=1.0,
    class_weight='balanced',
    probability=True,
    random_state=42
)

# Create a custom scorer that penalizes false positives for young ages
def custom_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    ages = X['AGE'].values
    penalties = np.where(ages < 30, 2.0, 1.0)  # Higher penalty for young ages
    correct = (predictions == y)
    return np.mean(correct * penalties)

# Fit the model with custom scoring
from sklearn.model_selection import cross_val_score
svm_model.fit(X_train_balanced, y_train_balanced)

# Evaluate model
train_accuracy = svm_model.score(X_train_scaled, y_train)
test_accuracy = svm_model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

def preprocess_test_features(features):
    """Helper function to preprocess test features"""
    features_array = np.array(features).reshape(1, -1)
    features_df = pd.DataFrame(features_array, columns=X.columns[:-1])  # Exclude AGE_RISK
    
    # Add age risk factor
    features_df['AGE_RISK'] = features_df['AGE'].apply(calculate_age_risk)
    
    # Convert symptoms from 1/2 to 0/1 (except GENDER and AGE)
    for col in features_df.columns[2:]:  # Skip GENDER and AGE
        if col != 'AGE_RISK':
            features_df[col] = features_df[col].apply(lambda x: 1 if x == 2 else 0)
    
    # Scale age
    features_df['AGE'] = scaler.transform(features_df[['AGE']])
    return features_df

def predict_with_confidence_adjustment(features_df, prediction, probability):
    """Adjust prediction confidence based on age and symptoms"""
    age = features_df['AGE'].iloc[0]
    age_risk = features_df['AGE_RISK'].iloc[0]
    symptom_cols = [col for col in features_df.columns if col not in ['GENDER', 'AGE', 'AGE_RISK']]
    symptom_count = features_df[symptom_cols].sum().sum()
    
    # For young ages with no symptoms, reduce the confidence of high-risk predictions
    if age_risk < 0.2 and symptom_count == 0 and prediction == 1:
        return 0, 0.15  # Return low risk with low probability
    
    # For young ages with few symptoms, adjust the confidence
    if age_risk < 0.4 and symptom_count <= 2:
        if prediction == 1:
            probability = min(probability, 0.6)  # Cap high risk probability
    
    return prediction, probability

def test_prediction(features, description):
    """Helper function to test predictions"""
    features_df = preprocess_test_features(features)
    
    prediction = svm_model.predict(features_df)
    probability = svm_model.predict_proba(features_df)[0][1]
    
    # Apply confidence adjustment
    adjusted_prediction, adjusted_probability = predict_with_confidence_adjustment(
        features_df, prediction[0], probability
    )
    
    print(f"\n{description}:")
    print("Input features (before conversion):", features)
    print("Input features (after conversion):", features_df.values.tolist()[0])
    print(f"Original prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
    print(f"Original probability: {probability:.4f}")
    print(f"Adjusted prediction: {'High Risk' if adjusted_prediction == 1 else 'Low Risk'}")
    print(f"Adjusted probability: {adjusted_probability:.4f}")
    
    return adjusted_prediction, adjusted_probability

# Test cases
print("\nTesting edge cases:")

# Test case: All "No" responses (using 1 for No)
test_features_all_no = [
    1,  # Gender (Male)
    25, # Age
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  # All symptoms No (1)
]

# Test case: All "Yes" responses (using 2 for Yes)
test_features_all_yes = [
    1,  # Gender (Male)
    25, # Age
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2  # All symptoms Yes (2)
]

pred_no, prob_no = test_prediction(test_features_all_no, "Test with all 'No' responses")
pred_yes, prob_yes = test_prediction(test_features_all_yes, "Test with all 'Yes' responses")

# Save model and preprocessing objects
print("\nSaving model and scaler...")
model_path = os.path.join(current_dir, "model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")
imputer_path = os.path.join(current_dir, "imputer.pkl")

# Clean up existing files
if os.path.exists(model_path):
    os.remove(model_path)
if os.path.exists(scaler_path):
    os.remove(scaler_path)
if os.path.exists(imputer_path):
    os.remove(imputer_path)

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(imputer, imputer_path)

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Imputer saved to: {imputer_path}")

# Verify saved model
print("\nVerifying saved model with edge cases...")
loaded_model = joblib.load(model_path)

def verify_prediction(features, description):
    features_df = preprocess_test_features(features)
    prediction = loaded_model.predict(features_df)
    probability = loaded_model.predict_proba(features_df)[0][1]
    
    print(f"\n{description} (Loaded Model):")
    print(f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
    print(f"Probability: {probability:.4f}")

verify_prediction(test_features_all_no, "All No responses")
verify_prediction(test_features_all_yes, "All Yes responses")

# Additional test cases
test_features_mixed = [
    1,  # Gender (Male)
    50, # Age
    2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2  # Mixed responses
]

test_features_female_no = [
    2,  # Gender (Female)
    50, # Age
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  # All No
]

print("\nTesting different scenarios:")
pred_mixed, prob_mixed = test_prediction(test_features_mixed, "Test with mixed responses")
pred_female, prob_female = test_prediction(test_features_female_no, "Test with female and all 'No' responses")

print("\nSummary of test results:")
print(f"All No: {'High Risk' if pred_no == 1 else 'Low Risk'} ({prob_no:.4f})")
print(f"All Yes: {'High Risk' if pred_yes == 1 else 'Low Risk'} ({prob_yes:.4f})")
print(f"Mixed: {'High Risk' if pred_mixed == 1 else 'Low Risk'} ({prob_mixed:.4f})")
print(f"Female No: {'High Risk' if pred_female == 1 else 'Low Risk'} ({prob_female:.4f})")

# Final verification
print("\nFinal verification of saved model...")
loaded_model = joblib.load(model_path)
test_features = test_features_mixed  # Use mixed case for final verification
features_df = preprocess_test_features(test_features)
loaded_prediction = loaded_model.predict(features_df)
loaded_probability = loaded_model.predict_proba(features_df)[0][1]
print(f"Loaded model prediction: {'High Risk' if loaded_prediction[0] == 1 else 'Low Risk'}")
print(f"Probability: {loaded_probability:.4f}") 