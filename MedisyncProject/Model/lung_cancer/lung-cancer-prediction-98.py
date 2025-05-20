import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load and preprocess data
df = pd.read_csv("Model/surveylungcancer.csv")
df.drop_duplicates(inplace=True)

# Encode categorical columns
encoder = LabelEncoder()
df['LUNG_CANCER'] = encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = encoder.fit_transform(df['GENDER'])

# Split features and target
X = df.drop(['LUNG_CANCER'], axis=1)
y = df['LUNG_CANCER']

# Convert 2,1 to 1,0 in feature columns
for i in X.columns[2:]:
    X[i] = X[i].apply(lambda x: x-1)

# Oversample minority class
X_over, y_over = RandomOverSampler().fit_resample(X, y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, random_state=42, stratify=y_over)

# Scale AGE column
scaler = StandardScaler()
X_train['AGE'] = scaler.fit_transform(X_train[['AGE']])
X_test['AGE'] = scaler.transform(X_test[['AGE']])

print("Data preprocessing completed.")

# Train and save models
print("\nTraining and saving models...")

# Save SVM model (98% accuracy)
svm_model = SVC(gamma=10, C=100, probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'Model/svm_model.pkl')
print("SVM model saved successfully")

# Save KNN model (78% accuracy)
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, 'Model/knn_model.joblib')
print("KNN model saved successfully")
<<<<<<< SEARCH
joblib.dump(gb_model, 'Model/gradient_boosting_model.pkl')
print("Gradient Boosting model saved successfully")
=======
joblib.dump(gb_model, 'Model/gradient_boosting_model.joblib')
print("Gradient Boosting model saved successfully")

# Save Logistic Regression model (85% accuracy)
log_model = LogisticRegression(solver='lbfgs', C=1, max_iter=200)
log_model.fit(X_train, y_train)
joblib.dump(log_model, 'Model/logistic_model.pkl')
print("Logistic Regression model saved successfully")

# Save Random Forest model (88% accuracy)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'Model/random_forest_model.pkl')
print("Random Forest model saved successfully")

# Save Gradient Boosting model (88% accuracy)
gb_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, 'Model/gradient_boosting_model.pkl')
print("Gradient Boosting model saved successfully")

# Save the scaler for preprocessing
joblib.dump(scaler, 'Model/scaler.pkl')
print("Scaler saved successfully")

print("\nAll models have been saved successfully!") 