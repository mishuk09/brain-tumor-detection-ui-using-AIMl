import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# 📌 Step 1: Load the dataset
df = pd.read_csv("brain_tumor_features.csv")

# 📌 Step 2: Feature Engineering
# Adding new features (Example: Squaring Mean_Intensity)
df["Intensity_Square"] = df["Mean_Intensity"] ** 2  # Feature transformation
df["Texture_Log"] = np.log1p(df["Texture"])  # Log transformation

# Define features and labels
X = df[['Mean_Intensity', 'Texture', 'Edge_Count', 'Intensity_Square', 'Texture_Log']]
y = df['Label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 📌 Step 3: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Step 4: Handle Class Imbalance using SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 📌 Step 5: Hyperparameter Tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_rf = grid_search.best_estimator_

# 📌 Step 6: Train an XGBoost Model
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# 📌 Step 7: Train an SVM Model
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# 📌 Step 8: Ensemble Learning (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_rf),
    ('svm', svm_model),
    ('xgb', xgb_model)
], voting='soft')  # 'soft' gives probability-based voting

ensemble_model.fit(X_train_resampled, y_train_resampled)

# 📌 Step 9: Model Evaluation
y_pred = ensemble_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 Final Model Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 📌 Step 10: Cross-Validation Score (for more stability)
cv_scores = cross_val_score(ensemble_model, X_scaled, y_encoded, cv=5)
print(f"\n📌 Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%\n")

# 📌 Step 11: Save the Model, Scaler, and Label Encoder
joblib.dump({"model": ensemble_model, "scaler": scaler, "label_encoder": label_encoder}, "model.pkl")
print("✅ Model saved as 'model.pkl'")
