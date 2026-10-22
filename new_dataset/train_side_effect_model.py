
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load data
X = np.load("X_molecular_features.npy")
Y = np.load("Y_side_effects.npy")

print("Features shape:", X.shape)
print("Labels shape:", Y.shape)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# XGBoost model
base_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(base_model, n_jobs=-1)

# Train model
print("\nTraining model...")
model.fit(X_train, Y_train)

# Test model
print("Testing model...")
Y_pred = model.predict(X_test)

# Evaluation results
print("\n===== TEST RESULTS =====")

print("Exact-match accuracy:",
      round(accuracy_score(Y_test, Y_pred), 4))

print("Micro precision:",
      round(precision_score(
          Y_test, Y_pred, average="micro", zero_division=0
      ), 4))

print("Micro recall:",
      round(recall_score(
          Y_test, Y_pred, average="micro", zero_division=0
      ), 4))

print("Micro F1-score:",
      round(f1_score(
          Y_test, Y_pred, average="micro", zero_division=0
      ), 4))

print("Macro F1-score:",
      round(f1_score(
          Y_test, Y_pred, average="macro", zero_division=0
      ), 4))

# Save model
joblib.dump(model, "drugsafe_side_effect_model.pkl")

print("\nModel saved successfully.")
