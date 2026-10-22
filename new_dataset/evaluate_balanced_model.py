import numpy as np
import xgboost as xgb
import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load test data
X_test = np.load("X_test.npy")

y_test = np.load(
    "y_test.npy",
    allow_pickle=True
)

# Load model
model = xgb.XGBClassifier()

model.load_model(
    "drugsafe_molecular_xgboost.json"
)

# Load encoder
label_encoder = joblib.load(
    "severity_label_encoder.pkl"
)

# Encode test labels
y_test_encoded = label_encoder.transform(
    y_test
)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test_encoded,
    y_pred
)

print("\n==============================")
print("BALANCED XGBOOST EVALUATION")
print("==============================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test_encoded,
        y_pred
    )
)