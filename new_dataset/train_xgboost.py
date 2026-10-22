import numpy as np
import xgboost as xgb
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


# =========================
# LOAD DATA
# =========================

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load(
    "y_train.npy",
    allow_pickle=True
)

y_test = np.load(
    "y_test.npy",
    allow_pickle=True
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================
# ENCODE SEVERITY LABELS
# =========================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)

y_test_encoded = label_encoder.transform(
    y_test
)


print("\nClasses:")
print(label_encoder.classes_)


# =========================
# BALANCE THE CLASSES
# =========================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded
)

print("\nClass balancing enabled.")


# =========================
# CREATE XGBOOST MODEL
# =========================

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# =========================
# TRAIN MODEL
# =========================

print("\nTraining balanced XGBoost...")

model.fit(
    X_train,
    y_train_encoded,
    sample_weight=sample_weights
)

print("Training completed!")


# =========================
# SAVE MODEL
# =========================

model.save_model(
    "drugsafe_molecular_xgboost.json"
)

joblib.dump(
    label_encoder,
    "severity_label_encoder.pkl"
)


print("\nModel saved as:")
print("drugsafe_molecular_xgboost.json")

print("\nEncoder saved as:")
print("severity_label_encoder.pkl")

print("\nClasses:")
print(label_encoder.classes_)