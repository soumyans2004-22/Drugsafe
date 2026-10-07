import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "new_dataset",
    "extracted",
    "train.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "drugsafe_no_rdkit_side_effect_model.pkl"
)


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(TRAIN_PATH)

print("Dataset shape:", df.shape)


# =========================
# COMBINE SMILES
# =========================

df["combined_smiles"] = (
    df["SMILES_A"].fillna("")
    + " "
    + df["SMILES_B"].fillna("")
)


# =========================
# GET 50 BINARY EVENTS
# =========================

target_columns = [
    col for col in df.columns
    if col.startswith("Target_Binary_")
]

print(
    "Adverse-event targets:",
    len(target_columns)
)


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=2000
)

X = vectorizer.fit_transform(
    df["combined_smiles"]
)

y = df[target_columns].astype(int)


print(
    "Feature shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)


# =========================
# XGBOOST
# =========================

base_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2
)


# =========================
# MULTI-OUTPUT MODEL
# =========================

model = MultiOutputClassifier(
    base_model,
    n_jobs=1
)


print("\nTraining adverse-event model...")


model.fit(
    X,
    y
)


# =========================
# SAVE EVERYTHING
# =========================

artifact = {
    "model": model,
    "vectorizer": vectorizer,
    "target_names": np.array(
        target_columns
    )
}

joblib.dump(
    artifact,
    MODEL_PATH
)


print("\nTraining completed.")

print(
    "Model saved to:",
    MODEL_PATH
)

print(
    "Number of adverse events:",
    len(target_columns)
)