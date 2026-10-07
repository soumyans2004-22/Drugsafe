import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("new_dataset/extracted/train.csv")

print("Dataset shape:", df.shape)


# 2. Combine both drug SMILES
df["combined_smiles"] = (
    df["SMILES_A"].fillna("") + " " +
    df["SMILES_B"].fillna("")
)


# 3. Convert SMILES into TF-IDF features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=2000
)

X = vectorizer.fit_transform(df["combined_smiles"])


# 4. Encode severity
encoder = LabelEncoder()
y = encoder.fit_transform(df["Severity"])

print("Feature shape:", X.shape)
print("Classes:", encoder.classes_)


# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)


# 6. Custom class weights
class_weights = {
    0: 1.5,   # Major
    1: 2.0,   # Minor
    2: 1.0    # Moderate
}

sample_weights = [
    class_weights[class_id]
    for class_id in y_train
]


# 7. Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=2
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


# 8. Prediction
y_pred = model.predict(X_test)


# 9. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# 10. Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)


# 11. Confusion matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# 12. Save model
model.save_model(
    "drugsafe_no_rdkit_xgboost.json"
)

joblib.dump(
    vectorizer,
    "smiles_tfidf_vectorizer.pkl"
)

joblib.dump(
    encoder,
    "severity_no_rdkit_encoder.pkl"
)


print("\nModel saved successfully!")