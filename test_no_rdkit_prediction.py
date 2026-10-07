import os
import requests
import joblib
import xgboost as xgb
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "drugsafe_no_rdkit_xgboost.json")
VECTORIZER_PATH = os.path.join(BASE_DIR, "smiles_tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "severity_no_rdkit_encoder.pkl")


def get_smiles(drug_name):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        "compound/name/"
        + quote(drug_name)
        + "/property/ConnectivitySMILES/JSON"
    )

    response = requests.get(url, timeout=15)

    if response.status_code != 200:
        return None

    return response.json()["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]


# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

vectorizer = joblib.load(VECTORIZER_PATH)
encoder = joblib.load(ENCODER_PATH)

# Test drugs
drug_a = "Warfarin"
drug_b = "Ibuprofen"

smiles_a = get_smiles(drug_a)
smiles_b = get_smiles(drug_b)

print("\nDrug A:", drug_a)
print("SMILES A:", smiles_a)

print("\nDrug B:", drug_b)
print("SMILES B:", smiles_b)

if not smiles_a or not smiles_b:
    print("\nCould not get SMILES from PubChem.")
    exit()

# Convert SMILES to TF-IDF
combined = smiles_a + " " + smiles_b
X = vectorizer.transform([combined])

# Prediction
probabilities = model.predict_proba(X)[0]
prediction = model.predict(X)[0]

risk = encoder.inverse_transform([prediction])[0]
confidence = max(probabilities) * 100

print("\n-----------------------------")
print("DrugSafe Prediction")
print("-----------------------------")
print("Risk Level :", risk)
print("Confidence :", round(confidence, 2), "%")

print("\nProbabilities:")

for label, probability in zip(encoder.classes_, probabilities):
    print(label, ":", round(probability * 100, 2), "%")