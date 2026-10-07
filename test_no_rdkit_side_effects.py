import os
import joblib
import requests
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "drugsafe_no_rdkit_side_effect_model.pkl"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
vectorizer = artifact["vectorizer"]
target_names = artifact["target_names"]


def get_smiles(drug_name):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        "compound/name/"
        + drug_name
        + "/property/CanonicalSMILES/JSON"
    )

    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return None

    try:
        data = response.json()
        return data["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]
    except:
        return None


drug_a = "Warfarin"
drug_b = "Ibuprofen"

smiles_a = get_smiles(drug_a)
smiles_b = get_smiles(drug_b)

print("Drug A:", drug_a)
print("Drug B:", drug_b)

if not smiles_a or not smiles_b:
    print("Could not retrieve SMILES.")
    exit()

combined_smiles = smiles_a + " " + smiles_b

X = vectorizer.transform([combined_smiles])

# Get probability for every adverse event
probabilities = []

for estimator in model.estimators_:
    probability = estimator.predict_proba(X)[0][1]
    probabilities.append(probability)

probabilities = np.array(probabilities)

# Select top 5 events
top_indices = np.argsort(probabilities)[::-1][:5]

print("\nTop 5 Predicted Adverse Events:")

for index in top_indices:

    event_name = target_names[index]
    event_name = event_name.replace(
        "Target_Binary_", ""
    ).replace("_", " ")

    probability = probabilities[index] * 100

    print(
        f"- {event_name}: {probability:.2f}%"
    )