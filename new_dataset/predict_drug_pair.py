import numpy as np
import joblib
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
from urllib.parse import quote


# Load model
model = xgb.XGBClassifier()
model.load_model("drugsafe_molecular_xgboost.json")

# Load severity encoder
label_encoder = joblib.load("severity_label_encoder.pkl")

# Morgan fingerprint
generator = AllChem.GetMorganGenerator(
    radius=2,
    fpSize=128
)


def get_smiles(drug_name):

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/name/{quote(drug_name)}/property/"
        "ConnectivitySMILES/JSON"
    )

    headers = {
        "User-Agent": "DrugSafe/1.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    if response.status_code != 200:
        return None

    data = response.json()

    try:
        return data["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]
    except (KeyError, IndexError):
        return None


def smiles_to_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(128, dtype=np.int8)

    fp = generator.GetFingerprint(mol)

    return np.array(fp, dtype=np.int8)


drug_a = input("Enter Drug A: ")
drug_b = input("Enter Drug B: ")

smiles_a = get_smiles(drug_a)
smiles_b = get_smiles(drug_b)

if not smiles_a or not smiles_b:
    print("One or both drugs could not be found.")
    exit()

print("\nDrug A:", drug_a)
print("SMILES A:", smiles_a)

print("\nDrug B:", drug_b)
print("SMILES B:", smiles_b)


fp_a = smiles_to_fingerprint(smiles_a)
fp_b = smiles_to_fingerprint(smiles_b)

X = np.hstack([fp_a, fp_b]).reshape(1, -1)

prediction = model.predict(X)

risk_level = label_encoder.inverse_transform(prediction)[0]

print("\n==============================")
print("DrugSafe Prediction")
print("==============================")
print("Drug A:", drug_a)
print("Drug B:", drug_b)
print("Predicted Severity:", risk_level)
print("==============================")