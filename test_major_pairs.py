import os
import requests
import numpy as np
import joblib
import xgboost as xgb

from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# MODEL PATHS
# =========================

MODEL_PATH = "drugsafe_molecular_xgboost.json"
ENCODER_PATH = "severity_label_encoder.pkl"


# =========================
# LOAD MODEL
# =========================

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)


# =========================
# MORGAN FINGERPRINT
# =========================

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

    try:

        response = requests.get(
            url,
            headers={"User-Agent": "DrugSafe/1.0"},
            timeout=15
        )

        if response.status_code != 200:
            return None

        data = response.json()

        return data["PropertyTable"]["Properties"][0][
            "ConnectivitySMILES"
        ]

    except Exception:
        return None


def fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = generator.GetFingerprint(mol)

    return np.array(fp, dtype=np.int8)


# =========================
# CANDIDATE DRUG PAIRS
# =========================

pairs = [

    ("Warfarin", "Ibuprofen"),
    ("Warfarin", "Aspirin"),
    ("Warfarin", "Naproxen"),
    ("Warfarin", "Diclofenac"),
    ("Warfarin", "Ketorolac"),

    ("Methotrexate", "Ibuprofen"),
    ("Methotrexate", "Naproxen"),

    ("Digoxin", "Amiodarone"),
    ("Digoxin", "Verapamil"),

    ("Theophylline", "Ciprofloxacin"),
    ("Theophylline", "Clarithromycin"),

    ("Simvastatin", "Clarithromycin"),
    ("Atorvastatin", "Clarithromycin"),

    ("Tramadol", "Sertraline"),
    ("Tramadol", "Fluoxetine"),

    ("Lithium", "Ibuprofen"),
    ("Lithium", "Naproxen"),

    ("Clopidogrel", "Omeprazole"),
    ("Phenytoin", "Fluconazole"),
]


# =========================
# TEST PAIRS
# =========================

major_pairs = []

print("\n==============================")
print("DRUGSAFE MAJOR PAIR TEST")
print("==============================\n")

for drug_a, drug_b in pairs:

    smiles_a = get_smiles(drug_a)
    smiles_b = get_smiles(drug_b)

    if not smiles_a or not smiles_b:

        print(
            f"{drug_a} + {drug_b} -> "
            "SMILES not found"
        )

        continue

    fp_a = fingerprint(smiles_a)
    fp_b = fingerprint(smiles_b)

    if fp_a is None or fp_b is None:

        print(
            f"{drug_a} + {drug_b} -> "
            "Invalid structure"
        )

        continue

    X = np.hstack(
        [fp_a, fp_b]
    ).reshape(1, -1)

    prediction = model.predict(X)

    risk = encoder.inverse_transform(
        prediction
    )[0]

    probabilities = model.predict_proba(X)[0]

    confidence = np.max(probabilities) * 100

    print(
        f"{drug_a} + {drug_b} "
        f"-> {risk} "
        f"({confidence:.2f}%)"
    )

    if risk == "Major":

        major_pairs.append(
            (
                drug_a,
                drug_b,
                confidence
            )
        )


# =========================
# SHOW MAJOR PAIRS
# =========================

print("\n==============================")
print("PAIRS PREDICTED AS MAJOR")
print("==============================")

if major_pairs:

    for drug_a, drug_b, confidence in major_pairs:

        print(
            f"{drug_a} + {drug_b} "
            f"-> Major "
            f"({confidence:.2f}%)"
        )

else:

    print("No Major pairs found.")


print("\nTesting completed.")