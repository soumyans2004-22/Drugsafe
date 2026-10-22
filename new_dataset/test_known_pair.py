import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem


# Load training data
df = pd.read_csv("extracted/train.csv")

# Take the first drug pair
row = df.iloc[0]

smiles_a = row["SMILES_A"]
smiles_b = row["SMILES_B"]
actual = row["Severity"]


# Morgan fingerprint generator
generator = AllChem.GetMorganGenerator(
    radius=2,
    fpSize=128
)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(128, dtype=np.int8)

    return np.array(
        generator.GetFingerprint(mol),
        dtype=np.int8
    )


# Create 256 features
fp_a = smiles_to_fingerprint(smiles_a)
fp_b = smiles_to_fingerprint(smiles_b)

X = np.hstack([fp_a, fp_b]).reshape(1, -1)


# Load model
model = xgb.XGBClassifier()
model.load_model("drugsafe_molecular_xgboost.json")


# Load encoder
encoder = joblib.load("severity_label_encoder.pkl")


# Predict
prediction = model.predict(X)

predicted = encoder.inverse_transform(prediction)[0]


print("Actual Severity:", actual)
print("Predicted Severity:", predicted)

if actual == predicted:
    print("Result: Correct prediction")
else:
    print("Result: Different prediction")