import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem


# Load datasets
train = pd.read_csv("extracted/train.csv")
test = pd.read_csv("extracted/test.csv")


# Find completely unseen structures
train_structures = set(train["SMILES_A"]) | set(train["SMILES_B"])
test_structures = set(test["SMILES_A"]) | set(test["SMILES_B"])

unseen_structures = test_structures - train_structures


# Select test pairs containing an unseen structure
unseen_test = test[
    test["SMILES_A"].isin(unseen_structures) |
    test["SMILES_B"].isin(unseen_structures)
].copy()


print("Completely unseen structures:", len(unseen_structures))
print("Test pairs containing unseen structures:", len(unseen_test))


# Morgan fingerprints
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


# Create features
features = []

for _, row in unseen_test.iterrows():

    fp_a = smiles_to_fingerprint(row["SMILES_A"])
    fp_b = smiles_to_fingerprint(row["SMILES_B"])

    features.append(
        np.hstack([fp_a, fp_b])
    )


X_unseen = np.array(features)


# Load model
model = xgb.XGBClassifier()
model.load_model("drugsafe_molecular_xgboost.json")


# Load label encoder
label_encoder = joblib.load(
    "severity_label_encoder.pkl"
)


# Predict
predictions = model.predict(X_unseen)

predicted_severity = label_encoder.inverse_transform(
    predictions
)


# Add predictions
unseen_test["Predicted_Severity"] = predicted_severity


# Mark which side is unseen
unseen_test["Unseen_A"] = unseen_test["SMILES_A"].isin(
    unseen_structures
)

unseen_test["Unseen_B"] = unseen_test["SMILES_B"].isin(
    unseen_structures
)


# Save results
unseen_test.to_csv(
    "unseen_drug_predictions.csv",
    index=False
)


print("\nPrediction completed!")

print(
    "Predictions saved to: unseen_drug_predictions.csv"
)

print("\nPredicted severity counts:")

print(
    unseen_test["Predicted_Severity"].value_counts()
)