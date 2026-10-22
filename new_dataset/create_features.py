import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


# Load dataset
df = pd.read_csv("extracted/train.csv")

# Morgan fingerprint generator
generator = AllChem.GetMorganGenerator(
    radius=2,
    fpSize=128
)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(128, dtype=np.int8)

    fp = generator.GetFingerprint(mol)

    return np.array(fp, dtype=np.int8)


# Create fingerprints for Drug A and Drug B
drug_a_features = np.array(
    [smiles_to_fingerprint(x) for x in df["SMILES_A"]]
)

drug_b_features = np.array(
    [smiles_to_fingerprint(x) for x in df["SMILES_B"]]
)


# Combine Drug A + Drug B
X = np.hstack([
    drug_a_features,
    drug_b_features
])


# Target
y = df["Severity"]


# Save features
np.save("X_molecular_features.npy", X)
np.save("y_severity.npy", y.to_numpy())


print("Feature creation completed!")
print("Number of rows:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target values:", y.unique())