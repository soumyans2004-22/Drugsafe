import pandas as pd
import numpy as np

# Load training dataset
df = pd.read_csv("extracted/train.csv")

# Load molecular features
X = np.load("X_molecular_features.npy")

# Get all side-effect columns
side_effect_columns = [
    col for col in df.columns
    if col.startswith("Target_Binary_")
]

print("Number of side effects:", len(side_effect_columns))

print("\nSide effects:")
for col in side_effect_columns:
    print(col.replace("Target_Binary_", ""))

# Extract side-effect targets
Y = df[side_effect_columns].values

# Save data
np.save("Y_side_effects.npy", Y)

# Save side-effect names
np.save(
    "side_effect_names.npy",
    np.array(
        [col.replace("Target_Binary_", "") for col in side_effect_columns],
        dtype=object
    )
)

print("\nX shape:", X.shape)
print("Y shape:", Y.shape)

print("\nSide-effect data created successfully.")python create_side_effect_data.py