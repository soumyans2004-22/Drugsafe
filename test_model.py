import joblib
import xgboost as xgb
import pandas as pd

print("Loading model files...")

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model("drugsafe_xgboost.json")

# Load encoders
le_drug_a = joblib.load("drug_a_encoder.pkl")
le_drug_b = joblib.load("drug_b_encoder.pkl")
le_level = joblib.load("level_encoder.pkl")

print("XGBoost model loaded successfully!")
print("Drug A encoder loaded!")
print("Drug B encoder loaded!")
print("Level encoder loaded!")

# Get drug names
drug_a = input("Enter Drug A: ")
drug_b = input("Enter Drug B: ")

drug_a = drug_a.strip()
drug_b = drug_b.strip()

# Find drug names
match_a = [x for x in le_drug_a.classes_ if drug_a.lower() == x.lower()]
match_b = [x for x in le_drug_b.classes_ if drug_b.lower() == x.lower()]

if not match_a:
    print("Drug A not found in dataset.")
    exit()

if not match_b:
    print("Drug B not found in dataset.")
    exit()

drug_a = match_a[0]
drug_b = match_b[0]

# Encode drugs
drug_a_encoded = le_drug_a.transform([drug_a])[0]
drug_b_encoded = le_drug_b.transform([drug_b])[0]

# Create input data
X_new = pd.DataFrame({
    "Drug_A_encoded": [drug_a_encoded],
    "Drug_B_encoded": [drug_b_encoded]
})

# Make prediction
prediction = model.predict(X_new)

# Convert prediction to risk level
risk_level = le_level.inverse_transform(prediction)[0]

print()
print("================================")
print("         DRUGSAFE RESULT")
print("================================")
print("Drug A     :", drug_a)
print("Drug B     :", drug_b)
print("Risk Level :", risk_level)
print("================================")