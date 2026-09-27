from django.shortcuts import render
from django.conf import settings

import pandas as pd
import joblib
import xgboost as xgb
import os
import re


# ==============================
# FILE PATHS
# ==============================
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "drugsafe_xgboost.json"
)

DRUG_A_ENCODER_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "drug_a_encoder.pkl"
)

DRUG_B_ENCODER_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "drug_b_encoder.pkl"
)

LEVEL_ENCODER_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "level_encoder.pkl"
)

CSV_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "DDInter_SIDER_30000.csv"
)


# ==============================
# LOAD MODEL
# ==============================

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

le_drug_a = joblib.load(DRUG_A_ENCODER_PATH)
le_drug_b = joblib.load(DRUG_B_ENCODER_PATH)
le_level = joblib.load(LEVEL_ENCODER_PATH)


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(CSV_PATH)


# ==============================
# SIDE EFFECT FUNCTION
# ==============================

def get_side_effects(text):

    if pd.isna(text):
        return []

    effects = re.split(r"[;,|]", str(text))

    effects = [
        effect.strip()
        for effect in effects
        if effect.strip()
    ]

    # Remove duplicates
    unique_effects = []

    for effect in effects:

        if effect.lower() not in [
            x.lower() for x in unique_effects
        ]:
            unique_effects.append(effect)

    return unique_effects[:5]


# ==============================
# HOME PAGE
# ==============================

def home(request):

    return render(
        request,
        "predictor/home.html"
    )


# ==============================
# PREDICTION
# ==============================

def predict(request):

    if request.method == "POST":

        drug_a_input = request.POST.get(
            "drug_a", ""
        ).strip()

        drug_b_input = request.POST.get(
            "drug_b", ""
        ).strip()

        # Check empty input
        if not drug_a_input or not drug_b_input:

            return render(
                request,
                "predictor/home.html",
                {
                    "error": "Please enter both Drug A and Drug B."
                }
            )

        # ==============================
        # FIND DRUG A
        # ==============================

        match_a = [
            x for x in le_drug_a.classes_
            if drug_a_input.lower() == x.lower()
        ]

        # ==============================
        # FIND DRUG B
        # ==============================

        match_b = [
            x for x in le_drug_b.classes_
            if drug_b_input.lower() == x.lower()
        ]

        if not match_a or not match_b:

            return render(
                request,
                "predictor/home.html",
                {
                    "error": "One or both drugs were not found in the dataset."
                }
            )

        drug_a = match_a[0]
        drug_b = match_b[0]

        # ==============================
        # ENCODE DRUGS
        # ==============================

        drug_a_encoded = le_drug_a.transform(
            [drug_a]
        )[0]

        drug_b_encoded = le_drug_b.transform(
            [drug_b]
        )[0]

        # ==============================
        # MODEL INPUT
        # ==============================

        X_new = pd.DataFrame({
            "Drug_A_encoded": [drug_a_encoded],
            "Drug_B_encoded": [drug_b_encoded]
        })

        # ==============================
        # PREDICT RISK LEVEL
        # ==============================

        prediction = model.predict(X_new)

        risk_level = le_level.inverse_transform(
            prediction
        )[0]

        # ==============================
        # FIND SIDE EFFECTS
        # ==============================

        drug_a_effects = []
        drug_b_effects = []

        pair_rows = df[
            (
                (df["Drug_A"].astype(str).str.lower() == drug_a.lower())
                &
                (df["Drug_B"].astype(str).str.lower() == drug_b.lower())
            )
            |
            (
                (df["Drug_A"].astype(str).str.lower() == drug_b.lower())
                &
                (df["Drug_B"].astype(str).str.lower() == drug_a.lower())
            )
        ]

        for _, row in pair_rows.iterrows():

            csv_drug_a = str(
                row["Drug_A"]
            ).lower()

            if csv_drug_a == drug_a.lower():

                drug_a_effects.extend(
                    get_side_effects(
                        row["Side_Effect_A"]
                    )
                )

                drug_b_effects.extend(
                    get_side_effects(
                        row["Side_Effect_B"]
                    )
                )

            else:

                drug_a_effects.extend(
                    get_side_effects(
                        row["Side_Effect_B"]
                    )
                )

                drug_b_effects.extend(
                    get_side_effects(
                        row["Side_Effect_A"]
                    )
                )

        # ==============================
        # REMOVE DUPLICATES
        # ==============================

        drug_a_effects = list(
            dict.fromkeys(drug_a_effects)
        )[:5]

        drug_b_effects = list(
            dict.fromkeys(drug_b_effects)
        )[:5]

        # ==============================
        # NO SIDE EFFECT DATA
        # ==============================

        if not drug_a_effects:

            drug_a_effects = [
                "No side effect information available"
            ]

        if not drug_b_effects:

            drug_b_effects = [
                "No side effect information available"
            ]

        # ==============================
        # SEND RESULT TO HTML
        # ==============================

        context = {

            "drug_a": drug_a,

            "drug_b": drug_b,

            "risk_level": risk_level,

            "drug_a_effects": drug_a_effects,

            "drug_b_effects": drug_b_effects,
        }

        return render(
            request,
            "predictor/home.html",
            context
        )

    return render(
        request,
        "predictor/home.html"
    )