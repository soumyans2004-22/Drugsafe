from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.conf import settings

from .models import PredictionHistory

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

SEVERITY_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "drugsafe_molecular_xgboost.json"
)

SEVERITY_ENCODER_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "severity_label_encoder.pkl"
)

SIDE_EFFECT_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "drugsafe_side_effect_model.pkl"
)

SIDE_EFFECT_NAMES_PATH = os.path.join(
    settings.BASE_DIR,
    "drug",
    "new_dataset",
    "side_effect_names.npy"
)


# =========================
# LOAD SEVERITY MODEL
# =========================

severity_model = xgb.XGBClassifier()

severity_model.load_model(
    SEVERITY_MODEL_PATH
)

severity_encoder = joblib.load(
    SEVERITY_ENCODER_PATH
)


# =========================
# LOAD SIDE EFFECT MODEL
# =========================

side_effect_model = joblib.load(
    SIDE_EFFECT_MODEL_PATH
)

side_effect_names = np.load(
    SIDE_EFFECT_NAMES_PATH,
    allow_pickle=True
)


# =========================
# MOLECULAR FINGERPRINT
# =========================

generator = AllChem.GetMorganGenerator(
    radius=2,
    fpSize=128
)


def smiles_to_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fingerprint = generator.GetFingerprint(
        mol
    )

    return np.array(
        fingerprint,
        dtype=np.int8
    )


# =========================
# PUBCHEM SMILES LOOKUP
# =========================

def get_smiles(drug_name):

    drug_name = drug_name.strip()

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/name/{quote(drug_name)}/property/"
        "ConnectivitySMILES/JSON"
    )

    headers = {
        "User-Agent": "DrugSafe/1.0"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            return None

        data = response.json()

        return data[
            "PropertyTable"
        ][
            "Properties"
        ][0][
            "ConnectivitySMILES"
        ]

    except Exception:

        return None


# =========================
# HOME PAGE
# =========================

@login_required
def home(request):

    return render(
        request,
        "predictor/home.html"
    )


# =========================
# PREDICTION
# =========================

@login_required
def predict(request):

    if request.method != "POST":

        return render(
            request,
            "predictor/home.html"
        )

    # =========================
    # GET DRUG NAMES
    # =========================

    drug_a = request.POST.get(
        "drug_a",
        ""
    ).strip()

    drug_b = request.POST.get(
        "drug_b",
        ""
    ).strip()


    # =========================
    # CHECK INPUT
    # =========================

    if not drug_a or not drug_b:

        return render(
            request,
            "predictor/home.html",
            {
                "error":
                "Please enter both Drug A and Drug B."
            }
        )


    # =========================
    # GET SMILES FROM PUBCHEM
    # =========================

    smiles_a = get_smiles(
        drug_a
    )

    smiles_b = get_smiles(
        drug_b
    )


    if not smiles_a:

        return render(
            request,
            "predictor/home.html",
            {
                "error":
                f"Could not find '{drug_a}' in PubChem."
            }
        )


    if not smiles_b:

        return render(
            request,
            "predictor/home.html",
            {
                "error":
                f"Could not find '{drug_b}' in PubChem."
            }
        )


    # =========================
    # CREATE FINGERPRINTS
    # =========================

    fingerprint_a = smiles_to_fingerprint(
        smiles_a
    )

    fingerprint_b = smiles_to_fingerprint(
        smiles_b
    )


    if fingerprint_a is None:

        return render(
            request,
            "predictor/home.html",
            {
                "error":
                f"Invalid molecular structure for {drug_a}."
            }
        )


    if fingerprint_b is None:

        return render(
            request,
            "predictor/home.html",
            {
                "error":
                f"Invalid molecular structure for {drug_b}."
            }
        )


    # =========================
    # COMBINE MOLECULAR FEATURES
    # =========================

    X_new = np.hstack(
        [
            fingerprint_a,
            fingerprint_b
        ]
    ).reshape(
        1,
        -1
    )


    # ==================================================
    # RISK / SEVERITY PREDICTION
    # ==================================================

    severity_prediction = severity_model.predict(
        X_new
    )

    risk_level = severity_encoder.inverse_transform(
        severity_prediction
    )[0]


    # ==================================================
    # RISK PROBABILITIES
    # ==================================================

    probabilities = severity_model.predict_proba(
        X_new
    )[0]


    # =========================
    # CONFIDENCE
    # =========================

    confidence = float(
        np.max(probabilities) * 100
    )


    # =========================
    # INDIVIDUAL PROBABILITIES
    # =========================

    major_probability = float(
        probabilities[0] * 100
    )

    minor_probability = float(
        probabilities[1] * 100
    )

    moderate_probability = float(
        probabilities[2] * 100
    )


    # ==================================================
    # DEBUG INFORMATION
    # ==================================================

    print("\n==============================")
    print("DRUGSAFE PREDICTION")
    print("==============================")

    print(
        "Drug A:",
        drug_a
    )

    print(
        "Drug B:",
        drug_b
    )

    print(
        "SMILES A:",
        smiles_a
    )

    print(
        "SMILES B:",
        smiles_b
    )

    print(
        "Major:",
        round(
            major_probability,
            2
        ),
        "%"
    )

    print(
        "Minor:",
        round(
            minor_probability,
            2
        ),
        "%"
    )

    print(
        "Moderate:",
        round(
            moderate_probability,
            2
        ),
        "%"
    )

    print(
        "Predicted:",
        risk_level
    )

    print(
        "Confidence:",
        round(
            confidence,
            2
        ),
        "%"
    )

    print("==============================\n")


    # ==================================================
    # SIDE EFFECT PREDICTION
    # ==================================================

    side_effect_probabilities = (
        side_effect_model.predict_proba(
            X_new
        )
    )


    effect_scores = []


    for i, probabilities in enumerate(
        side_effect_probabilities
    ):

        probability = float(
            probabilities[0][1]
        )

        effect_scores.append(
            (
                str(
                    side_effect_names[i]
                ),
                probability
            )
        )


    # =========================
    # SORT SIDE EFFECTS
    # =========================

    effect_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # =========================
    # SELECT TOP 5
    # =========================

    top_side_effects = [
        effect[0]
        for effect in effect_scores[:5]
    ]


    # ==================================================
    # SAVE PREDICTION HISTORY
    # ==================================================

    PredictionHistory.objects.create(
        user=request.user,
        drug_a=drug_a,
        drug_b=drug_b,
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        risk_level=risk_level,
        confidence=confidence,
        major_probability=major_probability,
        minor_probability=minor_probability,
        moderate_probability=moderate_probability,
        side_effects=", ".join(
            top_side_effects
        )
    )


    # ==================================================
    # RESULT CONTEXT
    # ==================================================

    context = {

        "drug_a":
        drug_a,

        "drug_b":
        drug_b,

        "smiles_a":
        smiles_a,

        "smiles_b":
        smiles_b,

        "risk_level":
        risk_level,

        "confidence":
        round(
            confidence,
            2
        ),

        # Three severity probabilities
        "major_probability":
        round(
            major_probability,
            2
        ),

        "minor_probability":
        round(
            minor_probability,
            2
        ),

        "moderate_probability":
        round(
            moderate_probability,
            2
        ),

        # Top 5 associated adverse events
        "side_effects":
        top_side_effects
    }


    # =========================
    # SHOW RESULT
    # =========================

    return render(
        request,
        "predictor/home.html",
        context
    )