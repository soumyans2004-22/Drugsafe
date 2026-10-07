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


# =========================
# MODEL PATHS
# =========================

BASE_DIR = os.path.join(
    settings.BASE_DIR,
    "drug"
)

SEVERITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "drugsafe_no_rdkit_xgboost.json"
)

SEVERITY_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "smiles_tfidf_vectorizer.pkl"
)

SEVERITY_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "severity_no_rdkit_encoder.pkl"
)

SIDE_EFFECT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "drugsafe_no_rdkit_side_effect_model.pkl"
)


# =========================
# LOAD SEVERITY MODEL
# =========================

severity_model = xgb.XGBClassifier()

severity_model.load_model(
    SEVERITY_MODEL_PATH
)

severity_vectorizer = joblib.load(
    SEVERITY_VECTORIZER_PATH
)

severity_encoder = joblib.load(
    SEVERITY_ENCODER_PATH
)


# =========================
# LOAD ADVERSE-EVENT MODEL
# =========================

side_effect_artifact = joblib.load(
    SIDE_EFFECT_MODEL_PATH
)

side_effect_model = side_effect_artifact["model"]

side_effect_vectorizer = side_effect_artifact["vectorizer"]

side_effect_names = side_effect_artifact["target_names"]


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
    # GET SMILES
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


    # ==================================================
    # COMBINED SMILES
    # ==================================================

    combined_smiles = (
        smiles_a
        + " "
        + smiles_b
    )


    # ==================================================
    # RISK / SEVERITY PREDICTION
    # ==================================================

    X_severity = severity_vectorizer.transform(
        [combined_smiles]
    )

    severity_prediction = severity_model.predict(
        X_severity
    )

    risk_level = severity_encoder.inverse_transform(
        severity_prediction
    )[0]


    # ==================================================
    # RISK PROBABILITIES
    # ==================================================

    probabilities = severity_model.predict_proba(
        X_severity
    )[0]


    confidence = float(
        np.max(probabilities) * 100
    )


    probability_map = dict(
        zip(
            severity_encoder.classes_,
            probabilities
        )
    )


    major_probability = float(
        probability_map.get(
            "Major",
            0
        ) * 100
    )

    minor_probability = float(
        probability_map.get(
            "Minor",
            0
        ) * 100
    )

    moderate_probability = float(
        probability_map.get(
            "Moderate",
            0
        ) * 100
    )


    # ==================================================
    # ADVERSE-EVENT PREDICTION
    # ==================================================

    X_side_effect = side_effect_vectorizer.transform(
        [combined_smiles]
    )

    side_effect_probabilities = []


    for estimator in side_effect_model.estimators_:

        probability = estimator.predict_proba(
            X_side_effect
        )[0][1]

        side_effect_probabilities.append(
            probability
        )


    side_effect_probabilities = np.array(
        side_effect_probabilities
    )


    # ==================================================
    # TOP 5 ADVERSE EVENTS
    # ==================================================

    top_indices = np.argsort(
        side_effect_probabilities
    )[::-1][:5]


    top_side_effects = []


    for index in top_indices:

        event_name = side_effect_names[index]

        event_name = event_name.replace(
            "Target_Binary_",
            ""
        ).replace(
            "_",
            " "
        )

        probability = (
            side_effect_probabilities[index]
            * 100
        )

        top_side_effects.append({
            "name": event_name,
            "probability": round(
                float(probability),
                2
            )
        })


    # ==================================================
    # DEBUG
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
        "Risk Level:",
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

    print(
        "\nTop Adverse Events:"
    )

    for event in top_side_effects:

        print(
            "-",
            event["name"],
            ":",
            event["probability"],
            "%"
        )

    print("==============================\n")


    # ==================================================
    # SAVE ADVERSE EVENTS
    # ==================================================

    side_effect_text = ", ".join(
        event["name"]
        for event in top_side_effects
    )


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
        side_effects=side_effect_text
    )


    # ==================================================
    # MOLECULAR IMAGE URLS
    # ==================================================

    molecule_image_a = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        "compound/name/"
        + quote(drug_a)
        + "/PNG?image_size=300x300"
    )

    molecule_image_b = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        "compound/name/"
        + quote(drug_b)
        + "/PNG?image_size=300x300"
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

        "side_effects":
        top_side_effects,

        "molecule_image_a":
        molecule_image_a,

        "molecule_image_b":
        molecule_image_b
    }


    return render(
        request,
        "predictor/home.html",
        context
    )


# =========================
# PREDICTION HISTORY
# =========================

@login_required
def history(request):

    predictions = PredictionHistory.objects.filter(
        user=request.user
    ).order_by(
        "-created_at"
    )

    return render(
        request,
        "predictor/history.html",
        {
            "predictions": predictions
        }
    )