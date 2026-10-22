import requests
from urllib.parse import quote


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

        return data["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]

    except Exception as e:
        print("Connection error:", e)
        return None


drug = input("Enter drug name: ")

smiles = get_smiles(drug)

if smiles:
    print("\nDrug:", drug)
    print("SMILES:", smiles)
else:
    print("\nDrug not found or PubChem is temporarily unavailable.")