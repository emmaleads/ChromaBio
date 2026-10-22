"""Cross-check data/restricted_amines.json against PubChem, matching by CAS number.

Run from the repo root (needs internet and RDKit):
    python tools/verify_amines.py

MISMATCH means the SMILES in the file does not describe the compound that has that CAS number.
Always confirm any MISMATCH by hand on PubChem before changing the file.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]


def pubchem_smiles(cas):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(cas) + "/property/CanonicalSMILES/JSON")
    with urllib.request.urlopen(url, timeout=20) as r:
        props = json.load(r)["PropertyTable"]["Properties"][0]
    for k, v in props.items():
        if "SMILES" in k:
            return v
    return None


def canon(smiles):
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return (Chem.MolToSmiles(m), rdMolDescriptors.CalcMolFormula(m)) if m else (None, None)


def main():
    path = ROOT / "data" / "restricted_amines.json"
    amines = json.load(open(path, encoding="utf-8"))["amines"]
    bad = 0
    for i, a in enumerate(amines, 1):
        mine, mine_f = canon(a["smiles"])
        try:
            ref_raw = pubchem_smiles(a["cas"])
        except Exception as e:  # noqa: BLE001
            print(f"{i:2d} {a['cas']:>9s} {a['name']}: NOT FOUND on PubChem ({e})")
            bad += 1
            continue
        ref, ref_f = canon(ref_raw)
        if mine and ref and mine == ref:
            print(f"{i:2d} {a['cas']:>9s} OK        {a['name']}")
        else:
            bad += 1
            print(f"{i:2d} {a['cas']:>9s} MISMATCH  {a['name']}\n"
                  f"      file    : {a['smiles']}  ({mine_f})\n"
                  f"      PubChem : {ref_raw}  ({ref_f})")
        time.sleep(0.4)  # be polite to the PubChem server
    print(f"\n{len(amines) - bad} of {len(amines)} entries match PubChem.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
