"""Safety and use-case suitability.

Three layers, from most to least certain:
  1. Hard rules   : azo cleavage -> restricted aromatic amines, structural alerts, regulatory lookup
  2. Model output : optional ADMET-AI toxicity predictions (probabilities, secondary evidence)
  3. Verdicts     : traffic light per application. Rules decide red; models can only raise green -> amber.

Screening tool only. Not regulatory, safety or legal advice.
"""
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

DATA = Path(__file__).resolve().parents[1] / "data"
AZO = Chem.MolFromSmarts("[#6][NX2;!R]=[NX2;!R][#6]")  # azo drawn as N=N, not in a ring
NITRO_AROM = Chem.MolFromSmarts("c[N+](=O)[O-]")
ARYL_NH2 = Chem.MolFromSmarts("c[NX3;H2]")

MODEL_FLAG = 0.5  # ADMET-AI probability at or above this raises a flag
TOX_ENDPOINTS = ["AMES", "DILI", "hERG", "ClinTox", "Skin_Reaction", "Carcinogens_Lagunin"]
ENDPOINT_LABELS = {"AMES": "Mutagenicity (Ames)", "DILI": "Liver injury (DILI)", "hERG": "hERG blockade",
                   "ClinTox": "Clinical toxicity", "Skin_Reaction": "Skin reaction",
                   "Carcinogens_Lagunin": "Carcinogenicity"}
USES = ["textile", "food", "cosmetic", "stain"]
USE_LABELS = {"textile": "Textiles", "food": "Food", "cosmetic": "Cosmetics", "stain": "Biological stain"}
LEVEL_EMOJI = {"red": "🔴", "amber": "🟡", "green": "🟢", "grey": "⚪"}


def _canon(smiles):
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(m) if m else None


def _load_json(name):
    path = DATA / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_RESTRICTED = None
_REGULATORY = None
_BRENK = None
_ADMET = None


def restricted_amines():
    """{canonical smiles: entry} from data/restricted_amines.json."""
    global _RESTRICTED
    if _RESTRICTED is None:
        _RESTRICTED = {}
        for e in (_load_json("restricted_amines.json") or {}).get("amines", []):
            c = _canon(e.get("smiles"))
            if c:
                _RESTRICTED[c] = e
    return _RESTRICTED


def regulatory_db():
    """{canonical smiles: entry} from data/regulatory.json."""
    global _REGULATORY
    if _REGULATORY is None:
        _REGULATORY = {}
        for e in (_load_json("regulatory.json") or {}).get("dyes", []):
            c = _canon(e.get("smiles"))
            if c:
                _REGULATORY[c] = e
    return _REGULATORY


# ---------- layer 1: hard rules ----------
def azo_cleavage(mol):
    """Cleave every azo bond, cap the fragments as amines (N=N -> 2 x NH2), compare to the restricted list."""
    out = {"has_azo": False, "amines": [], "restricted": []}
    matches = mol.GetSubstructMatches(AZO)
    if not matches:
        return out
    out["has_azo"] = True
    rw = Chem.RWMol(mol)
    cleaved = set()
    for _, n1, n2, _ in matches:
        rw.RemoveBond(n1, n2)
        cleaved.update((n1, n2))
    for i in cleaved:
        a = rw.GetAtomWithIdx(i)
        a.SetNumExplicitHs(2)
        a.SetNoImplicit(True)
    Chem.SanitizeMol(rw)
    index_groups = Chem.GetMolFrags(rw, asMols=False)
    frag_mols = Chem.GetMolFrags(rw, asMols=True, sanitizeFrags=True)
    table = restricted_amines()
    for idx, frag in zip(index_groups, frag_mols):
        if not cleaved.intersection(idx):
            continue
        smi = Chem.MolToSmiles(frag)
        out["amines"].append(smi)
        if smi in table:
            out["restricted"].append({"smiles": smi, "name": table[smi].get("name", smi),
                                      "cas": table[smi].get("cas", "")})
    return out


def structural_alerts(mol):
    """Generic alerts: RDKit Brenk catalogue plus a few dye-relevant checks."""
    global _BRENK
    if _BRENK is None:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _BRENK = FilterCatalog(p)
    alerts = sorted({m.GetDescription() for m in _BRENK.GetMatches(mol)})
    if mol.HasSubstructMatch(NITRO_AROM) and "Nitroaromatic" not in alerts:
        alerts.append("Nitroaromatic")
    if mol.HasSubstructMatch(ARYL_NH2) and "Primary aromatic amine" not in alerts:
        alerts.append("Primary aromatic amine")
    return alerts


def is_restricted_amine(mol):
    return restricted_amines().get(Chem.MolToSmiles(mol))


def regulatory_entry(mol):
    return regulatory_db().get(Chem.MolToSmiles(mol))


# ---------- layer 2: optional ADMET-AI ----------
def admet_predict(smiles):
    """Return (dict of endpoint -> probability, error message). Lazy import keeps the app light."""
    global _ADMET
    try:
        from admet_ai import ADMETModel
    except ImportError:
        return None, "admet-ai is not installed (pip install admet-ai)."
    try:
        if _ADMET is None:
            _ADMET = ADMETModel()
        preds = _ADMET.predict(smiles=smiles)
        return {k: float(preds[k]) for k in TOX_ENDPOINTS if k in preds}, None
    except Exception as e:  # noqa: BLE001
        return None, f"ADMET-AI failed: {e}"


# ---------- layer 3: verdicts ----------
def _V(level, why):
    return {"level": level, "why": why}


def _status(entry, use):
    if not entry:
        return None, ""
    d = (entry.get("uses") or {}).get(use) or {}
    s = (d.get("status") or "").lower()
    return (None if s in ("", "todo") else s), d.get("source", "")


def _flag(admet, key):
    return bool(admet) and admet.get(key, 0.0) >= MODEL_FLAG


def verdicts(azo, direct_restricted, entry, admet):
    names = [r["name"] for r in azo["restricted"]]
    if direct_restricted:
        names.append(direct_restricted.get("name", "restricted amine"))
    ames, skin = _flag(admet, "AMES"), _flag(admet, "Skin_Reaction")
    out = {}

    # textiles
    st, src = _status(entry, "textile")
    if names:
        out["textile"] = _V("red", "Releases or is a restricted aromatic amine: " + ", ".join(names))
    elif st in ("banned", "restricted"):
        out["textile"] = _V("red", f"Listed as {st} for textiles ({src or 'see lookup'})")
    elif st == "approved":
        out["textile"] = _V("green", f"Listed as permitted for textiles ({src or 'see lookup'})")
    elif azo["has_azo"]:
        out["textile"] = _V("amber", "Azo dye: no listed amine released in this check, but confirm with "
                                     "standard textile amine testing")
    else:
        out["textile"] = _V("green", "No restricted-amine route found (screening check only)")

    # food
    st, src = _status(entry, "food")
    if st in ("banned", "restricted", "not_approved"):
        out["food"] = _V("red", f"Not permitted as a food colour ({src or 'see lookup'})")
    elif names:
        out["food"] = _V("red", "Releases a restricted aromatic amine: " + ", ".join(names))
    elif st == "approved":
        out["food"] = _V("amber" if ames else "green",
                         f"Listed as authorised ({src or 'see lookup'})"
                         + ("; model also flags possible mutagenicity" if ames else ""))
    else:
        out["food"] = _V("amber", "No approval record in the lookup. Only authorised food colours may be "
                                  "used, so verify before any food use")

    # cosmetics
    st, src = _status(entry, "cosmetic")
    if st in ("banned", "not_approved"):
        out["cosmetic"] = _V("red", f"Not permitted in cosmetics ({src or 'see lookup'})")
    elif st == "approved" and not (names or ames or skin):
        out["cosmetic"] = _V("green", f"Listed as permitted ({src or 'see lookup'})")
    elif names or ames or skin or st == "restricted":
        why = []
        if names: why.append("releases a restricted amine")
        if ames: why.append("model flags possible mutagenicity")
        if skin: why.append("model flags possible skin reaction")
        if st == "restricted": why.append("restricted in the lookup")
        out["cosmetic"] = _V("amber", "Check carefully: " + "; ".join(why))
    else:
        out["cosmetic"] = _V("amber", "No record in the lookup. Check the Colour Index number against the "
                                      "cosmetics colourant annexes")

    # biological stain (a handling question, not a regulatory one)
    if names or ames:
        why = "releases a restricted amine" if names else "model flags possible mutagenicity"
        out["stain"] = _V("amber", f"Usable as a stain in principle, but handle as a hazard: {why}")
    else:
        out["stain"] = _V("green", "No hard alerts found. Handle according to the safety data sheet")
    return out


def assess(smiles, use_admet=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    azo = azo_cleavage(mol)
    direct = is_restricted_amine(mol)
    entry = regulatory_entry(mol)
    admet, admet_err = (None, None)
    if use_admet:
        admet, admet_err = admet_predict(Chem.MolToSmiles(mol))
    return {
        "azo": azo,
        "direct_restricted": direct,
        "alerts": structural_alerts(mol),
        "regulatory": entry,
        "admet": admet,
        "admet_error": admet_err,
        "verdicts": verdicts(azo, direct, entry, admet),
    }


if __name__ == "__main__":
    # quick self-test: python -m src.safety
    cases = {
        "o-aminoazotoluene": "Cc1ccccc1N=Nc1ccc(N)c(C)c1",
        "4-aminoazobenzene": "Nc1ccc(cc1)N=Nc1ccccc1",
        "Sudan I": "C1=CC=C(C=C1)N=NC2=C(C=CC3=CC=CC=C32)O",
        "Methylene blue": "CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2",
    }
    for n, s in cases.items():
        r = assess(s)
        print(n, "| azo:", r["azo"]["has_azo"], "| amines:", r["azo"]["amines"],
              "| restricted:", [x["name"] for x in r["azo"]["restricted"]],
              "| textile:", r["verdicts"]["textile"]["level"])
