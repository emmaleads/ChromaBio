"""SMILES (+ solvent) -> lambda_max, band width, intensity, using the trained Chemprop model."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"
TARGETS = ["abs_nm", "abs_fwhm_nm", "log_eps"]


def _chemprop_cmd():
    exe = shutil.which("chemprop")
    if exe:
        return [exe]
    return [sys.executable, "-c", "from chemprop.cli.main import main; main()"]


def predict_dye(smiles, solvent="O", model_path=MODEL_PATH):
    """Return {'abs_nm': ..., 'abs_fwhm_nm': ..., 'log_eps': ...} for one dye in one solvent."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Copy best.pt into models/.")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in.csv"), os.path.join(tmp, "out.csv")
        pd.DataFrame({"Chromophore": [smiles], "Solvent": [solvent]}).to_csv(inp, index=False)
        cmd = _chemprop_cmd() + ["predict", "-i", inp, "--smiles-columns", "Chromophore", "Solvent",
                                 "--model-paths", str(model_path), "-o", out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("Chemprop prediction failed:\n" + res.stderr[-1500:])
        r = pd.read_csv(out)
    vals = r[[c for c in r.columns if c not in ("Chromophore", "Solvent")][-3:]].values[0]
    return dict(zip(TARGETS, (float(v) for v in vals)))


def name_to_smiles(name, timeout=8):
    """Resolve a compound name to SMILES via PubChem (needs internet). Returns None on failure."""
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name) + "/property/CanonicalSMILES/JSON")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            props = json.load(resp)["PropertyTable"]["Properties"][0]
        for k, v in props.items():
            if "SMILES" in k:
                return v
    except Exception:
        return None
    return None
