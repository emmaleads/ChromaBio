"""Streamlit app: dye -> lambda_max, colour, structure, key properties."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st

from src.dye_colour import dye_colour
from src.predict import MODEL_PATH, name_to_smiles, predict_dye
from src.safety import ENDPOINT_LABELS, LEVEL_EMOJI, USE_LABELS, USES, assess
from src.properties import canonical, dye_classes, key_properties, parse, structure_image

MAE_NM = 22.5  # held-out-dye MAE of the lambda_max model

# Approximate literature lambda_max (nm) for reference only; verify before citing.
DEMO_DYES = {
    "Methylene blue": ("CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2", 664),
    "Rhodamine B": ("CCN(CC)C1=CC2=C(C=C1)C(=C3C=CC(=[N+](CC)CC)C=C3O2)C4=CC=CC=C4C(=O)O", 554),
    "Crystal violet": ("CN(C)c1ccc(cc1)C(=C1C=CC(C=C1)=[N+](C)C)c1ccc(cc1)N(C)C", 590),
    "Tartrazine": ("C1=CC(=CC=C1N=NC2C(=NN(C2=O)C3=CC=C(C=C3)S(=O)(=O)O)C(=O)O)S(=O)(=O)O", 427),
    "Sudan I": ("C1=CC=C(C=C1)N=NC2=C(C=CC3=CC=CC=C32)O", 480),
    "Indigo": ("O=C1c2ccccc2NC1=C1Nc2ccccc2C1=O", 605),
}
SOLVENTS = {"Water": "O", "Ethanol": "CCO", "DMSO": "CS(C)=O", "Acetonitrile": "CC#N",
            "Dichloromethane": "ClCCl", "Toluene": "Cc1ccccc1"}

st.set_page_config(page_title="ChromaBio", page_icon="🎨", layout="wide")
st.title("🎨 ChromaBio")
st.caption("Enter a dye, get its predicted absorption maximum, the colour you would see, "
           "its structure and key properties. Predictions are model estimates, not measurements.")

if not MODEL_PATH.exists():
    st.error(f"Trained model not found at {MODEL_PATH}. Copy best.pt into the models/ folder.")
    st.stop()


@st.cache_data(show_spinner="Predicting…")
def cached_predict(smiles, solvent):
    return predict_dye(smiles, solvent)


with st.sidebar:
    st.header("Input")
    mode = st.radio("Choose input", ["Example dye", "SMILES", "Name (PubChem)"])
    default_ref = None
    if mode == "Example dye":
        pick = st.selectbox("Dye", list(DEMO_DYES))
        smiles, default_ref = DEMO_DYES[pick]
    elif mode == "SMILES":
        smiles = st.text_input("SMILES", value="")
    else:
        name = st.text_input("Dye name", value="")
        smiles = name_to_smiles(name) if name else ""
        if name and not smiles:
            st.warning("Could not resolve that name (or no internet). Try a SMILES instead.")
    solvent_name = st.selectbox("Solvent", list(SOLVENTS))
    conc = st.select_slider("Display concentration (µM)", options=[1, 2, 5, 10, 20, 50, 100], value=20)
    go = st.button("Predict", type="primary")

if not smiles:
    st.info("Choose an example dye or enter a SMILES or name in the sidebar.")
    st.stop()

mol = parse(smiles)
if mol is None:
    st.error("That SMILES could not be parsed by RDKit.")
    st.stop()

smi = canonical(mol)
try:
    pred = cached_predict(smi, SOLVENTS[solvent_name])
except Exception as e:  # noqa: BLE001
    st.error(str(e))
    st.stop()

lmax, fwhm, log_eps = pred["abs_nm"], max(pred["abs_fwhm_nm"], 20.0), pred["log_eps"]
col = dye_colour(lmax, fwhm, log_eps=log_eps, conc_molar=conc * 1e-6)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Structure")
    st.image(structure_image(mol))
    classes = dye_classes(mol)
    st.write("**Dye class (heuristic):** " + (", ".join(classes) if classes else "not recognised"))
    st.code(smi, language=None)
with c2:
    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("λmax (nm)", f"{lmax:.0f}", help=f"Typical error on unseen dyes: ±{MAE_NM:.0f} nm")
    m2.metric("Band width (nm)", f"{pred['abs_fwhm_nm']:.0f}")
    m3.metric("log ε", f"{log_eps:.2f}")
    if default_ref:
        st.caption(f"Approx. literature λmax for this dye: {default_ref} nm (verify before citing)")
    if lmax < 400:
        st.warning("Predicted absorption is at or below the violet edge: likely colourless or pale yellow.")

    st.markdown("**Predicted colour (transmitted light)**")
    st.markdown(f"<div style='width:100%;height:90px;border-radius:10px;background:{col['hex']};"
                f"border:1px solid #888'></div>", unsafe_allow_html=True)
    st.caption(f"{col['hex']}  ·  swatches below show the colour at λmax − {MAE_NM:.0f} nm, "
               f"λmax, and λmax + {MAE_NM:.0f} nm (model uncertainty)")
    sw = st.columns(3)
    for box, shift in zip(sw, (-MAE_NM, 0, MAE_NM)):
        h = dye_colour(lmax + shift, fwhm, log_eps=log_eps, conc_molar=conc * 1e-6)["hex"]
        box.markdown(f"<div style='height:40px;border-radius:8px;background:{h};border:1px solid #888'></div>",
                     unsafe_allow_html=True)

st.subheader("Predicted absorption band")
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.plot(col["wavelengths"], col["absorbance"], color="black")
ax.fill_between(col["wavelengths"], col["absorbance"], color=col["hex"], alpha=0.0)
ax.axvline(lmax, ls="--", color="gray")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Absorbance (approx.)")
ax.set_xlim(380, 780)
st.pyplot(fig)
st.caption("Single-band approximation. Dyes with a second strong band (e.g. malachite green) "
           "can show a different real colour.")

st.subheader("Key properties")
props = key_properties(mol)
st.table({"Property": list(props), "Value": [str(v) for v in props.values()]})

st.subheader("Safety and use-case suitability")
use_admet = st.checkbox("Include ADMET-AI toxicity predictions (slower; needs the admet-ai package)", value=False)


@st.cache_data(show_spinner="Running safety checks…")
def cached_assess(smiles, use_admet):
    return assess(smiles, use_admet)


safety = cached_assess(smi, use_admet)
st.table({"Application": [USE_LABELS[u] for u in USES],
          "Verdict": [LEVEL_EMOJI[safety["verdicts"][u]["level"]] for u in USES],
          "Why": [safety["verdicts"][u]["why"] for u in USES]})
st.caption("🟢 no alerts found (not a guarantee of safety) · 🟡 check further · 🔴 hard rule triggered · "
           "Rules decide red; model predictions can only raise green to amber.")

with st.expander("Evidence behind the verdicts"):
    azo = safety["azo"]
    if azo["has_azo"]:
        st.markdown("**Azo cleavage products (amines):** " + ", ".join(f"`{a}`" for a in azo["amines"]))
        st.markdown("**Restricted amines released:** " + (", ".join(r["name"] for r in azo["restricted"]) or "none found"))
    else:
        st.markdown("**Azo bonds:** none found (azo drawn as N=N only).")
    st.markdown("**Structural alerts:** " + (", ".join(safety["alerts"]) or "none"))
    reg = safety["regulatory"]
    st.markdown("**Regulatory lookup:** " + (f"entry found for {reg['name']}" if reg else "no entry for this structure"))
    if use_admet:
        if safety["admet"]:
            st.markdown("**ADMET-AI predicted probabilities** (model estimates, trained mostly on drug-like molecules, "
                        "so less reliable for dyes):")
            st.table({"Endpoint": [ENDPOINT_LABELS.get(k, k) for k in safety["admet"]],
                      "Probability": [f"{v:.2f}" for v in safety["admet"].values()]})
        else:
            st.warning(safety["admet_error"] or "No ADMET predictions available.")

st.divider()
st.caption("Screening tool only. Not regulatory, safety or legal advice.")
