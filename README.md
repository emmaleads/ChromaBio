# Dye λmax & Colour Predictor

Enter a dye (SMILES or name) and get its **predicted absorption maximum (λmax)**, the **colour you would see**, its **structure**, and **key properties**. A safety and use-case suitability module (structural alerts, regulatory flags, toxicity predictions) is planned as the next layer.

> Pitch: *Is this dye the right colour, and is it safe for my purpose?*

![screenshot](docs/screenshot.png)  <!-- add a screenshot of the app -->

## How it works
1. **Model**: a Chemprop graph neural network takes the dye and solvent as two molecular inputs and predicts λmax, absorption band width (FWHM) and intensity (log ε).
2. **Colour**: a Gaussian absorption band is built from the predictions, passed through Beer-Lambert at a chosen concentration, and converted to sRGB with CIE 1931 colour matching functions. The result is the colour of the *transmitted* light.
3. **Chemistry**: RDKit provides the structure drawing, key properties and heuristic dye-class detection.

## Results (held-out dyes)
The split is grouped by chromophore, so no dye in the test set appears in training. This measures performance on genuinely new dyes.

| Target | MAE | R² |
|---|---|---|
| λmax (nm) | 22.5 | 0.88 |
| Band width, FWHM (nm) | 12.3 | 0.64 |
| log ε | 0.21 | 0.60 |

## Run it
```bash
git clone <your-repo-url>
cd dye-lambda-max-predictor
pip install -r requirements.txt
streamlit run app/app.py
```
The trained model must be at `models/best.pt`.

## Reproduce training
Open `notebooks/train_chemprop.ipynb` in Google Colab (GPU runtime) and run all cells. Data splits used for the reported results are in `data/splits/`.

## Repository layout
```
app/app.py            Streamlit app
src/predict.py        SMILES -> λmax, width, intensity (Chemprop)
src/dye_colour.py     absorption band -> perceived colour
src/properties.py     RDKit properties, dye class, structure image
models/best.pt        trained model
data/                 dataset and train/val/test splits
notebooks/            training notebook
```

## Limitations
- Predictions are model estimates, not measurements. Typical λmax error on unseen dyes is about ±22 nm.
- The colour model uses a single absorption band. Dyes with a strong second band (e.g. malachite green) can differ from the predicted hue.
- The training data covers organic chromophores in common solvents; unusual structures may be outside its domain.
- Screening tool only. Not regulatory, safety or legal advice.

## Data and citation
Joung, J. F., Han, M., Jeong, M. & Park, S. *Experimental database of optical properties of organic compounds.* Scientific Data 7, 295 (2020). Please check the dataset licence on its Figshare page before redistributing.
