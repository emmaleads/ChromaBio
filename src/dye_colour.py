"""Predict the perceived colour of a dye solution from its absorption band.

Inputs (e.g. from the Chemprop model): lambda_max (nm), FWHM (nm), log10(epsilon).
Output: sRGB colour of the *transmitted* light, i.e. what you see.
"""
import numpy as np

WL = np.arange(380, 781, 1.0)  # visible range, nm


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _cmf(wl):
    """CIE 1931 colour matching functions (Wyman, Sloan & Shirley 2013 fit)."""
    x = 1.056 * _g(wl, 599.8, 37.9, 31.0) + 0.362 * _g(wl, 442.0, 16.0, 26.7) - 0.065 * _g(wl, 501.1, 20.4, 26.2)
    y = 0.821 * _g(wl, 568.8, 46.9, 40.5) + 0.286 * _g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _g(wl, 437.0, 11.8, 36.0) + 0.681 * _g(wl, 459.0, 26.0, 13.8)
    return x, y, z


def _illuminant(wl, T=6504.0):
    """Blackbody at ~D65 temperature, a simple stand-in for daylight."""
    l = wl * 1e-9
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    return 1.0 / (l ** 5 * (np.exp(h * c / (l * k * T)) - 1.0))


def absorbance_spectrum(lmax, fwhm=50.0, peak_abs=1.0):
    sigma = max(fwhm, 5.0) / 2.355
    return peak_abs * np.exp(-0.5 * ((WL - lmax) / sigma) ** 2)


def peak_absorbance(log_eps, conc_molar=2e-5, path_cm=1.0):
    """Beer-Lambert: A = epsilon * c * l."""
    return (10 ** log_eps) * conc_molar * path_cm


def _xyz_to_srgb(xyz):
    m = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(m @ xyz, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def dye_colour(lmax, fwhm=50.0, log_eps=None, peak_abs=None, conc_molar=2e-5):
    """Return dict with hex colour, RGB (0-255) and the spectrum.

    Intensity: pass `peak_abs` directly, or `log_eps` (+ concentration), or neither
    (defaults to peak absorbance 1.0 so the hue is clearly visible).
    """
    if peak_abs is None:
        peak_abs = peak_absorbance(log_eps, conc_molar) if log_eps is not None else 1.0
    A = absorbance_spectrum(lmax, fwhm, peak_abs)
    trans = 10 ** (-A)
    x, y, z = _cmf(WL)
    ill = _illuminant(WL)
    k = 1.0 / np.sum(ill * y)  # white (no absorber) -> Y = 1
    xyz = k * np.array([np.sum(trans * ill * c) for c in (x, y, z)])
    white = k * np.array([np.sum(ill * c) for c in (x, y, z)])
    xyz = xyz * (np.array([0.9505, 1.0, 1.089]) / white)  # adapt illuminant to D65 white
    rgb = (_xyz_to_srgb(xyz) * 255).round().astype(int)
    return {"hex": "#%02x%02x%02x" % tuple(rgb), "rgb": tuple(int(v) for v in rgb),
            "wavelengths": WL, "absorbance": A}


if __name__ == "__main__":
    tests = [("Tartrazine (yellow)", 427, 80), ("Rhodamine B (pink/red)", 554, 40),
             ("Methylene blue (blue)", 664, 55), ("Crystal violet (violet)", 590, 50),
             ("Malachite green (green)", 617, 50), ("UV-only (colourless)", 300, 40)]
    for n, l, w in tests:
        print(f"{n:28s} lmax={l}  ->  {dye_colour(l, w)['hex']}")
