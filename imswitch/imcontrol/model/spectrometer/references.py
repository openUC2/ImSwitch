"""Known lines and reference spectra for the wavelength calibration.

Line wavelengths are in air (nm), as a spectrometer in air measures them. Raman
standards are shifts in cm^-1. The NeoPixel (WS2812B) spectra here are Gaussian
placeholders built from typical datasheet values; replace them with the spectra of
your own LED ring measured on a reference spectrometer
(``load_reference_spectra``), otherwise the LED points are only good to a few nm.
"""
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

# Neon glow lamp (NE-2 and similar), strong Ne I lines, air wavelengths (NIST ASD).
NEON_LINES_NM = [
    540.056, 585.249, 588.190, 594.483, 597.553, 603.000, 607.434, 609.616, 614.306,
    616.359, 621.728, 626.650, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288,
    659.895, 667.828, 671.704, 692.947, 703.241,
]

# Compact fluorescent lamp: mercury lines (sharp, air wavelengths, NIST ASD).
MERCURY_LINES_NM = [404.656, 435.833, 546.074, 576.960, 579.066]

# Phosphor peaks of a compact fluorescent lamp (Tb3+, Eu3+): a few nm wide, approximate.
CFL_PHOSPHOR_PEAKS_NM = [487.7, 542.4, 611.6]

# Raman standards, shifts in cm^-1 (cyclohexane and polystyrene: ASTM E1840;
# acetonitrile: the strong bands AutoOpenRaman calibrates with, rounded).
RAMAN_STANDARDS_CM1 = {
    "cyclohexane": [384.1, 426.3, 801.3, 1028.3, 1157.6, 1266.4, 1444.4, 2664.4, 2852.9,
                    2923.8, 2938.3],
    "polystyrene": [620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3, 2852.4,
                    2904.5, 3054.3],
    "acetonitrile": [918.0, 1376.0, 2253.0, 2942.0],
}

LINE_LISTS_NM = {
    "neon": NEON_LINES_NM,
    "mercury": MERCURY_LINES_NM,
}

# Typical WS2812B emission (peak nm, FWHM nm). Placeholders, see the module docstring.
NEOPIXEL_PLACEHOLDER = {
    "r": (628.0, 18.0),
    "g": (520.0, 32.0),
    "b": (468.0, 24.0),
}


def gaussian_spectrum(peak_nm: float, fwhm_nm: float, lo: float = 380.0, hi: float = 780.0,
                      step: float = 0.5) -> Dict:
    """A normalised Gaussian emission spectrum on a regular grid."""
    nm = np.arange(lo, hi + step / 2, step)
    sigma = fwhm_nm / 2.3548
    return {"wavelengthNm": nm.tolist(),
            "intensity": np.exp(-0.5 * ((nm - peak_nm) / sigma) ** 2).tolist()}


def placeholder_led_spectra() -> Dict[str, Dict]:
    """NeoPixel R, G, B placeholder spectra keyed "r", "g", "b"."""
    out = {}
    for ch, (peak, fwhm) in NEOPIXEL_PLACEHOLDER.items():
        spec = gaussian_spectrum(peak, fwhm)
        spec.update(label=f"NeoPixel {ch.upper()} (placeholder, {peak:g} nm)", placeholder=True)
        out[ch] = spec
    return out


def white_phosphor_led_spectrum() -> Dict:
    """A phone-flash-like white LED (blue die at 450 nm + broad phosphor), for the simulator."""
    nm = np.arange(380.0, 780.5, 0.5)
    blue = np.exp(-0.5 * ((nm - 450.0) / 9.0) ** 2)
    phosphor = 0.55 * np.exp(-0.5 * ((nm - 565.0) / 55.0) ** 2)
    return {"wavelengthNm": nm.tolist(), "intensity": (blue + phosphor).tolist(),
            "label": "white phosphor LED (model)", "placeholder": True}


def normalise_spectrum(spec: Dict) -> Dict:
    """Sorted, finite, peak-normalised copy of {"wavelengthNm": [...], "intensity": [...]}."""
    nm = np.asarray(spec["wavelengthNm"], dtype=float)
    val = np.asarray(spec["intensity"], dtype=float)
    if nm.shape != val.shape or nm.size < 3:
        raise ValueError("a reference spectrum needs matching wavelength and intensity arrays")
    ok = np.isfinite(nm) & np.isfinite(val)
    nm, val = nm[ok], val[ok]
    order = np.argsort(nm)
    nm, val = nm[order], val[order]
    peak = float(np.max(val))
    if peak <= 0:
        raise ValueError("a reference spectrum needs positive intensities")
    out = dict(spec)
    out.update(wavelengthNm=nm.tolist(), intensity=(val / peak).tolist())
    return out


def _read_csv(path: str) -> Dict[str, Dict]:
    """CSV with a header: wavelength column first, then one column per source."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]
    if len(rows) < 4:
        raise ValueError(f"{path}: needs a header and at least three rows")
    header, body = rows[0], rows[1:]
    data = np.array([[float(v) for v in r[:len(header)]] for r in body], dtype=float)
    out = {}
    for j, name in enumerate(header[1:], start=1):
        key = name.strip() or f"col{j}"
        out[key] = normalise_spectrum({"wavelengthNm": data[:, 0], "intensity": data[:, j],
                                       "label": key, "placeholder": False})
    return out


def load_reference_spectra(path: str) -> Dict[str, Dict]:
    """Reference spectra from a JSON or CSV file, keyed by source/channel name.

    JSON: {"r": {"wavelengthNm": [...], "intensity": [...]}, ...} (or the same under
    "sources"). CSV: header "wavelength_nm,r,g,b" (any column names). Spectra are
    normalised to a peak of 1.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.lower().endswith((".csv", ".txt")):
        return _read_csv(path)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc = doc.get("sources", doc)
    return {key: normalise_spectrum({"placeholder": False, "label": key, **spec})
            for key, spec in doc.items()}


def spectrum_at(spec: Dict, nm: np.ndarray) -> np.ndarray:
    """Reference intensity interpolated at nm (0 outside the measured range)."""
    return np.interp(nm, spec["wavelengthNm"], spec["intensity"], left=0.0, right=0.0)


def line_list(name: str) -> Optional[List[float]]:
    return LINE_LISTS_NM.get(name)
