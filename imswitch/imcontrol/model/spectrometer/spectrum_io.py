"""Save and load spectra: HDF5 (one file) or CSV with a JSON sidecar.

A record is a dict with 1-D arrays "px" and "counts", optional "nm", "ramanShiftCm1",
"dark" (collapsed dark), "roi" (2-D mean frame region) and a JSON-able "metadata".
"""
import csv
import json
import os
from typing import Dict, List

import numpy as np

AXES = ("px", "nm", "ramanShiftCm1", "counts", "dark")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, default=_json_default)


def save_hdf5(path: str, record: Dict) -> str:
    import h5py
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("spectrum")
        for key in AXES:
            if record.get(key) is not None:
                g.create_dataset(key, data=np.asarray(record[key], dtype=np.float64))
        if record.get("roi") is not None:
            f.create_dataset("roi", data=np.asarray(record["roi"], dtype=np.float32),
                             compression="gzip", compression_opts=4)
        f.attrs["metadata"] = to_json(record.get("metadata", {}))
        f.attrs["format"] = "imswitch-spectrum/1"
    return path


def load_hdf5(path: str) -> Dict:
    import h5py
    with h5py.File(path, "r") as f:
        out = {k: f["spectrum"][k][()] for k in f["spectrum"]}
        if "roi" in f:
            out["roi"] = f["roi"][()]
        out["metadata"] = json.loads(f.attrs["metadata"])
    return out


def save_csv(path: str, record: Dict) -> List[str]:
    """CSV with one column per axis, plus <name>.json with the metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [k for k in AXES if record.get(k) is not None]
    data = np.column_stack([np.asarray(record[k], dtype=np.float64) for k in cols])
    heads = {"px": "pixel", "nm": "wavelength_nm", "ramanShiftCm1": "raman_shift_cm-1",
             "counts": "counts", "dark": "dark_counts"}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([heads[c] for c in cols])
        for row in data:
            w.writerow([f"{v:.6g}" for v in row])
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write(to_json(record.get("metadata", {})))
    return [path, meta_path]


def load_csv(path: str) -> Dict:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    back = {"pixel": "px", "wavelength_nm": "nm", "raman_shift_cm-1": "ramanShiftCm1",
            "counts": "counts", "dark_counts": "dark"}
    data = np.array(rows[1:], dtype=np.float64)
    out = {back[h]: data[:, i] for i, h in enumerate(rows[0])}
    meta_path = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            out["metadata"] = json.load(fh)
    return out
