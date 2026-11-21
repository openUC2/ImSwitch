"""Virtual lightsheet microscope: a 3D phantom seen through a Gaussian light sheet.

Lab frame (µm): the detection objective looks along z; the camera sees x (columns)
and y (rows). The sheet travels along +x (enters at the left edge of the image), is
thin in z and wide in y, and has its waist in the centre of the image at z = 0.

Stage (configurable): X, Y move the sample in the image plane, ``sampleScanAxis``
(default A, as on the openUC2 rig) pushes it through the sheet, ``objectiveAxis`` (default Z) moves the
detection objective (focal plane at z = objective + focusOffsetUm).
A sample point s therefore shows up at lab position s + (X, Y, scan).

Optics (paraxial, Gaussian):
  sheet waist      w0 = λex / (π · NAill)            (1/e² half thickness)
  Rayleigh range   zR = π · w0² · n / λex
  sheet thickness  w(x) = w0 · sqrt(1 + ((x - xWaist) / zR)²)
  sheet intensity  (w0 / w(x)) · exp(-2 z² / w(x)²) · exp(-(x - xEntry) / attenuation)
  detection PSF    σ0 = 0.21 · λem / NAdet, widened by the geometric defocus blur
                   |z - zFocus| · tan(asin(NAdet / n)) / 2
So a low illumination NA gives a thick, even sheet (poor sectioning); a high one a thin
sheet that widens towards the image edges; a misplaced objective (A) blurs everything.
"""
import math

import cv2
import numpy as np

DEFAULTS = dict(
    width=512, height=512,           # camera pixels
    magnification=10.0,
    detectionNA=0.2,
    illuminationNA=0.05,
    cameraPixelUm=3.45,              # physical camera pixel; sample pixel = this / magnification
    excitationNm=488.0, emissionNm=520.0,
    refractiveIndex=1.33,            # water / agarose
    sheetWaistOffsetUm=0.0,          # waist position along x relative to the image centre
    attenuationUm=1500.0,            # illumination 1/e decay into the sample (scattering)
    focusOffsetUm=0.0,               # objective focal plane at z = A + this (0: in focus at A = 0)
    sampleScanAxis="A",              # stage axis that moves the sample through the sheet (openUC2: A)
    objectiveAxis="Z",               # stage axis that moves the detection objective
    sampleSizeUm=(1500.0, 1500.0, 800.0),
    nSpheroids=40, nBeads=30000, seed=0,
    brightness=150.0,                # counts per ms at full laser for a cell in the sheet waist
    background=100.0, readNoise=2.0, bitDepth=12,
)


class LightsheetSimulator:
    def __init__(self, **params):
        p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
        self.p = p
        self.width, self.height = int(p["width"]), int(p["height"])
        self.pixelUm = p["cameraPixelUm"] / p["magnification"]
        self.rng = np.random.default_rng(p["seed"])
        self._buildPhantom()
        self.setIlluminationNA(p["illuminationNA"])

    # ------------------------------------------------------------------ optics
    def setIlluminationNA(self, na: float):
        p = self.p
        p["illuminationNA"] = float(min(max(na, 1e-3), 0.9 * p["refractiveIndex"]))
        lam = p["excitationNm"] / 1000.0
        self.w0 = lam / (math.pi * p["illuminationNA"])
        self.zR = math.pi * self.w0 ** 2 * p["refractiveIndex"] / lam

    def optics(self) -> dict:
        """What the current settings mean optically (shown in the UI)."""
        p = self.p
        lam_em = p["emissionNm"] / 1000.0
        return {
            "illuminationNA": p["illuminationNA"], "detectionNA": p["detectionNA"],
            "magnification": p["magnification"], "pixelSizeUm": self.pixelUm,
            "fovUm": [self.width * self.pixelUm, self.height * self.pixelUm],
            "sheetWaistUm": 2 * self.w0,                       # full 1/e² thickness
            "sheetRayleighRangeUm": self.zR,
            "sheetUsableFieldUm": 2 * self.zR,                 # where the sheet stays < √2·waist
            "lateralResolutionUm": 0.61 * lam_em / p["detectionNA"],
            "depthOfFieldUm": lam_em * p["refractiveIndex"] / p["detectionNA"] ** 2,
            "sampleScanAxis": p["sampleScanAxis"], "objectiveAxis": p["objectiveAxis"],
        }

    def _sheetThickness(self, x_um: np.ndarray) -> np.ndarray:
        return self.w0 * np.sqrt(1 + ((x_um - self.p["sheetWaistOffsetUm"]) / self.zR) ** 2)

    def _blurSigmaPx(self, dz_um: float) -> float:
        p = self.p
        na, n = p["detectionNA"], p["refractiveIndex"]
        sigma0 = 0.21 * (p["emissionNm"] / 1000.0) / na
        geometric = abs(dz_um) * math.tan(math.asin(min(na / n, 0.99))) / 2
        return math.hypot(sigma0, geometric) / self.pixelUm

    # ----------------------------------------------------------------- phantom
    def _buildPhantom(self):
        """Spheroids (shells of 5 µm nuclei around a hollow core) and sub-resolution beads,
        in a box centred on the origin; one spheroid sits at the origin."""
        p, rng = self.p, self.rng
        size = np.asarray(p["sampleSizeUm"], dtype=float)
        centres = (rng.random((int(p["nSpheroids"]), 3)) - 0.5) * size * 0.9
        centres[0] = 0.0
        radii = rng.uniform(40, 110, len(centres))
        radii[0] = 70.0
        cells = []
        for c, R in zip(centres, radii):
            n = int(4 * math.pi * R ** 2 / 90)               # ~ one nucleus per 90 µm² of shell
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            shell = R * (1 - 0.25 * rng.random(n) ** 2)       # a few layers thick
            cells.append(c + v * shell[:, None])
        cells = np.concatenate(cells)
        self.cells = cells.astype(np.float32)
        self.cellR = rng.uniform(3.5, 6.0, len(cells)).astype(np.float32)
        self.cellB = rng.uniform(0.5, 1.0, len(cells)).astype(np.float32)
        self.beads = ((rng.random((int(p["nBeads"]), 3)) - 0.5) * size).astype(np.float32)

    # ------------------------------------------------------------------ render
    def render(self, position: dict, intensity: float, exposureMs: float, maxIntensity: float = 1024.0) -> np.ndarray:
        """One camera frame (uint16) for stage ``position`` {X, Y, Z, A}, laser ``intensity``
        (0..maxIntensity) and exposure in ms."""
        p = self.p
        W, H, px = self.width, self.height, self.pixelUm
        X, Y = float(position.get("X", 0.0)), float(position.get("Y", 0.0))
        Z = float(position.get(p["sampleScanAxis"], 0.0))   # Z: sample through the sheet, A: objective (lab names)
        A = float(position.get(p["objectiveAxis"], 0.0))
        x_lab = (np.arange(W, dtype=np.float32) - W / 2 + 0.5) * px
        w = self._sheetThickness(x_lab)
        entry = x_lab[0]
        column = (self.w0 / w) * np.exp(-(x_lab - entry) / p["attenuationUm"])

        # z planes through the sheet: ±2 widest thickness, at most 40 planes
        zmax = 2.0 * float(w.max())
        n = int(min(40, max(7, math.ceil(2 * zmax / max(self.w0 / 2, 0.25)))))
        planes = np.linspace(-zmax, zmax, n)
        dz = planes[1] - planes[0]

        # sample points that can show up in this frame (lab = sample + stage)
        half_x, half_y = W / 2 * px + 10, H / 2 * px + 10
        c = self.cells
        sel = ((np.abs(c[:, 0] + X) < half_x + self.cellR) & (np.abs(c[:, 1] + Y) < half_y + self.cellR)
               & (np.abs(c[:, 2] + Z) < zmax + self.cellR))
        cc, cr, cb = c[sel], self.cellR[sel], self.cellB[sel]
        b = self.beads
        bsel = (np.abs(b[:, 0] + X) < half_x) & (np.abs(b[:, 1] + Y) < half_y) & (np.abs(b[:, 2] + Z) < zmax + dz)
        bb = b[bsel]

        # planes grouped by detection blur (one Gaussian per group, not per plane)
        groups = {}
        for z in planes:
            slab = np.zeros((H, W), np.float32)
            dzc = cc[:, 2] + Z - z
            hit = np.abs(dzc) < cr
            rr = np.sqrt(np.maximum(cr[hit] ** 2 - dzc[hit] ** 2, 0)) / px
            cols = (cc[hit, 0] + X) / px + W / 2
            rows = (cc[hit, 1] + Y) / px + H / 2
            for cx, cy, r, v in zip(cols, rows, rr, cb[hit]):
                cv2.circle(slab, (int(cx * 16), int(cy * 16)), max(int(r * 16), 8), float(v), -1,
                           lineType=cv2.LINE_AA, shift=4)
            bhit = np.abs(bb[:, 2] + Z - z) < dz / 2
            if bhit.any():
                bc = ((bb[bhit, 0] + X) / px + W / 2).astype(int)
                br = ((bb[bhit, 1] + Y) / px + H / 2).astype(int)
                ok = (bc >= 0) & (bc < W) & (br >= 0) & (br < H)
                np.add.at(slab, (br[ok], bc[ok]), 3.0)
            wz = column * np.exp(-2 * z ** 2 / w ** 2)       # sheet profile at this depth, per column
            sigma = self._blurSigmaPx(z - (A + p["focusOffsetUm"]))
            key = round(math.log(sigma, 1.15))                # ~15 % sigma bins
            acc = groups.setdefault(key, [np.zeros((H, W), np.float32), sigma])
            acc[0] += slab * wz[None, :] * (dz / self.w0)
        image = np.zeros((H, W), np.float32)
        for acc, sigma in groups.values():
            image += cv2.GaussianBlur(acc, (0, 0), sigmaX=sigma) if sigma > 0.3 else acc

        signal = image * p["brightness"] * (intensity / maxIntensity) * max(exposureMs, 0.0)
        signal = self.rng.poisson(np.clip(signal, 0, 1e6)).astype(np.float32)
        signal += p["background"] + self.rng.normal(0, p["readNoise"], (H, W)).astype(np.float32)
        return np.clip(signal, 0, 2 ** int(p["bitDepth"]) - 1).astype(np.uint16)
