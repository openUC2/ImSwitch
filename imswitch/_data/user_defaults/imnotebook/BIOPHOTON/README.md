# BIOPHOTON: experiments for the UPE imaging paper

Notebooks for the measurements David asked for (call of July 2026) towards the paper
*An open, affordable two-objective microscope for spatially resolved biophoton imaging*.
All samples: injured leaves in 2 % H2O2. All acquisitions go through ImSwitch with `imswitchclient`.

| Order | Notebook | What | Camera | Output |
|---|---|---|---|---|
| 1 | `E3_andor_long_exposure.ipynb` | 30 min frames back to back, ~12 h. Does the sample glow at all? | Andor iXon DU888, 4x4, EM 300, 1 MHz | signal over time, mean image |
| 2 | `E1_andor_photon_counting.ipynb` | 90 s frames, threshold → photon events, ~12 h. The reference photon budget | Andor, as above | events/px/s over time, event map |
| 3 | `E2_touptek_ptc_calibration.ipynb` | Photon transfer with the beta light (20 bright + 20 dark) and a dark ladder | ToupTek MTR3CMOS09000KMA, gain 23, HCG, -30 °C | e-/ADU, read noise, offset, dark current vs datasheet |
| 4 | `E4_touptek_long_exposure.ipynb` | Same protocol as E3, fixed black level 1000, no dark frames | ToupTek, as above | signal over time in e-/px/s (with E2) |
| any | `E0_environment_log.ipynb` | Camera + enclosure temperature, humidity, light, from one or both servers | - | environment.csv + plots |

E1-E4 log their own camera and the enclosure sensors (SHT45, TSL2591) into their run folder;
`OTHER_SERVERS` adds the other rig. E0 is for logging when no experiment runs.

## Setup

```bash
pip install -U https://github.com/openUC2/imswitchclient/archive/refs/heads/main.zip
```

Needs imswitchclient newer than 0.1.7 (`recordingManager.snapAndWait`), plus numpy, matplotlib
and tifffile, which ImSwitch already ships. Open the notebooks from this folder so
`import biophoton` finds `biophoton.py`. Each notebook starts with `DRY_RUN = True`: three
short frames through the whole chain. Set it to `False` for the real run.

ToupTek gain: ImSwitch maps its 0..23 scale linearly onto the camera's native analog gain range
(percent, 100 = 1x), so `23` is always this model's maximum and `0` its minimum.

## Data

Each run writes one new folder and never touches an older one:

```
data/<experiment>/<YYYY-MM-DD_HH-MM-SS>_<name>/
    README.md         summary, settings, figures
    config.json       settings, sample, camera read-back
    frames.csv        one row per frame          analysis.csv  per-frame results
    environment.csv   temperature/humidity/light events.log    timeline incl. manual steps
    raw/              full bit-depth TIFFs       figures/      PNGs
    report.ipynb      redraws everything from this folder alone
    biophoton.py      the code version the run was taken with
```

Frames and environment samples are written as they arrive, and every figure is drawn from the
files, so clearing a notebook or restarting the kernel loses nothing. `data/` is not in git.

## Checks

`python test_biophoton.py` checks the analysis on synthetic data: read-noise estimate and photon
counting on an EMCCD frame with amplifier glow, the temporal PTC, the dark-current fit, and a
full redraw of synthetic runs.
