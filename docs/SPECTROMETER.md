# Spectrometer

A grating spectrometer built from an area camera, such as the openUC2 openRAMAN block
(50 µm slit, AC254‑035‑A collimator, 1200 l/mm grating, CLENS‑100TEL lens on a HIK
MV‑CE060‑10UM). ImSwitch reads the camera, turns the slit image into a spectrum,
calibrates the wavelength axis with known sources and saves spectra with their metadata.
The React app "Spectrometer" shows it live.

| Part | File |
| --- | --- |
| numerics (numpy only, no ImSwitch imports, usable from a notebook) | `imswitch/imcontrol/model/spectrometer/` |
| instrument state: camera, dark, processing, calibration, simulator | `imswitch/imcontrol/model/managers/SpectrometerManager.py` |
| REST API, live loop, jobs, source switching, wizard | `imswitch/imcontrol/controller/controllers/SpectrometerController.py` |
| setup block | `SpectrometerInfo` in `imswitch/imcontrol/model/SetupInfo.py` |
| web app | `frontend/src/components/spectrometer/`, `frontend/src/backendapi/apiSpectrometerController.js` |
| virtual instrument | `imswitch/_data/user_defaults/imcontrol_setups/example_virtual_spectrometer.json` |
| tests | `imswitch/imcontrol/_test/unit/test_spectrometer_model.py`, `test_spectrometer_controller.py` |

It is a core controller, not a plugin: the plugin system does not yet pass setup
information to plugin managers (`MasterController`: `moduleInfo = None`), and the
spectrometer needs the setup, the detectors, the lasers, the LED matrix and the React
app. Because the numerics live in their own package, a plugin or a notebook can still
use them.

## Try it without hardware

```bash
python -m imswitch --no-ssl --http-port 8001 --config-file imswitch/_data/user_defaults/imcontrol_setups/example_virtual_spectrometer.json --config-folder <scratch>/cfg --data-folder <scratch>/data
```

Create both folders first; ImSwitch ignores the two flags for folders that do not exist
and then writes into `~/ImSwitchConfig`. In the virtual setup, `"simulate": true` makes
the manager draw frames itself. The simulated instrument has a 30 mm lens, so the lasers
at 488 and 635 nm, the Q‑Box line at 520 nm and the NeoPixel bands all fit on the
sensor. It also has a 16 px smile, stripes along the slit like the first prototype's
blade diffraction, shot and read noise, hot pixels and cosmic‑ray spikes. Its nominal
design (32 mm, 550 nm on the axis) is deliberately off by a few nm, so a calibration has
something to correct.

## Setup block

```json
"spectrometer": {
  "detector": "HIKCamera",
  "processing": {"rowStart": 300, "rowStop": 1700, "nAverage": 10},
  "design": {"cameraFocalLengthMm": 100, "collimatorFocalLengthMm": 35,
             "centerWavelengthNm": 592.4, "leverArmMm": 25},
  "laserWavelengthNm": 532,
  "ledMatrix": "ESP32 LEDMatrix",
  "ledIntensity": 64,
  "referenceSpectraFile": "neopixel_ring_reference.csv",
  "laserPowerFraction": 0.05,
  "alignmentTargetNm": 592.4
}
```

and `"Spectrometer"` in `availableWidgets`. Every field is optional. The design defaults
are those of the openRAMAN block: 1200 l/mm, a camera arm at 47.5° from the reversed
incoming beam, f = 100 mm, 2.4 µm pixels, 3072 columns, a 50 µm slit and shorter
wavelengths to the right. With these defaults the model reproduces the block's numbers:
grating turn −0.90° for 592.4 nm on the axis, a window of 561.7–623.1 nm, 0.0200 nm per
pixel, and a slit image 40.9 px wide (the first spectra measured 41 px).
`manualSources` lists sources switched by hand. By default these are the Q‑Box 520 nm
laser (±2 nm until it is measured), a neon glow lamp and a compact fluorescent lamp.
The settings and the calibration are stored in `<config>/spectrometer/<name>.json`.

## From frame to spectrum

All row and column numbers are full‑sensor indices, so the calibration stays valid when
the region changes. The rows run along the slit; the columns run along the dispersion.

1. **Region:** the rows lit by the slit (`autoSpectrometerRoi` finds them) and, if
   wanted, a column range.
2. **Dark:** the mean of N full frames taken with the light blocked and the sources off.
   It is subtracted when the exposure and gain still match (a warning appears otherwise).
3. **Averaging:** N fresh frames (a new frame counter, or two exposure times without one).
   Cosmic rays are removed with a per‑pixel median over the frames, in chunks of 8.
   With `frameCombine: "sum"` the frames are added instead. This is for integrations
   longer than the camera's longest exposure (1 s on the HIK). The metadata records
   `integrationMs`.
4. **Spikes in a single frame:** a median along the slit (5 rows). A spectral line is
   long in that direction, so it survives at any width, while a spike does not. The
   threshold follows the noise, which is measured from the frame itself (read noise plus
   shot noise). The pixel must also stand 30 % above its neighbours.
5. **Smile:** the curvature of a narrow line along the slit, as a polynomial in the row,
   measured from a laser or lamp line. Every row is shifted onto the reference row
   before the rows are combined. The bow on the first prototype was 16 px.
6. **Combining the rows** (mean or sum) gives counts per column.
7. **Axes:** nm from the calibration (or from the design, marked "not calibrated"), and
   the Raman shift in cm⁻¹ = 10⁷/λ_laser − 10⁷/λ.

**Checked on the first‑light frames** (openuc2‑openraman `data/2026-11-15`, white LED
with and without the FEL0500 filter, 8 bit), against that repository's own analysis:

| Quantity | This pipeline | Analysis |
| --- | --- | --- |
| filter edge at 50 % | column 1398.0 | 1398.0 |
| transmission plateau | 0.693 | 0.693 |
| LED peak | column 2472–2476 | 2474 |
| edge column per 256‑row band | 1384 → 1401 → 1393 | 1386 → 1402 → 1395 |
| slit rows found | all 2048 | the slit fills all rows |

The bands give the same bow; they are 1–2 px lower here because the steepest point of
the edge is taken, not the 50 % crossing. The slit on that prototype is lit very
unevenly: the top rows get 20–40 % of the brightest row. "Find slit" therefore uses
15 % of the peak above the dark level, measured on the brightest 5 % of the columns.

The camera should run in 12‑bit mode. The HIK driver already selects Mono12 on mono
cameras; the 8‑bit BMPs of the first light came from the vendor viewer. The page shows
the raw peak as a percentage of full scale; keep it below about 80 %.

## Calibration

**Models.**
- **design:** the nominal grating model, not measured.
- **grating:** a fit of the grating model. One line fixes the grating turn (an offset).
  Two lines far enough apart also fix the dispersion (the camera focal length). Three
  lines also fix the camera‑axis pixel. Weak priors (focal length ±5 %, axis ±10 % of the
  sensor) hold back what the lines cannot fix.
- **polynomial:** pixel → nm, order ≤ number of lines − 1, for many lines (a neon lamp).
- **auto:** the grating model below 5 lines, the polynomial from 5.

The residual of every line (nm) and the rms are shown and stored.

**Sources.**
- **Lasers in ImSwitch** (setup `lasers` with a `wavelength`). These give the sharpest
  points. Diode lasers can be a few nm off their nominal wavelength, so enter the
  measured value and its uncertainty in the wizard if you have one. The wizard runs them
  at `laserPowerFraction` (5 %) of their range and always switches them off again.
- **NeoPixel ring, R, G and B one at a time,** through the LED matrix. Each band's centre
  (the centroid above half maximum) is matched to the centre of a reference spectrum
  measured on another spectrometer. Load that file in the wizard (CSV header
  `wavelength_nm,r,g,b`) or set `referenceSpectraFile`. Without it, Gaussian placeholders
  from typical WS2812B data are used (628, 520 and 468 nm) and the wizard warns that
  these points can be a few nm off. A band that runs off the sensor is refused. The band
  width also gives a rough dispersion check (shown as nm/px).
- **Switched by hand:** the Q‑Box 520 nm laser, or a lamp with known lines (neon NE‑2:
  540.06 and 585.25–703.24 nm; fluorescent lamp: Hg 404.66, 435.83, 546.07, 576.96,
  579.07 nm). The wizard waits until you press Continue. Lamp lines are matched by a
  search over offset and dispersion scale (the design may be several % off), then
  matched again more tightly after each refit. In the simulation, all 18 neon lines on
  the sensor came out with an rms of 0.001 nm.

**Wizard flow:**
1. All sources off, then a dark frame.
2. For each source: switch it on (or wait for you), average N frames, find the line,
   switch it off.
3. The first narrow line also measures the smile.
4. Fit, then review the table of residuals and line widths.
5. Apply and save, or discard.

On cancel and on errors every source is switched off.

**Resolution:** `checkSpectrometerResolution` gives the FWHM of the strongest line in
px, nm and cm⁻¹, against the geometric slit image of the design, which includes the
anamorphic factor cos α / cos β. A ratio near 1 means the camera lens is focused.

## Grating alignment with a known line

`getSpectrometerAlignment(lineNm, targetCenterNm)` compares where a known line sits now
with where it must sit for the target window, using the grating model. The page draws
both positions as markers on the live spectrum.

If the line is not on the sensor in the target window, it is first parked on the camera
axis, which is a turn you can see, and the grating is then turned by the computed angle.
With the 520 nm Q‑Box laser and the openRAMAN design:
- park the line on column 1536 (grating turn −3.82°);
- then turn by +2.92°, which is 1.28 mm at the 25 mm lever;
- the target window is then 561.7–623.1 nm, with 592.4 nm on the axis.

Once the instrument is calibrated, only a peak within max(2 nm, 3 FWHM) of the line
counts. Otherwise the page says the line is not in the spectrum. Before calibration it
says it is taking the strongest line as the known one.

## REST API

All under `/imswitch/api/SpectrometerController/`. Units: nm, cm⁻¹, ms; pixels are
full‑sensor columns.

| Endpoint | Does |
| --- | --- |
| `getSpectrometerState` | camera, processing, calibration summary, dark, design, live/job/wizard state, warnings |
| `startSpectrometerLive(periodMs)` / `stopSpectrometerLive` | process the latest frame every period |
| `getSpectrometerSpectrum(which, maxPoints, withPeaks)` | `px`, `nm`, `ramanShiftCm1`, `counts`, metadata, peaks; which = auto / live / acquired |
| `getSpectrometerImage(full, corrected, maxWidth)` | PNG data URL of the region, or of a full frame with the region rows |
| `getSpectrometerRoiArray(which, corrected)` | the region as a NumPy `.npy` file (float32, raw or corrected; rows and columns in the headers `X-Rows` / `X-Cols`) |
| `setSpectrometerCamera(exposureMs, gain, bitDepth)` | camera settings (negative or 0 keeps the value) |
| `setSpectrometerSettings` (POST) | region, collapse, frames, spikes, dark, smile, live average |
| `autoSpectrometerRoi(frac, pad)` | find the slit rows |
| `acquireSpectrometerSpectrum(nAverage)` / `takeSpectrometerDark(nAverage)` | background jobs; poll `getSpectrometerAcquisition`, stop with `stopSpectrometerAcquisition` |
| `saveSpectrometerSpectrum(name, fileFormat, which, includeRoi, notes)` | `<data>/Spectrometer/<time>_<name>.h5` or `.csv` + `.json` |
| `getSpectrometerCalibration` / `setSpectrometerCalibration` (POST) / `resetSpectrometerCalibration` | calibration from known lines `[{px, nm, sigmaNm}]` |
| `measureSpectrometerSmile` / `clearSpectrometerSmile` | with one narrow line on |
| `setSpectrometerLaserWavelength(laserNm)` | excitation for the cm⁻¹ axis |
| `getSpectrometerSources` / `setSpectrometerSource(sourceId, active, power)` / `allSpectrometerSourcesOff` | calibration sources |
| `setSpectrometerReferenceSpectrum` (POST) | reference spectrum of an LED channel |
| `startSpectrometerWizard` (POST), `getSpectrometerWizard`, `continueSpectrometerWizard`, `skipSpectrometerWizardStep`, `cancelSpectrometerWizard`, `applySpectrometerWizard(save)` | wavelength calibration |
| `checkSpectrometerResolution`, `getSpectrometerAlignment(lineNm, targetCenterNm)` | resolution, grating alignment |

**HDF5 layout:**
- `spectrum/px`, `spectrum/nm`, `spectrum/ramanShiftCm1`, `spectrum/counts`,
  `spectrum/dark` (the collapsed dark);
- `roi`: the mean raw frame region, gzip;
- the `metadata` attribute (JSON): exposure, gain, bit depth, frames, rows, dark,
  smile, spikes, saturation, the full calibration, the laser wavelength, the sources
  that were on and notes.

## Next steps: Raman, OCT, multispectral imaging

The spectrum record (axes plus counts plus metadata) is the same for all three.

- **Raman.** The excitation laser goes in `laserWavelengthNm`, and the cm⁻¹ axis follows.
  - Measure the laser line itself, at low power with the edge filter out, to set 0 cm⁻¹.
  - Then check the calibration with a Raman standard; the values are in
    `references.RAMAN_STANDARDS_CM1`: cyclohexane and polystyrene from ASTM E1840, and
    acetonitrile.
  - A wizard source of kind "raman standard" would convert those lines to nm through the
    laser wavelength and fit the same way. This is AutoOpenRaman's fine step after the
    neon step.
  - Processing (despiking, baseline, smoothing, normalisation) could later call RamanSPy
    as an optional dependency.
- **OCT.** The same block with a fibre end as the slit and the grating at the OCT stop.
  - `calibration.resample_linear_k` resamples a spectrum onto an evenly spaced k axis;
    an FFT per line then gives the A‑scan.
  - The camera then reads a few rows at kHz rates. The region and `collapse` already
    describe that; a line‑rate acquisition would replace `acquireRoi` with the camera's
    chunk readout (`getChunk`).
- **Multispectral imaging with galvos.**
  - At each scan point of `GalvoScannerController` (or a stage scan), take one spectrum
    with `acquireRoi(1)` + `processAcquired`.
  - Write the cube (y, x, λ) with the common OME‑Zarr writer
    (`model/aufgaben/storage.NgffImage`), with the λ axis from the calibration in the
    metadata.
  - The live loop is not needed there; the camera's frame counter keeps scan points and
    frames paired.

## AutoOpenRaman

[Arcadia Science's AutoOpenRaman](https://github.com/Arcadia-Science/2024-auto-openraman)
(checked 17 Nov 2026) is under the **MIT licence**, which is compatible with ImSwitch's
GPL‑3.0 if its copyright notice is kept with any copied code. It contains:
- a PyQt5 + pyqtgraph GUI on Micro‑Manager through pycromanager;
- a two‑step calibration (`RamanCalibrator`): a linear neon fit from the most prominent
  peaks, then a fine fit on acetonitrile Raman bands;
- processing with [RamanSPy](https://github.com/barahona-research-group/RamanSPy)
  (BSD‑3‑Clause).

Recommendation:
- **Don't copy the GUI or the acquisition code.** The GUI is Qt; this fork is headless
  with a React app, and ImSwitch drives the camera itself.
- **Take over the calibration idea, not the code.** The neon step is covered here, and
  more robustly: lines are matched by offset and scale instead of in peak order. The
  acetonitrile fine step is worth adding as a "Raman standard" source.
- **Use RamanSPy for processing** as an optional dependency. Adding a dependency needs
  the maintainers' consent (agents.md).

Nothing from AutoOpenRaman was copied into this module.

## Limits and open points

- **Bit depth.** A switch exists only if the camera driver has a bit-depth parameter
  (`bitDepthParameter`). The HIK driver chooses Mono12 itself.
- **LED points depend on their reference spectra.** With the placeholders they are only
  good to a few nm, and the instrument response tilts broad bands. Lasers and lamps are
  better.
- **Wavelength only.** There is no intensity (instrument response) calibration yet. The
  plan is a halogen lamp against a Planck curve.
- **Live view during jobs.** The live loop and the jobs read the same camera. A long
  exposure (up to 1 s on the HIK) slows the live view.
