// src/backendapi/apiSpectrometerController.js
// API wrappers for the SpectrometerController (grating spectrometer on a camera).
// Grouped in one module like apiOPMController: the Spectrometer app uses them together.
// Units: nm, cm^-1, ms; pixels are full-sensor columns/rows.

import createAxiosInstance from "./createAxiosInstance";

const get = async (path, params, timeout) => {
  const response = await createAxiosInstance().get(`/SpectrometerController/${path}`, {
    params,
    timeout,
  });
  return response.data;
};

const post = async (path, body) => {
  const response = await createAxiosInstance().post(`/SpectrometerController/${path}`, body);
  return response.data;
};

// camera, processing, calibration summary, dark, design, live/job/wizard state, warnings
export const apiSpectrometerGetState = () => get("getSpectrometerState");

export const apiSpectrometerStartLive = (periodMs = 250) =>
  get("startSpectrometerLive", { periodMs });
export const apiSpectrometerStopLive = () => get("stopSpectrometerLive");

// which: auto | live | acquired. Arrays px, nm, ramanShiftCm1, counts (decimated, maxima kept)
export const apiSpectrometerGetSpectrum = (which = "auto", maxPoints = 1536) =>
  get("getSpectrometerSpectrum", { which, maxPoints, withPeaks: true });

// PNG data URL of the region (raw or corrected) or of a full frame with the region rows
export const apiSpectrometerGetImage = (full = false, corrected = false, maxWidth = 1536) =>
  get("getSpectrometerImage", { full, corrected, maxWidth });

// region of a spectrum as a NumPy .npy blob (float32, rows x columns); corrected: dark,
// spikes, smile applied. A JSON error comes back as a blob of type application/json.
export const apiSpectrometerGetRoiArray = async (which = "auto", corrected = false) => {
  const response = await createAxiosInstance().get(
    "/SpectrometerController/getSpectrometerRoiArray",
    { params: { which, corrected }, responseType: "blob" },
  );
  return response.data;
};

// negative / 0 keeps the current value
export const apiSpectrometerSetCamera = ({ exposureMs = -1, gain = -1, bitDepth = 0 }) =>
  get("setSpectrometerCamera", { exposureMs, gain, bitDepth });

// {rowStart, rowStop, colStart, colStop, collapse, nAverage, frameCombine (mean|sum),
//  removeSpikes, spikeThreshold, subtractDark, smileCorrection, liveAverage};
// only the given fields change
export const apiSpectrometerSetSettings = (settings) =>
  post("setSpectrometerSettings", settings);

export const apiSpectrometerAutoRoi = (frac = 0.15, pad = 0.02) =>
  get("autoSpectrometerRoi", { frac, pad });

export const apiSpectrometerSetLaserWavelength = (laserNm) =>
  get("setSpectrometerLaserWavelength", { laserNm });

// background jobs: poll apiSpectrometerGetAcquisition
export const apiSpectrometerAcquire = (nAverage = 0) =>
  get("acquireSpectrometerSpectrum", { nAverage });
export const apiSpectrometerTakeDark = (nAverage = 10) =>
  get("takeSpectrometerDark", { nAverage });
export const apiSpectrometerGetAcquisition = () => get("getSpectrometerAcquisition");
export const apiSpectrometerStopAcquisition = () => get("stopSpectrometerAcquisition");
export const apiSpectrometerClearDark = () => get("clearSpectrometerDark");

// fileFormat: hdf5 | csv; which: auto | live | acquired
export const apiSpectrometerSave = ({
  name = "spectrum",
  fileFormat = "hdf5",
  which = "auto",
  includeRoi = true,
  notes = "",
}) => get("saveSpectrometerSpectrum", { name, fileFormat, which, includeRoi, notes });

export const apiSpectrometerGetCalibration = () => get("getSpectrometerCalibration");
// {points: [{px, nm, sigmaNm}], model: auto|grating|polynomial, order, save, notes}
export const apiSpectrometerSetCalibration = (body) => post("setSpectrometerCalibration", body);
export const apiSpectrometerResetCalibration = () => get("resetSpectrometerCalibration");
export const apiSpectrometerMeasureSmile = (nAverage = 5) =>
  get("measureSpectrometerSmile", { nAverage }, 120000);
export const apiSpectrometerClearSmile = () => get("clearSpectrometerSmile");
export const apiSpectrometerCheckResolution = (which = "auto") =>
  get("checkSpectrometerResolution", { which });
export const apiSpectrometerGetAlignment = (lineNm = 520, targetCenterNm = 0, which = "auto") =>
  get("getSpectrometerAlignment", { lineNm, targetCenterNm, which });

export const apiSpectrometerGetSources = () => get("getSpectrometerSources");
// power: fraction of the laser range / of 255 for an LED channel (-1 keeps)
export const apiSpectrometerSetSource = (sourceId, active, power = -1) =>
  get("setSpectrometerSource", { sourceId, active, power });
export const apiSpectrometerAllSourcesOff = () => get("allSpectrometerSourcesOff");
// {key: "r"|"g"|"b", wavelengthNm: [...], intensity: [...], label}
export const apiSpectrometerSetReference = (body) =>
  post("setSpectrometerReferenceSpectrum", body);

// {sources: [ids], model, order, nAverage, nDark, takeDark, measureSmile, settleS,
//  power: {id: f}, wavelengths: {id: nm}, sigmas: {id: nm}}
export const apiSpectrometerStartWizard = (body) => post("startSpectrometerWizard", body);
export const apiSpectrometerGetWizard = () => get("getSpectrometerWizard");
export const apiSpectrometerContinueWizard = () => get("continueSpectrometerWizard");
export const apiSpectrometerSkipWizardStep = () => get("skipSpectrometerWizardStep");
export const apiSpectrometerCancelWizard = () => get("cancelSpectrometerWizard");
export const apiSpectrometerApplyWizard = (save = true) =>
  get("applySpectrometerWizard", { save });
