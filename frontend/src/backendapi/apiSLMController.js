// src/backendapi/apiSLMController.js
// API wrappers for the SLMController (SLM / DMD display conjugate to the sample).
// One function per endpoint; the SLM app is their only user.

import createAxiosInstance from "./createAxiosInstance";

const BASE = "/SLMController";

const get = async (path, params) => {
  const response = await createAxiosInstance().get(`${BASE}/${path}`, { params });
  return response.data;
};

const post = async (path, body) => {
  const response = await createAxiosInstance().post(`${BASE}/${path}`, body);
  return response.data;
};

// Display and status
export const apiSLMGetStatus = () => get("getSLMStatus");
export const apiSLMGetDisplays = () => get("getSLMDisplays");
export const apiSLMOpenDisplay = (display = "", displayIndex = -1) =>
  get("openSLMDisplay", { display, displayIndex });
export const apiSLMCloseDisplay = () => get("closeSLMDisplay");
export const apiSLMSetSettleTime = (settleTimeMs) => get("setSLMSettleTime", { settleTimeMs });

// Patterns, images, camera masks
export const apiSLMShowPattern = (kind, params = {}) => get("showSLMPattern", { kind, ...params });
export const apiSLMShowImage = (dataB64, fit = "center") =>
  post("showSLMImage", { data_b64: dataB64, fit });
export const apiSLMShowCameraMask = (maskB64, cameraWidth, cameraHeight, value = 255, invert = false) =>
  post("showSLMCameraMask", {
    mask_b64: maskB64,
    camera_width: cameraWidth,
    camera_height: cameraHeight,
    value,
    invert,
  });
// The preview is an <img src>; a nonce defeats caching.
export const slmPreviewUrl = (maxSize = 512) =>
  `${createAxiosInstance().defaults.baseURL}${BASE}/getSLMPreview?maxSize=${maxSize}&t=${Date.now()}`;

// Camera <-> display affine
export const apiSLMGetAffine = () => get("getSLMAffine");
export const apiSLMSetAffine = (coeffs) => get("setSLMAffine", coeffs);
export const apiSLMResetAffine = () => get("resetSLMAffine");
export const apiSLMComputeAffine = (cameraPoints, displayPoints) =>
  post("computeSLMAffine", { camera_points: cameraPoints, display_points: displayPoints });
export const apiSLMGetCalibrationPoints = (nPoints = 5) => get("getSLMCalibrationPoints", { nPoints });
export const apiSLMRunAutoCalibration = (nPoints = 5, spotRadius = 15, detectorName = "") =>
  get("runSLMAutoCalibration", { nPoints, spotRadius, detectorName });
export const apiSLMGetCalibrationStatus = () => get("getSLMCalibrationStatus");
export const apiSLMStopCalibration = () => get("stopSLMCalibration");

// Sequences
export const apiSLMUploadSequence = (name, frames) => post("uploadSLMSequence", { name, frames });
export const apiSLMListSequences = () => get("listSLMSequences");
export const apiSLMDeleteSequence = (name) => get("deleteSLMSequence", { name });
export const slmSequenceFrameUrl = (name, index, maxSize = 160) =>
  `${createAxiosInstance().defaults.baseURL}${BASE}/getSLMSequenceFrame?name=${encodeURIComponent(name)}&index=${index}&maxSize=${maxSize}`;
export const apiSLMShowSequenceFrame = (name, index) => get("showSLMSequenceFrame", { name, index });
export const apiSLMPlaySequence = (name, intervalMs = 500, loop = false) =>
  get("playSLMSequence", { name, intervalMs, loop });
export const apiSLMStopSequence = () => get("stopSLMSequence");

// Synchronised acquisition
export const apiSLMStartAcquisition = (body) => post("startSLMSequenceAcquisition", body);
export const apiSLMGetAcquisitionStatus = () => get("getSLMAcquisitionStatus");
export const apiSLMStopAcquisition = () => get("stopSLMSequenceAcquisition");
export const slmAcquiredFrameUrl = (index = -1, maxSize = 512) =>
  `${createAxiosInstance().defaults.baseURL}${BASE}/getSLMAcquiredFrame?index=${index}&maxSize=${maxSize}&t=${Date.now()}`;
