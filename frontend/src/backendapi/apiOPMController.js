// src/backendapi/apiOPMController.js
// API wrappers for the OPMController (dvOPM digital twin and stage-scan acquisition).
// Grouped in one module like apiStageMapController: the OPM app uses them together.

import createAxiosInstance from "./createAxiosInstance";

const get = async (path, params) => {
  const response = await createAxiosInstance().get(`/OPMController/${path}`, { params });
  return response.data;
};

const post = async (path, body, config) => {
  const response = await createAxiosInstance().post(`/OPMController/${path}`, body, config);
  return response.data;
};

export const apiOPMGetState = () => get("getOPMState");

// optics: OPMOptics fields ({} = defaults, live twin geometry)
export const apiOPMGetRigLayout = (optics = {}) => post("getOPMRigLayout", optics);

export const apiOPMTraceOptics = (optics = {}) => post("traceOPMOptics", optics);

export const apiOPMExportOptiland = (optics = {}) =>
  post("exportOPMOptiland", optics, { responseType: "blob" });

// Twin's own travel check for moveOPMAxes on/off (the stage's soft limits stay)
export const apiOPMSetLimits = (enabled) => get("setOPMLimits", { enabled });

export const apiOPMSetMedium = (n, move = true) => get("setOPMMedium", { n, move });

// move: {sample: {X, Y, Z}, launch: {x, y, z, phi_deg, focus}, relative, track} in mm / degrees
export const apiOPMMoveAxes = (move) => post("moveOPMAxes", move);

// "We are here": {sample: {X, Y, Z}, launch: {...}}; missing keys take the defaults
export const apiOPMSetReference = (reference = {}) => post("setOPMReference", reference);

export const apiOPMGetRigConfig = () => get("getOPMRigConfig");

export const apiOPMSetRigConfig = (values) => post("setOPMRigConfig", { values });

export const apiOPMGetScanPlan = (length_mm, fps, step_um) =>
  get("getOPMScanPlan", { length_mm, fps, ...(step_um ? { step_um } : {}) });

export const apiOPMStartScan = (request) => post("startOPMScan", request);

export const apiOPMStopScan = () => get("stopOPMScan");

export const apiOPMGetScanStatus = () => get("getOPMScanStatus");
