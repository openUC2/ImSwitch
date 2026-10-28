// src/backendapi/apiFlowStopController.js
// FlowStop (flow-cell / PlanktoScope-style imaging) API wrappers.

import createAxiosInstance from "./createAxiosInstance";

const BASE = "/FlowStopController";

const get = async (path, params) => {
  const instance = createAxiosInstance();
  const response = await instance.get(`${BASE}${path}`, { params });
  return response.data;
};

const post = async (path, body) => {
  const instance = createAxiosInstance();
  const response = await instance.post(`${BASE}${path}`, body);
  return response.data;
};

/** Devices + ranges the UI renders its controls from. */
export const apiFlowStopGetHardware = () => get("/getFlowStopHardware");

/** {isRunning, imagesTaken, numImages, progress, elapsedSeconds, etaSeconds, relativePath, ...} */
export const apiFlowStopGetStatus = () => get("/getFlowStopStatus");

/** Stored acquisition parameters. */
export const apiFlowStopGetParameters = () => get("/getExperimentParameters");

/** Persist acquisition parameters; returns the stored set. */
export const apiFlowStopSetParameters = (params) => post("/setFlowStopParameters", params);

/** EcoTaxa-style sample metadata. */
export const apiFlowStopGetMetadata = () => get("/getFlowStopMetadata");

/** Persist sample metadata; returns the stored set. */
export const apiFlowStopSetMetadata = (metadata) => post("/setFlowStopMetadata", metadata);

/** Start an acquisition, optionally overriding the stored parameters. */
export const apiFlowStopStart = (params) => post("/startFlowStopExperiment", params || {});

export const apiFlowStopStop = () => get("/stopFlowStopExperiment");

/** Jog the pump axis by `value` motor steps (relative). Negative reverses. */
export const apiFlowStopMovePump = (value, speed) => get("/movePump", { value, speed });

export const apiFlowStopStopPump = () => get("/stopPump");

/** Jog the focus axis by `value` motor steps (relative). */
export const apiFlowStopMoveFocus = (value, speed) => get("/moveFocus", { value, speed });

export const apiFlowStopStopFocus = () => get("/stopFocus");

export const apiFlowStopSetIllumination = (value, enabled = true) =>
  get("/setIlluIntensity", { value, enabled });

/** 'auto' or 'manual'. */
export const apiFlowStopSetAutoExposure = (value) => get("/changeAutoExposureTime", { value });

export const apiFlowStopSetExposureTime = (value) => get("/changeExposureTime", { value });

/** List the files of an acquisition folder via the shared FileManager. */
export const apiFlowStopListFiles = async (relativePath) => {
  const instance = createAxiosInstance();
  const response = await instance.get("/FileManager/", { params: { path: relativePath } });
  return response.data;
};
