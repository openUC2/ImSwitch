// src/backendapi/apiArkitektController.js
// API wrappers for the ArkitektController backend: binding this microscope
// to an Arkitekt server (device-code login), unbinding, settings, and the
// log of remote calls. Grouped in one module since the Arkitekt panel uses
// them together.

import createAxiosInstance from "./createAxiosInstance";

const get = async (path, params) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(`/ArkitektController/${path}`, { params });
  return response.data;
};

// POST with query parameters (scalar arguments), like the other side-effecting
// endpoints of this backend.
const post = async (path, params) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(`/ArkitektController/${path}`, null, { params });
  return response.data;
};

// {state, message, url, userCode, approveUrl, hasStoredLogin, actions, activity, ...}
export const apiArkitektGetStatus = () => get("getArkitektStatus");

// Starts the login in the background and returns at once; poll the status
// (or listen to sigArkitektStatus) for the code to approve.
export const apiArkitektBind = (url = "", redeemToken = "") =>
  post("bindArkitekt", { url, redeemToken });

// Stop a pending login or disconnect; the stored login is kept.
export const apiArkitektCancel = () => post("cancelArkitekt");

// Disconnect and forget the stored login on this microscope.
export const apiArkitektUnbind = () => post("unbindArkitekt");

// Only the given keys change: {url, appName, autoConnect, allowInsecureTransport, useMikro}
export const apiArkitektSetSettings = (settings) => post("setArkitektSettings", settings);

// [{id, name, datasetId, shape, dtype, positionUm, pixelSizeUm, time, thumbnail}]
export const apiArkitektGetUploads = () => get("getArkitektUploads");

export const apiArkitektClearActivity = () => post("clearArkitektActivity");
