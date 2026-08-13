// src/backendapi/apiStageMapController.js
// API wrappers for the StageMapController backend (MicroMagellan-style
// live stage mapping). Grouped in one module since the endpoints are
// small and always used together by the StageMap app.

import createAxiosInstance from "./createAxiosInstance";

const get = async (path, params) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(`/StageMapController/${path}`, {
    params,
  });
  return response.data;
};

export const apiStageMapGetStatus = () => get("getStageMapStatus");

export const apiStageMapGetParams = () => get("getStageMapParams");

export const apiStageMapSetParams = async (params) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/StageMapController/setStageMapParams",
    params,
  );
  return response.data;
};

export const apiStageMapStart = () => get("startStageMap");

export const apiStageMapStop = () => get("stopStageMap");

export const apiStageMapSnapTile = () => get("snapStageMapTile");

export const apiStageMapClear = () => get("clearStageMap");

export const apiStageMapSetChannel = (channel) =>
  get("setStageMapChannel", { channel });

export const apiStageMapGetTiles = (fromId = 0, includePreviews = true) =>
  get("getStageMapTiles", { fromId, includePreviews });

export const apiStageMapGotoPosition = (x, y, isAbsolute = true, isBlocking = false) =>
  get("gotoStagePosition", { x, y, isAbsolute, isBlocking });

export const apiStageMapSaveOmeTiff = (filename = "") =>
  get("saveStitchedOmeTiff", { filename });
