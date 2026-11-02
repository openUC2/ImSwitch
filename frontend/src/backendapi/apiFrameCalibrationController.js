// src/backendapi/apiFrameCalibrationController.js
// API wrappers for the FrameCalibrationController (one-button FRAME calibration).

import createAxiosInstance from "./createAxiosInstance";

const BASE = "/FrameCalibrationController";

const get = async (path) => (await createAxiosInstance().get(`${BASE}/${path}`)).data;
const post = async (path, body) => (await createAxiosInstance().post(`${BASE}/${path}`, body)).data;

export const apiFrameCalibrationGetStatus = () => get("getFrameCalibrationStatus");
export const apiFrameCalibrationGetPlate = () => get("getFrameCalibrationPlate");
export const apiFrameCalibrationGetSettings = () => get("getFrameCalibrationSettings");
export const apiFrameCalibrationSetSettings = (settings) => post("setFrameCalibrationSettings", settings);
export const apiFrameCalibrationStart = () => post("startFrameCalibration");
export const apiFrameCalibrationStop = () => post("stopFrameCalibration");
export const apiFrameCalibrationContinue = () => post("continueFrameCalibration");
