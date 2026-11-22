// src/backendapi/apiProcessingController.js
// API wrappers for the ProcessingController (deconvolution of stored stacks).
// One function per endpoint; used by the deconvolution panel of the OME-Zarr viewer.

import createAxiosInstance from "./createAxiosInstance";

const BASE = "/ProcessingController";

const get = async (path, params) => {
  const response = await createAxiosInstance().get(`${BASE}/${path}`, { params });
  return response.data;
};

const post = async (path, body) => {
  const response = await createAxiosInstance().post(`${BASE}/${path}`, body);
  return response.data;
};

/** Recipe defaults and limits, optics per channel (and where they came from), image shape. */
export const apiProcessingGetDefaults = (path, image = "") => get("getProcessingDefaults", { path, image });
/** Deconvolve a crop on the Pi: {crop, z, row, background, seconds, before: {xy, xz}, after: {xy, xz}}. */
export const apiProcessingPreview = (request) => post("previewProcessing", request);
/** Whole image in the background, on the worker or the Pi; poll apiProcessingGetStatus. */
export const apiProcessingStartJob = (request) => post("startProcessingJob", request);
/** The PSF the optics give (xy at focus, xz through the centre) with its FWHM. */
export const apiProcessingPreviewPSF = (request) => post("previewProcessingPSF", request);
/** Average the beads of a bead stack into a measured PSF, saved next to it as <stack>_psf.ome.zarr. */
export const apiProcessingDistillPSF = (request) => post("distillProcessingPSF", request);
export const apiProcessingGetStatus = () => get("getProcessingStatus");
export const apiProcessingCancelJob = () => get("cancelProcessingJob");
export const apiProcessingGetWorker = () => get("getProcessingWorker");
export const apiProcessingSetWorker = (url) => get("setProcessingWorker", { url });
