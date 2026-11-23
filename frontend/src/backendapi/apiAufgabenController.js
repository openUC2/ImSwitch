// src/backendapi/apiAufgabenController.js
// API wrappers for the AufgabenController (composable acquisition jobs).
// One function per endpoint; used by the Aufgaben app and the Live View macros.

import createAxiosInstance from "./createAxiosInstance";

const BASE = "/AufgabenController";

const get = async (path, params) => {
  const response = await createAxiosInstance().get(`${BASE}/${path}`, { params });
  return response.data;
};

const post = async (path, body) => {
  const response = await createAxiosInstance().post(`${BASE}/${path}`, body);
  return response.data;
};

// Catalogue and documents
export const apiAufgabenGetCatalog = () => get("getAufgabenCatalog");
export const apiAufgabenGetJobSchema = () => get("getAufgabenJobSchema");
export const apiAufgabenListJobs = () => get("listAufgabenJobs");
export const apiAufgabenGetJob = (name) => get("getAufgabenJob", { name });
export const apiAufgabenSaveJob = (job) => post("saveAufgabenJob", job);
export const apiAufgabenDeleteJob = (name) => get("deleteAufgabenJob", { name });
export const apiAufgabenValidateJob = (job) => post("validateAufgabenJob", job);

// useq bridge and labware
export const apiAufgabenImportUseq = (sequence) => post("importAufgabenUseq", sequence);
export const apiAufgabenExportUseq = (job) => post("exportAufgabenUseq", job);
export const apiAufgabenGetWellPlate = (loadName, wells = "", offsetXUm = 0, offsetYUm = 0) =>
  get("getAufgabenWellPlate", {
    load_name: loadName,
    wells,
    offset_x_um: offsetXUm,
    offset_y_um: offsetYUm,
  });

// Runs
export const apiAufgabenStartJob = (job, parameters = {}, confirmed = false) =>
  post("startAufgabenJob", { job, parameters, confirmed });
export const apiAufgabenPauseJob = () => get("pauseAufgabenJob");
export const apiAufgabenResumeJob = () => get("resumeAufgabenJob");
export const apiAufgabenFinishJob = () => get("finishAufgabenJob");
export const apiAufgabenAbortJob = (force = false) => get("abortAufgabenJob", { force });
export const apiAufgabenGetStatus = () => get("getAufgabenStatus");
export const apiAufgabenGetTimeline = (since = 0) => get("getAufgabenTimeline", { since });
export const apiAufgabenAnswerQuestion = (answer) => get("answerAufgabenQuestion", { answer });
export const apiAufgabenListRuns = (limit = 50) => get("listAufgabenRuns", { limit });

// Macros: saved jobs with metadata.macro, run from Live View buttons
export const apiAufgabenListMacros = () => get("listAufgabenMacros");
export const apiAufgabenListMacroTemplates = () => get("listAufgabenMacroTemplates");
export const apiAufgabenRunMacro = (name, parameters = {}, confirmed = false) =>
  post("runAufgabenMacro", { name, parameters, confirmed });
