// src/backendapi/apiRecordingControllerSnapJob.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Background snap jobs.
 *
 * A snap costs at least one full exposure, so the old blocking
 * `snapImageToPath` call kept the HTTP request open for the whole integration —
 * seconds to minutes for long-exposure work, during which the UI could not even
 * show progress. These three calls replace it:
 *
 *   startSnap()  -> { jobId, expectedDurationMs }   returns immediately
 *   getSnapStatus(jobId) -> { status, progress, result }
 *   cancelSnap(jobId)    -> aborts the exposure
 *
 * `expectedDurationMs` is the backend's exposure-based estimate; the UI counts
 * down against it locally so the countdown stays smooth regardless of polling.
 */

/**
 * Start a snap in the background.
 *
 * @param {object} options
 * @param {string} options.fileName - Optional description appended to the filename
 * @param {number} options.saveFormat - SaveFormat enum (1=TIFF, 5=PNG, 6=JPG)
 * @param {boolean} options.returnPreview - Include a PNG preview in the result
 * @param {number} options.previewMaxSize - Longest edge of that preview
 * @returns {Promise<{jobId: string, status: string, expectedDurationMs: number}>}
 */
export const apiRecordingControllerStartSnap = async ({
  fileName = "",
  saveFormat = 1,
  returnPreview = true,
  previewMaxSize = 1024,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(`/RecordingController/startSnap`, {
    params: { fileName, saveFormat, returnPreview, previewMaxSize },
  });
  return response.data;
};

/**
 * Poll a background snap.
 *
 * @param {string|null} jobId - Job to poll (null = most recent job)
 * @returns {Promise<{status: string, progress: number, elapsedMs: number, result?: object}>}
 *   status is one of pending | running | done | error | cancelled | unknown
 */
export const apiRecordingControllerGetSnapStatus = async (jobId = null) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (jobId) params.jobId = jobId;
  const response = await axiosInstance.get(`/RecordingController/getSnapStatus`, {
    params,
  });
  return response.data;
};

/**
 * Abort a running snap. Stops the exposure on the detectors the snap armed;
 * a detector that was already streaming keeps running.
 *
 * @param {string|null} jobId - Job to cancel (null = most recent job)
 */
export const apiRecordingControllerCancelSnap = async (jobId = null) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (jobId) params.jobId = jobId;
  const response = await axiosInstance.get(`/RecordingController/cancelSnap`, {
    params,
  });
  return response.data;
};

export default apiRecordingControllerStartSnap;
