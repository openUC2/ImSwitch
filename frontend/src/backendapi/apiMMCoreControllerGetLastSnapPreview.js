// src/backendapi/apiMMCoreControllerGetLastSnapPreview.js
// Build a URL for the latest captured snap frame as PNG. The frontend uses
// this as the src of an <img> tag — fetching is left to the browser so it
// can cache and render progressively. Pass a fresh cacheBust value (e.g. the
// job's finishedAt timestamp) to force the browser to re-fetch when a new
// snap is available for the same id.
const apiMMCoreControllerGetLastSnapPreviewUrl = ({
  hostIP,
  apiPort,
  jobId,
  cacheBust,
}) => {
  if (!hostIP || !apiPort || !jobId) return null;
  const bust = cacheBust ?? Date.now();
  return `${hostIP}:${apiPort}/imswitch/api/MMCoreController/getLastSnapPreview?jobId=${encodeURIComponent(jobId)}&_t=${bust}`;
};

export default apiMMCoreControllerGetLastSnapPreviewUrl;
