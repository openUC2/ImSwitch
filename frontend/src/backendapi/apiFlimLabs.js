/**
 * apiFlimLabs.js - Client for the FLIM LABS card, via the ImSwitch backend.
 *
 * The ImSwitch backend (FLIMLabsController + FLIMLabsDetectorManager) is the
 * SINGLE owner of the flim-imager server connection: its `/data` WebSocket is a
 * single-consumer queue drain upstream, so two clients would split the stream
 * between them. The browser therefore never talks to the FLIM server directly
 * any more - it goes through ImSwitch, which means the FLIM panel keeps working
 * while an ExperimentController acquisition uses the same card as a detector.
 *
 * (Before, this module spoke to the FLIM server's REST/WS directly. The payload
 * construction, binary-protocol parsing and frame assembly now live in
 * imswitch/imcontrol/model/interfaces/flimlabsclient.py.)
 */

const getApiBase = (hostIP, hostPort) =>
  `${hostIP}:${hostPort}/imswitch/api/FLIMLabsController`;

const jsonFetch = async (url, options = {}) => {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const msg = body && (body.error || body.detail || body.message);
    throw new Error(msg || `ImSwitch error ${response.status}`);
  }
  if (body && body.error) throw new Error(body.error);
  return body;
};

/**
 * Full FLIM state: server health, card serial, acquisition status, CPS,
 * frame number, calibration table and the calibration reference path.
 */
export const apiFlimGetStatus = (hostIP, hostPort) =>
  jsonFetch(`${getApiBase(hostIP, hostPort)}/getFlimStatus`);

/**
 * Latest intensity frame as a base64 PNG data URL. Returns the progressively
 * filling frame while one is being received (a FLIM frame takes seconds).
 */
export const apiFlimGetImage = (hostIP, hostPort, maxSize = 512) =>
  jsonFetch(`${getApiBase(hostIP, hostPort)}/getFlimImage?maxSize=${maxSize}`);

/** Accumulated phasor density as sparse [x, y, count] triplets. */
export const apiFlimGetPhasor = (hostIP, hostPort, maxPoints = 20000) =>
  jsonFetch(`${getApiBase(hostIP, hostPort)}/getFlimPhasor?maxPoints=${maxPoints}`);

/**
 * Start a run.
 * @param {'scouting'|'calibration'|'phasors'} step
 */
export const apiFlimStart = (hostIP, hostPort, {
  step = 'scouting',
  maxFrames = 0,
  tauNs = null,
  harmonics = 1,
  exportData = false,
  exportFilename = '',
} = {}) => {
  const params = new URLSearchParams({
    step,
    maxFrames: String(maxFrames),
    harmonics: String(harmonics),
    exportData: String(exportData),
    exportFilename,
  });
  if (tauNs !== null && tauNs !== undefined) params.append('tauNs', String(tauNs));
  return jsonFetch(`${getApiBase(hostIP, hostPort)}/startFlimAcquisition?${params}`);
};

/** Stop the running acquisition (and the auto-driven galvo scan). */
export const apiFlimStop = (hostIP, hostPort) =>
  jsonFetch(`${getApiBase(hostIP, hostPort)}/stopFlimAcquisition`);

/** Clear accumulated image, phasor histogram and calibration table. */
export const apiFlimReset = (hostIP, hostPort) =>
  jsonFetch(`${getApiBase(hostIP, hostPort)}/resetFlimBuffers`);

/** One-shot laser frequency measurement (sync signal must be connected). */
export const apiFlimDetectLaserFrequency = (hostIP, hostPort) =>
  jsonFetch(`${getApiBase(hostIP, hostPort)}/detectFlimLaserFrequency`);

/** Set a FLIM detector parameter (dwell_time, frequency_mhz, reconstruction...). */
export const apiFlimSetParameter = (hostIP, hostPort, name, value) =>
  jsonFetch(
    `${getApiBase(hostIP, hostPort)}/setFlimParameter?name=${encodeURIComponent(name)}` +
    `&value=${encodeURIComponent(value)}`
  );

const api = {
  apiFlimGetStatus,
  apiFlimGetImage,
  apiFlimGetPhasor,
  apiFlimStart,
  apiFlimStop,
  apiFlimReset,
  apiFlimDetectLaserFrequency,
  apiFlimSetParameter,
};

export default api;
