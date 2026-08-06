/**
 * apiFlimLabs.js - Direct client for the FLIM LABS flim-imager standalone server.
 *
 * Unlike the other backendapi modules this does NOT talk to the ImSwitch
 * backend: the FLIM server (Rust warp, default port 5249) runs in its own
 * Docker container (possibly on a different host) and allows any CORS origin,
 * so the browser can control it directly. Used by the dev-mode FLIM bridge
 * panel inside the galvo scanner UI.
 *
 * Targets flim-imager 2.x (the "flim-imager-2.0" repository). The /start
 * payload mirrors what the v2 Svelte frontend sends (see
 * svelte/src/lib/network/web.service.ts, buildImagingExperimentPayload) -
 * the v1 payload schema is NOT accepted by the v2 server.
 */

export const FLIM_DEFAULT_PORT = 5249;

/** Normalize "host[:port]" + port into an http base URL. */
export const flimBaseUrl = (host, port = FLIM_DEFAULT_PORT) => {
  let h = (host || 'localhost').trim();
  if (!/^https?:\/\//.test(h)) h = `http://${h}`;
  // strip trailing slash
  h = h.replace(/\/+$/, '');
  // if host already contains a port, keep it, else append
  const hasPort = /:\d+$/.test(h.replace(/^https?:\/\//, ''));
  return hasPort ? h : `${h}:${port}`;
};

export const flimWsUrl = (host, port = FLIM_DEFAULT_PORT) =>
  `${flimBaseUrl(host, port).replace(/^http/, 'ws')}/data`;

const jsonFetch = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const msg = body && (body.error || body.message);
    throw new Error(msg || `FLIM server error ${response.status}`);
  }
  return body;
};

/** GET /api/card/check -> { success, data: <serial number> } */
export const apiFlimCheckCard = (host, port) =>
  jsonFetch(`${flimBaseUrl(host, port)}/api/card/check`);

/** GET /api/card/list -> { success, data: [{card_id, serial_number, ...}] } */
export const apiFlimListCards = (host, port) =>
  jsonFetch(`${flimBaseUrl(host, port)}/api/card/list`);

/** GET /health -> { success, data: "ok" } (no hardware access) */
export const apiFlimHealth = (host, port) =>
  jsonFetch(`${flimBaseUrl(host, port)}/health`);

/**
 * POST /api/firmware/resolve -> { success, data: <firmware path> }
 * The v2 server accepts both `is_pico_mode` and the legacy `enable100ps`
 * field name; the canonical name is sent here.
 */
export const apiFlimResolveFirmware = (
  host,
  port,
  { sync = 'in', frequencyMhz = 80, channels = [1], channel = 'sma', reconstruction = 'PLF', enable100ps = false }
) =>
  jsonFetch(`${flimBaseUrl(host, port)}/api/firmware/resolve`, {
    method: 'POST',
    body: JSON.stringify({
      sync,
      frequency_mhz: frequencyMhz,
      channels,
      channel,
      sync_connection: 'sma',
      reconstruction,
      is_pico_mode: enable100ps,
    }),
  });

/**
 * The panel's "imaging" action maps to the v2 'tcspc' step: v2 renamed the
 * full acquisition (decay-collecting, exportable) step from v1's "imaging",
 * while 'scouting' remains the live intensity preview.
 */
const V2_STEP = { scouting: 'scouting', imaging: 'tcspc' };

/**
 * Build the /start payload for an imaging experiment (flim-imager 2.x schema).
 * All fields of ExperimentImagingParams that the server requires are present;
 * optional fields (max_frames, dwell_time for PLF, roi_masks...) are added
 * conditionally, mirroring the v2 frontend.
 */
export const buildFlimImagingPayload = ({
  firmware,
  step = 'scouting', // 'scouting' | 'imaging' (mapped to v2 'tcspc')
  frequencyMhz = 80,
  enable100ps = false,
  reconstruction = 'PLF',
  imageWidth = 256,
  imageHeight = 256,
  scanWidth = null, // defaults to image + offsets
  scanHeight = null,
  offsets = { top: 0, right: 0, bottom: 0, left: 0 },
  channels = [true, false, false, false, false, false, false, false],
  dwellTime = 5, // microseconds; ignored for PLF (pixel clock defines geometry)
  maxFrames = null,
  cardId = null, // optional multi-card selection
  serialNumber = null,
}) => {
  const enabledIdx = channels
    .map((on, i) => (on ? i : -1))
    .filter((i) => i >= 0);
  const effScanWidth = scanWidth ?? imageWidth + offsets.left + offsets.right;
  const effScanHeight = scanHeight ?? imageHeight + offsets.top + offsets.bottom;
  const geometry = {
    scan_width: effScanWidth,
    scan_height: effScanHeight,
    image_width: imageWidth,
    image_height: imageHeight,
    offset_top: offsets.top,
    offset_right: offsets.right,
    offset_bottom: offsets.bottom,
    offset_left: offsets.left,
  };
  const params = {
    acquisition_setup: 'Default',
    reconstruction,
    stop_acquisition_mode: 'Frames Count',
    step: V2_STEP[step] ?? step,
    is_preview: false,
    frequency_mhz: frequencyMhz,
    is_pico_mode: enable100ps,
    calibration: false,
    tau_ns: null,
    harmonics: 1,
    ...geometry,
    skip_frames: 0,
    calibration_offsets: [[], [], [], [], [], [], [], []],
    channels,
    bg_active_channels: enabledIdx,
    channels_to_show: enabledIdx,
    show_cps: true,
    show_kcps: false,
    show_sbr: true,
    show_intensity_traces: false,
    show_realtime_phasors: false,
    bin_width: 3000, // v2 default intensity-trace bin width (unused while traces are off)
    export_params: {
      export_data: false,
      export_filename: '',
      export_path: '',
      export_frames: false,
      export_global_image: false,
      export_notes: '',
      export_tags: [],
      channels_metadata: enabledIdx.map((i) => ({ id: i, alias: `Channel ${i + 1}` })),
    },
    acquisition_timestamp: Date.now(),
    reference_file: null,
    interleaving_type: null,
    // decay_roi mirrors the full frame (no ROI restriction)
    decay_roi: { ...geometry },
  };
  // The card slices lines by dwell time only in LF/F reconstruction; with
  // per-pixel markers (PLF) the v2 frontend omits it entirely.
  if (reconstruction !== 'PLF') params.dwell_time = dwellTime;
  if (maxFrames && maxFrames > 0) params.max_frames = maxFrames;
  const payload = {
    firmware,
    frequency: frequencyMhz,
    experiment: { type: 'imaging', params },
  };
  if (cardId !== null && serialNumber !== null) {
    payload.card_id = cardId;
    payload.serial_number = serialNumber;
  }
  return payload;
};

/** POST /start */
export const apiFlimStart = (host, port, payload) =>
  jsonFetch(`${flimBaseUrl(host, port)}/start`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });

/** POST /stop */
export const apiFlimStop = (host, port) =>
  jsonFetch(`${flimBaseUrl(host, port)}/stop`, { method: 'POST' });
