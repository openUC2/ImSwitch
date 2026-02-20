// src/backendapi/apiAutofocusControllerDoAutofocusBackground.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Run software autofocus (Z-sweep) via AutofocusController.autoFocus.
 * This starts the AF in a background thread on the backend.
 *
 * @param {Object} params - Autofocus parameters
 * @param {number} [params.rangez=100] - Z range ±µm from current position
 * @param {number} [params.resolutionz=10] - Z step size (µm)
 * @param {number} [params.defocusz=0] - Defocus offset
 * @param {number} [params.tSettle=0.1] - Settle time after each Z step (s)
 * @param {boolean} [params.isDebug=false] - Enable debug mode
 * @param {number} [params.nGauss=7] - Gaussian kernel size
 * @param {number} [params.nCropsize=2048] - Crop size for focus algorithm
 * @param {string} [params.focusAlgorithm="LAPE"] - Focus quality algorithm
 * @param {number} [params.static_offset=0] - Static Z offset after focusing
 * @param {boolean} [params.twoStage=false] - Use two-stage autofocus
 * @param {string} [params.illuminationChannel=""] - Illumination channel for AF
 * @returns {Promise<Object>} { status: "started", rangez, centerZ } or error
 */
const apiAutofocusControllerDoAutofocusBackground = async (params = {}) => {
  const axiosInstance = createAxiosInstance();

  // autoFocus uses GET with query params (all scalar types)
  const response = await axiosInstance.get(
    "/AutofocusController/autoFocus",
    {
      params: {
        rangez: params.rangez ?? 100,
        resolutionz: params.resolutionz ?? 10,
        defocusz: params.defocusz ?? 0,
        tSettle: params.tSettle ?? 0.1,
        isDebug: params.isDebug ?? false,
        nGauss: params.nGauss ?? 7,
        nCropsize: params.nCropsize ?? 2048,
        focusAlgorithm: params.focusAlgorithm ?? "LAPE",
        static_offset: params.static_offset ?? 0,
        twoStage: params.twoStage ?? false,
      },
    }
  );
  return response.data;
};

/**
 * Poll AutofocusController status until it finishes or errors.
 * @param {number} [intervalMs=500] - Poll interval in ms
 * @param {number} [timeoutMs=120000] - Max wait time in ms
 * @returns {Promise<Object>} Final status object { state, isRunning, currentZ, ... }
 */
export const waitForAutofocusComplete = async (intervalMs = 500, timeoutMs = 120000) => {
  const axiosInstance = createAxiosInstance();
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const res = await axiosInstance.get("/AutofocusController/getAutofocusStatus");
    const status = res.data;
    // AF states: idle, starting, scanning, fitting, moving, finished, error, aborted
    if (!status.isRunning || ["finished", "error", "aborted", "idle"].includes(status.state)) {
      return status;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Autofocus timed out");
};

export default apiAutofocusControllerDoAutofocusBackground;
