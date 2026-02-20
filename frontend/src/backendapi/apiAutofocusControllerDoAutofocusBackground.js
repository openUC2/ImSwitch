// src/backendapi/apiAutofocusControllerDoAutofocusBackground.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Run software autofocus (Z-sweep) in the background.
 *
 * @param {Object} params - Autofocus parameters
 * @param {number} [params.rangez=100] - Z range ±µm from current position
 * @param {number} [params.resolutionz=10] - Z step size (µm)
 * @param {number} [params.defocusz=0] - Defocus offset
 * @param {string} [params.axis="Z"] - Axis to focus on
 * @param {number} [params.tSettle=0.1] - Settle time after each Z step (s)
 * @param {boolean} [params.isDebug=false] - Enable debug mode
 * @param {number} [params.nGauss=7] - Gaussian kernel size
 * @param {number} [params.nCropsize=2048] - Crop size for focus algorithm
 * @param {string} [params.focusAlgorithm="LAPE"] - Focus quality algorithm
 * @param {number} [params.static_offset=0] - Static Z offset after focusing
 * @param {boolean} [params.twoStage=false] - Use two-stage autofocus
 * @returns {Promise<number|null>} Best focus Z position, or null if failed
 */
const apiAutofocusControllerDoAutofocusBackground = async (params = {}) => {
  const axiosInstance = createAxiosInstance();

  // doAutofocusBackground uses GET with query params (all scalar types)
  const response = await axiosInstance.get(
    "/AutofocusController/autoFocus",
    {
      params: {
        rangez: params.rangez ?? 100,
        resolutionz: params.resolutionz ?? 10,
        defocusz: params.defocusz ?? 0,
        axis: params.axis ?? "Z",
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

export default apiAutofocusControllerDoAutofocusBackground;
