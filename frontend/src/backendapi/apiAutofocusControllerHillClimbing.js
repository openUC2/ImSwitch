// src/backendapi/apiAutofocusControllerHillClimbing.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Run hill-climbing autofocus via AutofocusController.autoFocusHillClimbing.
 * Iteratively searches for peak contrast by moving in the gradient direction,
 * reversing and halving the step on decline, until convergence.
 *
 * @param {Object} params - Hill-climbing autofocus parameters
 * @param {number} [params.initial_step=20] - Starting step size (µm)
 * @param {number} [params.min_step=1] - Min step size / convergence criterion (µm)
 * @param {number} [params.step_reduction=0.5] - Factor to reduce step on reversal
 * @param {number} [params.max_iterations=50] - Max iterations safety limit
 * @param {number} [params.tSettle=0.1] - Settle time after each Z step (s)
 * @param {number} [params.nCropsize=2048] - Crop size for focus algorithm
 * @param {string} [params.focusAlgorithm="LAPE"] - Focus quality algorithm
 * @param {number} [params.nGauss=0] - Gaussian blur sigma
 * @param {number} [params.static_offset=0] - Static Z offset after focusing
 * @returns {Promise<Object>} { status: "started", centerZ, method } or error
 */
const apiAutofocusControllerHillClimbing = async (params = {}) => {
  const axiosInstance = createAxiosInstance();

  const response = await axiosInstance.get(
    "/AutofocusController/autoFocusHillClimbing",
    {
      params: {
        initial_step: params.initial_step ?? 20,
        min_step: params.min_step ?? 1,
        step_reduction: params.step_reduction ?? 0.5,
        max_iterations: params.max_iterations ?? 50,
        tSettle: params.tSettle ?? 0.1,
        nCropsize: params.nCropsize ?? 2048,
        focusAlgorithm: params.focusAlgorithm ?? "LAPE",
        nGauss: params.nGauss ?? 0,
        static_offset: params.static_offset ?? 0,
      },
    }
  );
  return response.data;
};

export default apiAutofocusControllerHillClimbing;
