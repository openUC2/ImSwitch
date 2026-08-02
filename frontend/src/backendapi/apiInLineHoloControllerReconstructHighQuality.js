// src/backendapi/apiInLineHoloControllerReconstructHighQuality.js
// Trigger a high-quality (iterative) single-shot hologram reconstruction

import createAxiosInstance from "./createAxiosInstance";

/**
 * High-quality single-shot reconstruction (button press). Uses the current dz
 * and, if enabled, the background normalization. Blocks until done.
 * @param {Object} [opts]
 * @param {("phase_retrieval"|"tv")} [opts.method="phase_retrieval"]
 * @param {number} [opts.iterations=30]
 * @param {number} [opts.supportThreshold] - optional override (0..1)
 * @param {number} [opts.tvWeight] - optional TV weight (tv method only)
 * @returns {Promise<Object>} { success, method, iterations, elapsed, amplitude, phase }
 */
const apiInLineHoloControllerReconstructHighQuality = async ({
  method = "phase_retrieval",
  iterations = 30,
  supportThreshold,
  tvWeight,
} = {}) => {
  const instance = createAxiosInstance();
  const params = new URLSearchParams({
    method,
    iterations: String(iterations),
  });
  if (supportThreshold !== undefined && supportThreshold !== null)
    params.append("support_threshold", String(supportThreshold));
  if (tvWeight !== undefined && tvWeight !== null)
    params.append("tv_weight", String(tvWeight));
  try {
    const response = await instance.get(
      `/InLineHoloController/reconstruct_highquality_inlineholo?${params.toString()}`
    );
    return response.data;
  } catch (error) {
    console.error("Error running high-quality reconstruction:", error);
    throw error;
  }
};

export default apiInLineHoloControllerReconstructHighQuality;
