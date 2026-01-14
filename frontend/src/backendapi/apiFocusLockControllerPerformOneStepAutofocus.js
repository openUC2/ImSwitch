import createAxiosInstance from './createAxiosInstance';

/**
 * Perform one-step autofocus using calibration data
 * @param {Object} params - Autofocus parameters
 * @param {number|null} params.target_focus_setpoint - Target focus value to reach (null = use stored setpoint)
 * @param {boolean} params.move_to_focus - Whether to move stage (default: true)
 * @param {number} params.max_attempts - Maximum correction attempts (default: 3)
 * @param {number} params.threshold_um - Success threshold in µm (default: 0.5)
 * @param {number} params.hard_focus_offset - Additional Z offset to apply (default: 0.0)
 * @returns {Promise} Promise containing autofocus result
 */
export default async function apiFocusLockControllerPerformOneStepAutofocus(params = {}) {
  const axios = createAxiosInstance();
  try {
    // Build query parameters
    const queryParams = new URLSearchParams();
    
    if (params.target_focus_setpoint !== null && params.target_focus_setpoint !== undefined) {
      queryParams.append('target_focus_setpoint', params.target_focus_setpoint);
    }
    if (params.move_to_focus !== undefined) {
      queryParams.append('move_to_focus', params.move_to_focus);
    }
    if (params.max_attempts !== undefined) {
      queryParams.append('max_attempts', params.max_attempts);
    }
    if (params.threshold_um !== undefined) {
      queryParams.append('threshold_um', params.threshold_um);
    }
    if (params.hard_focus_offset !== undefined) {
      queryParams.append('hard_focus_offset', params.hard_focus_offset);
    }
    
    const url = `/FocusLockController/performOneStepAutofocus${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    const response = await axios.get(url);
    return response.data;
  } catch (error) {
    console.error('Failed to perform one-step autofocus:', error);
    throw error;
  }
}
