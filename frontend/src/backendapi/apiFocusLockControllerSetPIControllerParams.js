import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerSetPIControllerParams = async (params) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    // Map camelCase to snake_case for API
    const apiParams = {};
    if (params.kp !== undefined) apiParams.kp = params.kp;
    if (params.ki !== undefined) apiParams.ki = params.ki;
    if (params.setPoint !== undefined) apiParams.set_point = params.setPoint;
    if (params.safetyDistanceLimit !== undefined) apiParams.safety_distance_limit = params.safetyDistanceLimit;
    if (params.safetyMoveLimit !== undefined) apiParams.safety_move_limit = params.safetyMoveLimit;
    if (params.minStepThreshold !== undefined) apiParams.min_step_threshold = params.minStepThreshold;
    if (params.safetyMotionActive !== undefined) apiParams.safety_motion_active = params.safetyMotionActive;
    
    const response = await axiosInstance.get('/FocusLockController/setPIControllerParams', {
      params: apiParams
    });
    return response.data;
  } catch (error) {
    console.error("Error setting PI controller parameters:", error);
    throw error;
  }
};

export default apiFocusLockControllerSetPIControllerParams;
