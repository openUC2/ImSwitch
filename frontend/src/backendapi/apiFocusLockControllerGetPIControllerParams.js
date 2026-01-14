import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetPIControllerParams = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getPIControllerParams');
    
    // Map snake_case to camelCase for React
    const data = response.data;
    return {
      kp: data.kp || 0.0,
      ki: data.ki || 0.0,
      setPoint: data.set_point || 0.0,
      safetyDistanceLimit: data.safety_distance_limit || 500.0,
      safetyMoveLimit: data.safety_move_limit || 3.0,
      minStepThreshold: data.min_step_threshold || 0.002,
      safetyMotionActive: data.safety_motion_active || false,
    };
  } catch (error) {
    console.error("Error getting PI controller parameters:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetPIControllerParams;
