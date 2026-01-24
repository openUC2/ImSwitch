import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerRunFocusCalibrationDynamic = async ({ scan_range_um = 2000, num_steps = 20, settle_time = 0.5 } = {}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post('/FocusLockController/runFocusCalibrationDynamic', null, {
      params: { 
        scan_range_um, 
        num_steps, 
        settle_time 
      }
    });
    return response.data;
  } catch (error) {
    console.error("Error running dynamic focus calibration:", error);
    throw error;
  }
};

export default apiFocusLockControllerRunFocusCalibrationDynamic;
