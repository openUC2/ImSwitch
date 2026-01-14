// Move to specific AprilTag by ID
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGridMoveToTag = async ({
  target_id,
  roi_tolerance_px = 8.0,
  max_iterations = 30,
  max_step_um = 1000.0,
  settle_time = 0.3
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/PixelCalibrationController/gridMoveToTag', {
    params: {
      target_id,
      roi_tolerance_px,
      max_iterations,
      max_step_um,
      settle_time
    }
  });
  return response.data;
};

export default apiPixelCalibrationControllerGridMoveToTag;
