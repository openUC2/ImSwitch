// Toggle AprilTag overlay on/off
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerToggleAprilTagOverlay = async (enabled = true) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/PixelCalibrationController/gridSetStreamOverlay', {
    params: { enabled }
  });
  return response.data;
};

export default apiPixelCalibrationControllerToggleAprilTagOverlay;
