// List all detectors plus their per-objective calibration status
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGetAvailableDetectors = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    '/PixelCalibrationController/getAvailableDetectors'
  );
  return response.data;
};

export default apiPixelCalibrationControllerGetAvailableDetectors;
