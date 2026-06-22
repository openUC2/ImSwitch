// Read-noise calibration: poll background acquisition progress
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerGetProgress = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    '/ReadNoiseCalibrationController/getProgress'
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerGetProgress;
