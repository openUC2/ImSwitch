// Read-noise calibration: detectors, illumination sources, active session, progress
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerGetStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    '/ReadNoiseCalibrationController/getStatus'
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerGetStatus;
