// Read-noise calibration: list all stored calibration sessions (most recent first)
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerListSessions = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    '/ReadNoiseCalibrationController/listSessions'
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerListSessions;
