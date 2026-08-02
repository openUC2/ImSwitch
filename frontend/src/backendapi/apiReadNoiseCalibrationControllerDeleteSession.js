// Read-noise calibration: delete a stored session folder
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerDeleteSession = async (sessionId) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/deleteSession',
    null,
    { params: { sessionId } }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerDeleteSession;
