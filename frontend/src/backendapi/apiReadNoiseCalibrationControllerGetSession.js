// Read-noise calibration: full session + result + comment for one stored session
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerGetSession = async (sessionId) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    '/ReadNoiseCalibrationController/getSession',
    { params: { sessionId } }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerGetSession;
