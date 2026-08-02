// Read-noise calibration: request the running capture loop to abort
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerStopAcquisition = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/stopAcquisition',
    null
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerStopAcquisition;
