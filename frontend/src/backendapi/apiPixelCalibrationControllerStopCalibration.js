// Request the running automatic calibration to abort at the next step.
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerStopCalibration = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/PixelCalibrationController/stopPixelCalibration'
  );
  return response.data;
};

export default apiPixelCalibrationControllerStopCalibration;
