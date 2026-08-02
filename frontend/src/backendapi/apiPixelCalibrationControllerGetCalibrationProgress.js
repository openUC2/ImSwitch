// Poll the live progress of the running automatic calibration.
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGetCalibrationProgress = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    '/PixelCalibrationController/getCalibrationProgress'
  );
  return response.data;
};

export default apiPixelCalibrationControllerGetCalibrationProgress;
