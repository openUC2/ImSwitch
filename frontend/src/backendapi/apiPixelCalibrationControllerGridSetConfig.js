// Set AprilTag grid configuration
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGridSetConfig = async (rows, cols, start_id = 0, pitch_mm = 4.0) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/PixelCalibrationController/gridSetConfig', {
    params: { rows, cols, start_id, pitch_mm }
  });
  return response.data;
};

export default apiPixelCalibrationControllerGridSetConfig;
