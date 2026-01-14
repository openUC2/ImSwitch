// Verify homing functionality
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerOverviewVerifyHoming = async (maxTimeS = 20.0) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/PixelCalibrationController/overviewVerifyHoming', {
    params: { maxTimeS }
  });
  return response.data;
};

export default apiPixelCalibrationControllerOverviewVerifyHoming;
