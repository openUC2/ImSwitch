// Identify axes using AprilTag grid
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerOverviewIdentifyAxes = async (stepUm = 2000.0, debug_dir = null) => {
  const axiosInstance = createAxiosInstance();
  const params = { stepUm };
  if (debug_dir) params.debug_dir = debug_dir;
  
  const response = await axiosInstance.get('/PixelCalibrationController/overviewIdentifyAxes', {
    params
  });
  return response.data;
};

export default apiPixelCalibrationControllerOverviewIdentifyAxes;
