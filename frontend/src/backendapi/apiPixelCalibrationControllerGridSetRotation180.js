// Set 180° rotation flag for AprilTag grid
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGridSetRotation180 = async (rotated = false) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/PixelCalibrationController/gridSetRotation180', {
    params: { rotated }
  });
  return response.data;
};

export default apiPixelCalibrationControllerGridSetRotation180;
