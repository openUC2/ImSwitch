// Detect AprilTags in the current frame
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGridDetectTags = async (save_annotated = false) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/PixelCalibrationController/gridDetectTags', {
    params: { save_annotated }
  });
  return response.data;
};

export default apiPixelCalibrationControllerGridDetectTags;
