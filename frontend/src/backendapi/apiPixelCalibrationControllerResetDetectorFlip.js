// Reset a detector's image flip to off (used before manual calibration so the
// user marks feature points on the raw, unflipped sensor).
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerResetDetectorFlip = async (detectorName) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (detectorName !== null && detectorName !== undefined && detectorName !== '') {
    params.detectorName = String(detectorName);
  }
  const response = await axiosInstance.post(
    '/PixelCalibrationController/resetDetectorFlip',
    null,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerResetDetectorFlip;
