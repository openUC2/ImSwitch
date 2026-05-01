// Perform stage affine calibration (per-detector)
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerCalibrateStageAffine = async ({
  detectorName = null,
  stepSizeUm = 100.0,
  pattern = "cross",
  nSteps = 4,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { stepSizeUm, pattern, nSteps };
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.post(
    '/PixelCalibrationController/calibrateStageAffine',
    null,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerCalibrateStageAffine;
