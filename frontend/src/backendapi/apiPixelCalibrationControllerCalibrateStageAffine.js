// Perform stage affine calibration
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerCalibrateStageAffine = async ({
  objectiveId = 0,
  stepSizeUm = 100.0,
  pattern = "cross",
  nSteps = 4,
  validate = false
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post('/PixelCalibrationController/calibrateStageAffine', null, {
    params: {
      objectiveId,
      stepSizeUm,
      pattern,
      nSteps,
      validate
    }
  });
  return response.data;
};

export default apiPixelCalibrationControllerCalibrateStageAffine;
