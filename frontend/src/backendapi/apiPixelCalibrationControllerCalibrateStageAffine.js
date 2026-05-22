// Start an affine stage/camera calibration in the background (per detector + objective)
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerCalibrateStageAffine = async ({
  detectorName = null,
  objectiveId = null,
  stepSizeUm = 100.0,
  pattern = 'cross',
  nSteps = 4,
  backlashUm = 0.0,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { stepSizeUm, pattern, nSteps, backlashUm };
  if (detectorName) params.detectorName = detectorName;
  if (objectiveId !== null && objectiveId !== undefined && objectiveId !== '') {
    params.objectiveId = String(objectiveId);
  }
  const response = await axiosInstance.post(
    '/PixelCalibrationController/calibrateStageAffine',
    null,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerCalibrateStageAffine;
