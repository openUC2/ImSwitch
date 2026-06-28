// Measure one axis' backlash (µm) from a reversing, camera-tracked 1-D scan.
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerMeasureBacklash = async ({
  axis = 'X',
  stepSizeUm = 20.0,
  nSteps = 8,
  detectorName = null,
  applyToStage = false,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { axis, stepSizeUm, nSteps, applyToStage };
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.post(
    '/PixelCalibrationController/measureBacklash',
    null,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerMeasureBacklash;
