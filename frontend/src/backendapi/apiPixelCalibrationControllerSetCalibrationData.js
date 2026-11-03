// Directly write calibration data for a (detector, objective) pair.
// detectorName and objectiveId are routed as query params (FastAPI),
// affineMatrix and metrics travel in the JSON body.
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerSetCalibrationData = async ({
  detectorName,
  objectiveId,
  affineMatrix,
  metrics,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (detectorName) params.detectorName = detectorName;
  if (objectiveId !== null && objectiveId !== undefined && objectiveId !== '') {
    params.objectiveId = String(objectiveId);
  }
  const body = { affineMatrix };
  if (metrics !== undefined) body.metrics = metrics;
  const response = await axiosInstance.post(
    '/PixelCalibrationController/setCalibrationData',
    body,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerSetCalibrationData;
