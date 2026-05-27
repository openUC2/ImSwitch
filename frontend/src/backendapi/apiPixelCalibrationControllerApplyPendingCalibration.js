// Apply (and persist) a pending calibration, optionally with edits.
// FastAPI maps simple-typed parameters (str) to query params and complex-typed
// parameters (list/dict) to the JSON body. We must respect that split or the
// server replies 422.
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerApplyPendingCalibration = async ({
  detectorName,
  objectiveId = null,
  affineMatrix = null,
  metrics = null,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { detectorName };
  if (objectiveId !== null && objectiveId !== undefined && objectiveId !== '') {
    params.objectiveId = String(objectiveId);
  }
  const body = {};
  if (affineMatrix !== null) body.affineMatrix = affineMatrix;
  if (metrics !== null) body.metrics = metrics;
  const response = await axiosInstance.post(
    '/PixelCalibrationController/applyPendingCalibration',
    Object.keys(body).length ? body : null,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerApplyPendingCalibration;
