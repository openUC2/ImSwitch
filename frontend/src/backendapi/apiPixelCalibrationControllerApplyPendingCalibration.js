// Apply (and persist) a pending calibration, optionally with edits
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerApplyPendingCalibration = async ({
  detectorName,
  affineMatrix = null,
  metrics = null,
}) => {
  const axiosInstance = createAxiosInstance();
  const body = { detectorName };
  if (affineMatrix !== null) body.affineMatrix = affineMatrix;
  if (metrics !== null) body.metrics = metrics;
  const response = await axiosInstance.post(
    '/PixelCalibrationController/applyPendingCalibration',
    body
  );
  return response.data;
};

export default apiPixelCalibrationControllerApplyPendingCalibration;
