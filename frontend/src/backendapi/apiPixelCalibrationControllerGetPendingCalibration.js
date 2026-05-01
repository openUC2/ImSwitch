// Get the currently pending (un-applied) calibration result for a detector
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGetPendingCalibration = async (detectorName = null) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.get(
    '/PixelCalibrationController/getPendingCalibration',
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerGetPendingCalibration;
