// Discard a pending (un-applied) calibration
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerDiscardPendingCalibration = async (detectorName) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/PixelCalibrationController/discardPendingCalibration',
    { detectorName }
  );
  return response.data;
};

export default apiPixelCalibrationControllerDiscardPendingCalibration;
