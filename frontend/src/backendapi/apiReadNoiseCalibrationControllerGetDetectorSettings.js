// Read-noise calibration: current exposure/gain/blacklevel for a detector
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerGetDetectorSettings = async (detectorName = null) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.get(
    '/ReadNoiseCalibrationController/getDetectorSettings',
    { params }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerGetDetectorSettings;
