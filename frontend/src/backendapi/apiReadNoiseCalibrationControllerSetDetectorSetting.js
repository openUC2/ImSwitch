// Read-noise calibration: set a single numeric detector parameter (exposure/gain/...)
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerSetDetectorSetting = async ({
  detectorName = null,
  name,
  value,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { name, value };
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/setDetectorSetting',
    null,
    { params }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerSetDetectorSetting;
