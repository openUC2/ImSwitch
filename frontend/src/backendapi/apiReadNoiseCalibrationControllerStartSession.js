// Read-noise calibration: create a new session folder and make it active
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerStartSession = async ({
  name = '',
  detectorName = null,
  nBright = 20,
  nDark = 20,
  numBins = 100,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { name, nBright, nDark, numBins };
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/startSession',
    null,
    { params }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerStartSession;
