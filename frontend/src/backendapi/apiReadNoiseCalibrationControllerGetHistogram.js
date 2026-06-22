// Read-noise calibration: grab one frame and return its histogram + saturation stats
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerGetHistogram = async (detectorName = null, numBins = 128) => {
  const axiosInstance = createAxiosInstance();
  const params = { numBins };
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.get(
    '/ReadNoiseCalibrationController/getHistogram',
    { params }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerGetHistogram;
