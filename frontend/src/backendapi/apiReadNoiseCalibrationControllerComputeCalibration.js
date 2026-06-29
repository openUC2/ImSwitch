// Read-noise calibration: run cal_readnoise on the session's bright+dark stacks
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerComputeCalibration = async ({
  sessionId = null,
  numBins = null,
  validRangeLow = null,
  validRangeHigh = null,
  saturationImage = false,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = { saturationImage };
  if (sessionId) params.sessionId = sessionId;
  if (numBins !== null && numBins !== undefined) params.numBins = numBins;
  if (validRangeLow !== null && validRangeLow !== undefined && validRangeLow !== '') {
    params.validRangeLow = validRangeLow;
  }
  if (validRangeHigh !== null && validRangeHigh !== undefined && validRangeHigh !== '') {
    params.validRangeHigh = validRangeHigh;
  }
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/computeCalibration',
    null,
    { params }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerComputeCalibration;
