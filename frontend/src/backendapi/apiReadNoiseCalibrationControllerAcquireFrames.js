// Read-noise calibration: capture a 'bright' or 'dark' stack in the background
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerAcquireFrames = async (kind, count = null) => {
  const axiosInstance = createAxiosInstance();
  const params = { kind };
  if (count !== null && count !== undefined) params.count = count;
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/acquireFrames',
    null,
    { params }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerAcquireFrames;
