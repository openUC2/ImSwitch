// Read-noise calibration: switch all illumination 'off' (saving prior state) or 'restore'
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerSetIllumination = async (state = 'off') => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/setIllumination',
    null,
    { params: { state } }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerSetIllumination;
