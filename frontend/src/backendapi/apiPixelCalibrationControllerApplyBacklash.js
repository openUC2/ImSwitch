// Apply a known backlash (µm) to the active positioner for one axis (no stage motion).
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerApplyBacklash = async ({
  axis = 'X',
  backlashUm = 0.0,
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/PixelCalibrationController/applyBacklash',
    null,
    { params: { axis, backlashUm } }
  );
  return response.data;
};

export default apiPixelCalibrationControllerApplyBacklash;
