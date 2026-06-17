// src/backendapi/apiPositionerControllerGetDevicePositionAxis.js
//
// GET /PositionerController/getDevicePositionAxis?axis=X
//
// Returns the raw device position for one axis (no offset applied). Used as
// a stable reference for offset calibration; the firmware preserves device
// steps across software restarts.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerGetDevicePositionAxis = async ({
  axis = "X",
  positionerName,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = { axis };
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/getDevicePositionAxis",
    { params }
  );
  return response.data;
};

export default apiPositionerControllerGetDevicePositionAxis;
