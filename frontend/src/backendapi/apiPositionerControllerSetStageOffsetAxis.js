// src/backendapi/apiPositionerControllerSetStageOffsetAxis.js
//
// Set the stage offset for one axis via the canonical contract:
//   offset = currentDevicePosition - knownPosition
// so that the stage reports ``knownPosition`` at the current physical place.
//
// Usual call: pass ``knownPosition`` only and let the backend snapshot the
// raw device position atomically. Pass ``currentDevicePosition`` only when
// you need a deterministic reference (e.g. the brightest sample of a heatmap
// captured in device coords). ``knownOffset`` is an escape hatch that stores
// the offset verbatim.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerSetStageOffsetAxis = async ({
  positionerName,
  knownPosition,
  currentDevicePosition,
  knownOffset,
  axis = "X",
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = { axis };
    if (positionerName !== undefined && positionerName !== null) {
      params.positionerName = positionerName;
    }
    if (knownPosition !== undefined && knownPosition !== null) {
      params.knownPosition = knownPosition;
    }
    if (currentDevicePosition !== undefined && currentDevicePosition !== null) {
      params.currentDevicePosition = currentDevicePosition;
    }
    if (knownOffset !== undefined && knownOffset !== null) {
      params.knownOffset = knownOffset;
    }
    const response = await axiosInstance.get(
      "/PositionerController/setStageOffsetAxis",
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error setting stage offset axis:", error);
    throw error;
  }
};

export default apiPositionerControllerSetStageOffsetAxis;
