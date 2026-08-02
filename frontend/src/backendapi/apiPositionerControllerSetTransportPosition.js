// Store the transportation position and persist it to the setup JSON.
// With useCurrent=true the current stage pose is snapshotted; otherwise the
// provided a/x/y/z values are used. Scalars are routed as query params (FastAPI).
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerSetTransportPosition = async ({
  positionerName = null,
  useCurrent = true,
  a = null,
  x = null,
  y = null,
  z = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = { useCurrent };
  if (positionerName) params.positionerName = positionerName;
  if (a !== null) params.a = a;
  if (x !== null) params.x = x;
  if (y !== null) params.y = y;
  if (z !== null) params.z = z;
  const response = await axiosInstance.post(
    "/PositionerController/setTransportPosition",
    null,
    { params },
  );
  return response.data;
};

export default apiPositionerControllerSetTransportPosition;
