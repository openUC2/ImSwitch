import createAxiosInstance from "./createAxiosInstance";

/**
 * Query CAN-bus power and emergency-stop status.
 *
 * @returns {Promise<Object>} { power, emergencyActive, available, estop }
 */
const apiUC2ConfigControllerGetBusStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getBusStatus");
  return response.data;
};

export default apiUC2ConfigControllerGetBusStatus;
