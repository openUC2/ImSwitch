import createAxiosInstance from "./createAxiosInstance";

/**
 * Read board/air temperatures from the firmware.
 *
 * @returns {Promise<Object>} { pcb, air, esp, pcb_ok, air_ok }
 */
const apiUC2ConfigControllerGetBoardTemperature = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getBoardTemperature",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetBoardTemperature;
