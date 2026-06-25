// Read the human-readable microscope stand model (default "openUC2 FRAME").
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetMicroscopeStandName = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getMicroscopeStandName",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetMicroscopeStandName;
