import createAxiosInstance from "./createAxiosInstance";

// Set objective metadata parameters (name, NA, magnification, pixelsize) for a given slot
const apiObjectiveControllerSetObjectiveParameters = async (params) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    `/ObjectiveController/setObjectiveParameters`,
    { params }
  );
  return response.data;
};

export default apiObjectiveControllerSetObjectiveParameters;
