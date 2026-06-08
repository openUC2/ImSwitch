import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerDismissHomingRecommendation = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/PositionerController/dismissHomingRecommendation",
  );
  return response.data;
};

export default apiPositionerControllerDismissHomingRecommendation;
