import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerDismissHomingRecommendation = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/PositionerController/dismissHomingRecommendation",
  );
  return response.data;
};

export default apiPositionerControllerDismissHomingRecommendation;
