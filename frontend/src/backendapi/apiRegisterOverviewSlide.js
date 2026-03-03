import createAxiosInstance from "./createAxiosInstance";

/**
 * Register a slide with corner picks – compute homography.
 * POST /ExperimentController/registerOverviewSlide
 */
const apiRegisterOverviewSlide = async (registrationData) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/ExperimentController/registerOverviewSlide",
    registrationData
  );
  return response.data;
};

export default apiRegisterOverviewSlide;
