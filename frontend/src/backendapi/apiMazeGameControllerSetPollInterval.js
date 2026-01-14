import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerSetPollInterval = async (interval) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(`/MazeGameController/setPollInterval?interval=${interval}`);
    return response.data;
  } catch (error) {
    console.error("Error setting poll interval:", error);
    throw error;
  }
};

export default apiMazeGameControllerSetPollInterval;
