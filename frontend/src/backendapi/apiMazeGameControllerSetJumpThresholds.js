import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerSetJumpThresholds = async (low, high) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(`/MazeGameController/setJumpThresholds?low=${low}&high=${high}`);
    return response.data;
  } catch (error) {
    console.error("Error setting jump thresholds:", error);
    throw error;
  }
};

export default apiMazeGameControllerSetJumpThresholds;
