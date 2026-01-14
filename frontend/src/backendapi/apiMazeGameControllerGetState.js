import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerGetState = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/MazeGameController/getState");
    return response.data;
  } catch (error) {
    console.error("Error fetching maze game state:", error);
    throw error;
  }
};

export default apiMazeGameControllerGetState;
