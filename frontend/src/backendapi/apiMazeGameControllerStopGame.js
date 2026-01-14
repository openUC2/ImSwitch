import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerStopGame = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/MazeGameController/stopGame");
    return response.data;
  } catch (error) {
    console.error("Error stopping maze game:", error);
    throw error;
  }
};

export default apiMazeGameControllerStopGame;
