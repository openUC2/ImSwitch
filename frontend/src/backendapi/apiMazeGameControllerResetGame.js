import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerResetGame = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/MazeGameController/resetGame");
    return response.data;
  } catch (error) {
    console.error("Error resetting maze game:", error);
    throw error;
  }
};

export default apiMazeGameControllerResetGame;
