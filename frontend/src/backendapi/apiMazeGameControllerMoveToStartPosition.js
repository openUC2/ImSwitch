import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerMoveToStartPosition = async (x, y) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(`/MazeGameController/moveToStartPosition?x=${x}&y=${y}`);
    return response.data;
  } catch (error) {
    console.error("Error moving to start position:", error);
    throw error;
  }
};

export default apiMazeGameControllerMoveToStartPosition;
