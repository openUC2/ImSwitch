import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerSetHistory = async (history) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(`/MazeGameController/setHistory?history=${history}`);
    return response.data;
  } catch (error) {
    console.error("Error setting history:", error);
    throw error;
  }
};

export default apiMazeGameControllerSetHistory;
