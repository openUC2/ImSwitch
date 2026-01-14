import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerGetCounter = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/MazeGameController/getCounter");
    return response.data;
  } catch (error) {
    console.error("Error fetching maze game counter:", error);
    throw error;
  }
};

export default apiMazeGameControllerGetCounter;
