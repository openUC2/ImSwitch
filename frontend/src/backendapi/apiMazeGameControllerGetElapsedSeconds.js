import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerGetElapsedSeconds = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/MazeGameController/getElapsedSeconds");
    return response.data;
  } catch (error) {
    console.error("Error fetching maze game elapsed time:", error);
    throw error;
  }
};

export default apiMazeGameControllerGetElapsedSeconds;
