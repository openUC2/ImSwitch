import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerSetDownscale = async (downscale) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(`/MazeGameController/setDownscale?downscale=${downscale}`);
    return response.data;
  } catch (error) {
    console.error("Error setting downscale:", error);
    throw error;
  }
};

export default apiMazeGameControllerSetDownscale;
