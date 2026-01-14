import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerSetCropSize = async (size) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(`/MazeGameController/setCropSize?size=${size}`);
    return response.data;
  } catch (error) {
    console.error("Error setting crop size:", error);
    throw error;
  }
};

export default apiMazeGameControllerSetCropSize;
