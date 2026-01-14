import createAxiosInstance from "./createAxiosInstance";

const apiMazeGameControllerStartGame = async (startX = null, startY = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    let url = "/MazeGameController/startGame";
    
    // Add optional start position parameters if provided
    const params = [];
    if (startX !== null) params.push(`startX=${startX}`);
    if (startY !== null) params.push(`startY=${startY}`);
    if (params.length > 0) url += `?${params.join("&")}`;
    
    const response = await axiosInstance.get(url);
    return response.data;
  } catch (error) {
    console.error("Error starting maze game:", error);
    throw error;
  }
};

export default apiMazeGameControllerStartGame;
