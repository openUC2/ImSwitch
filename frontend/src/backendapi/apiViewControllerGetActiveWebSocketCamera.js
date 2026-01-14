// src/backendapi/apiViewControllerGetActiveWebSocketCamera.js
import createAxiosInstance from "./createAxiosInstance";

const apiViewControllerGetActiveWebSocketCamera = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const response = await axiosInstance.get("/ViewController/getActiveWebSocketCamera");
    
    return response.data;
  } catch (error) {
    console.error("Failed to get active WebSocket camera:", error);
    throw error;
  }
};

export default apiViewControllerGetActiveWebSocketCamera;