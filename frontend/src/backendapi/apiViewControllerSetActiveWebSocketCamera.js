// src/backendapi/apiViewControllerSetActiveWebSocketCamera.js
import createAxiosInstance from "./createAxiosInstance";

const apiViewControllerSetActiveWebSocketCamera = async (cameraName) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const response = await axiosInstance.post("/ViewController/setActiveWebSocketCamera", {
      cameraName: cameraName
    });
    
    return response.data;
  } catch (error) {
    console.error("Failed to set active WebSocket camera:", error);
    throw error;
  }
};

export default apiViewControllerSetActiveWebSocketCamera;