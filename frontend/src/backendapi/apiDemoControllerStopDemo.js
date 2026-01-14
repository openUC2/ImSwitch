import createAxiosInstance from "./createAxiosInstance";

const apiDemoControllerStopDemo = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/DemoController/stopDemo"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping demo:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiDemoControllerStopDemo;

/*
// Example usage:

const stopDemo = () => {
  apiDemoControllerStopDemo()
    .then((data) => {
      console.log("Demo stopped successfully:", data);
    })
    .catch((err) => {
      setError("Failed to stop demo"); // Handle the error
    });
};
*/