import createAxiosInstance from "./createAxiosInstance";

const apiDemoControllerStartDemo = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/DemoController/startDemo"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error starting demo:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiDemoControllerStartDemo;

/*
// Example usage:

const startDemo = () => {
  apiDemoControllerStartDemo()
    .then((data) => {
      console.log("Demo started successfully:", data);
    })
    .catch((err) => {
      setError("Failed to start demo"); // Handle the error
    });
};
*/