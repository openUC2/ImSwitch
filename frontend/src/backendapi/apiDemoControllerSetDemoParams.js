import createAxiosInstance from "./createAxiosInstance";

const apiDemoControllerSetDemoParams = async (demoParams) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.post("/DemoController/setDemoParams", demoParams); // Send POST request with params
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error setting demo parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiDemoControllerSetDemoParams;

/*
// Example usage:

const setDemoParams = (params) => {
  apiDemoControllerSetDemoParams(params)
    .then((data) => {
      console.log("Demo parameters set successfully:", data);
    })
    .catch((err) => {
      setError("Failed to set demo parameters"); // Handle the error
    });
};
*/