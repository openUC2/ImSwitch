import createAxiosInstance from "./createAxiosInstance";

const apiDemoControllerGetDemoParams = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/DemoController/getDemoParams"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching demo parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiDemoControllerGetDemoParams;

/*
// Example usage:

const fetchDemoParams = () => {
  apiDemoControllerGetDemoParams()
    .then((data) => {
      setDemoParams(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch demo parameters"); // Handle the error
    });
};
*/