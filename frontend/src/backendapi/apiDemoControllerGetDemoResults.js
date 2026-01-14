import createAxiosInstance from "./createAxiosInstance";

const apiDemoControllerGetDemoResults = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/DemoController/getDemoResults"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching demo results:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiDemoControllerGetDemoResults;

/*
// Example usage:

const fetchDemoResults = () => {
  apiDemoControllerGetDemoResults()
    .then((data) => {
      setDemoResults(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch demo results"); // Handle the error
    });
};
*/