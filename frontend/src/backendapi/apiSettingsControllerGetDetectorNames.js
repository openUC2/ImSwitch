import createAxiosInstance from "./createAxiosInstance";

const apiSettingsControllerGetDetectorNames = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/SettingsController/getDetectorNames"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching detector names:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSettingsControllerGetDetectorNames;

/*
// Example usage:

const fetchDetectorNames = () => {
  apiSettingsControllerGetDetectorNames()
    .then((data) => {
      setDetectorNames(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch detector names"); // Handle the error
    });
};
*/