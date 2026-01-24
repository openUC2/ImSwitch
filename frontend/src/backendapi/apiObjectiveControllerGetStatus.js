import createAxiosInstance from "./createAxiosInstance";

// Function to get the current status
const apiObjectiveControllerGetStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/ObjectiveController/getstatus"); // Send GET request to the new endpoint
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching current status:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiObjectiveControllerGetStatus;

/*
Example usage:

const fetchCurrentStatus = () => {
  apiObjectiveControllerGetStatus()
    .then((data) => {
      setCurrentStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch current status"); // Handle the error
    });
};
*/
