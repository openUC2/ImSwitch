import createAxiosInstance from "./createAxiosInstance";

// Function to get the current objective
const apiObjectiveControllerGetCurrentObjective = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/ObjectiveController/getCurrentObjective"); // Send GET request to the new endpoint
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching current objective:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiObjectiveControllerGetCurrentObjective;

/*
Example usage:

const fetchCurrentObjective = () => {
  apiObjectiveControllerGetCurrentObjective()
    .then((data) => {
      setCurrentObjective(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch current objective"); // Handle the error
    });
};
*/
