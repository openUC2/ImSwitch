import createAxiosInstance from "./createAxiosInstance";

// Function to calibrate the objective
const apiObjectiveControllerCalibrateObjective = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/ObjectiveController/calibrateObjective"); // Send GET request to the new endpoint
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error calibrating the objective:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiObjectiveControllerCalibrateObjective;

/*
Example usage:

const calibrateObjective = () => {
  apiObjectiveControllerCalibrateObjective()
    .then((data) => {
      setCalibrationResult(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to calibrate the objective"); // Handle the error
    });
};
*/
