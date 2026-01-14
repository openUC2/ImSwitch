// src/api/apiPositionerControllerGetPositions.js
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerGetPositions = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/PositionerController/getPositionerPositions"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching positioner positions:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiPositionerControllerGetPositions;

/*
// Example usage:

const fetchPositionerPositions = () => {
  apiPositionerControllerGetPositions()
    .then((data) => {
      setPositionerData(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch positioner positions"); // Handle the error
    });
};
*/
