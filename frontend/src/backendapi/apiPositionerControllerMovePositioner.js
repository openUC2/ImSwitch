// src/api/apiPositionerControllerMovePositioner.js
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerMovePositioner = async ({
  positionerName,
  axis = "X",
  dist,
  isAbsolute = false,
  isBlocking = false,
  speed,
}) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance

    // Build the query string dynamically, only include parameters that are provided
    let url = "/PositionerController/movePositioner?";
    const queryParams = [];

    if (positionerName) {
      queryParams.push(`positionerName=${encodeURIComponent(positionerName)}`);
    }
    if (axis) {
      queryParams.push(`axis=${encodeURIComponent(axis)}`);
    }
    if (dist !== undefined) {
      queryParams.push(`dist=${encodeURIComponent(dist)}`);
    }
    if (isAbsolute !== undefined) {
      queryParams.push(`isAbsolute=${encodeURIComponent(isAbsolute)}`);
    }
    if (isBlocking !== undefined) {
      queryParams.push(`isBlocking=${encodeURIComponent(isBlocking)}`);
    }
    if (speed !== undefined) {
      queryParams.push(`speed=${encodeURIComponent(speed)}`);
    }

    // Join all query parameters with '&'
    url += queryParams.join("&");

    // Send GET request with the constructed URL
    const response = await axiosInstance.get(url);

    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error moving positioner:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiPositionerControllerMovePositioner;

/*
//Example:

  apiPositionerControllerMovePositioner({
    positionerName: "Positioner1",  // Optional
    axis: "Y",                      // Optional (defaults to "X")
    dist: 10,                        // Optional
    isAbsolute: true,                // Optional (defaults to false)
    isBlocking: true,                // Optional (defaults to false)
    speed: 5.0                       // Optional
  })
    .then(positionerResponse => {
      console.log(positionerResponse);  // The response from the API
    })
    .catch(error => {
      console.error("Error moving position:", error);
    });

*/
