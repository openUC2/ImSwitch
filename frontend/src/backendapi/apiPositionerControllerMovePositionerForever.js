// src/backendapi/apiPositionerControllerMovePositionerForever.js
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerMovePositionerForever = async ({
  axis = "X",
  speed = 5000,
  is_stop = false,
}) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance

    // Build the query string dynamically
    let url = "/PositionerController/movePositionerForever?";
    const queryParams = [];

    if (axis) {
      queryParams.push(`axis=${encodeURIComponent(axis)}`);
    }
    if (speed !== undefined) {
      queryParams.push(`speed=${encodeURIComponent(speed)}`);
    }
    if (is_stop !== undefined) {
      queryParams.push(`is_stop=${encodeURIComponent(is_stop)}`);
    }

    // Join all query parameters with '&'
    url += queryParams.join("&");

    // Send GET request with the constructed URL
    const response = await axiosInstance.get(url);

    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error moving positioner forever:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiPositionerControllerMovePositionerForever;

/*
// Example:

  apiPositionerControllerMovePositionerForever({
    axis: "X",                // Optional (defaults to "X")
    speed: 5000,              // Optional (defaults to 5000)
    is_stop: false            // Optional (defaults to false)
  })
    .then(positionerResponse => {
      console.log(positionerResponse);  // The response from the API
    })
    .catch(error => {
      console.error("Error moving positioner forever:", error);
    });

*/
