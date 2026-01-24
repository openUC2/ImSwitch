// src/backendapi/apiSTORMControllerSetProcessingParameters.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerSetProcessingParameters = async (processingParameters) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const requestBody = {
      processing_parameters: processingParameters
    };
    
    const url = `/STORMReconController/setSTORMProcessingParameters`;
    const response = await axiosInstance.post(url, requestBody); // Send POST request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error setting STORM processing parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerSetProcessingParameters;

/*
// Example usage:

const setProcessingParameters = () => {
  const processingParameters = {
    threshold: 0.2,
    fit_roi_size: 13,
    fitting_method: "2D_Phasor_CPU",
    filter_type: "bandpass",
    temporal_median_enabled: false,
    update_rate: 10,
    pixel_size_nm: 117.5,
    super_resolution_pixel_size_nm: 10.0,
    bandpass_filter: {
      center: 40.0,
      width: 90.0,
      filter_type: "gauss",
      show_filter: false
    },
    blob_detector: {
      min_threshold: 0.0,
      max_threshold: 255.0,
      min_area: 1.5,
      max_area: 80.0,
      min_circularity: null,
      min_convexity: null,
      min_inertia_ratio: null,
      blob_color: 255,
      min_dist_between_blobs: 0.0
    }
  };
  
  apiSTORMControllerSetProcessingParameters(processingParameters)
    .then((data) => {
      console.log("STORM processing parameters set:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to set STORM processing parameters:", err); // Handle the error
    });
};
*/