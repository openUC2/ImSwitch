// src/backendapi/apiSTORMControllerStartReconstructionLocal.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerStartReconstructionLocal = async (reconstructionRequest) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const url = `/STORMReconController/startSTORMReconstructionLocal`;
    const response = await axiosInstance.post(url, reconstructionRequest); // Send POST request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error starting STORM reconstruction local:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerStartReconstructionLocal;

/*
// Example usage:

const startReconstructionLocal = () => {
  const reconstructionRequest = {
    session_id: "test_session",
    acquisition_parameters: {
      session_id: "test_session",
      exposure_time: 50,
      max_frames: -1,
      crop_enabled: true,
      crop_x: 0,
      crop_y: 0,
      crop_width: 512,
      crop_height: 512,
      save_enabled: true,
      save_directory: "STORM",
      save_format: "tiff",
      process_locally: true,
      process_arkitekt: false,
      processing_parameters: null
    },
    processing_parameters: {
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
    },
    save_enabled: true
  };
  
  apiSTORMControllerStartReconstructionLocal(reconstructionRequest)
    .then((data) => {
      console.log("STORM reconstruction local started:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to start STORM reconstruction local:", err); // Handle the error
    });
};
*/