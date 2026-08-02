import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /ExperimentController/runAshlarStitching
 *
 * Launches Ashlar stitching in a background thread on the backend and
 * returns immediately.  Poll getOverviewAsyncStatus for completion.
 */
const apiExperimentControllerRunAshlarStitching = async ({
  pixelSize = 1.0,
  maximumShift = 50.0,
  alignChannel = 0,
  experimentDir = "",
} = {}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/ExperimentController/runAshlarStitching",
      { params: { pixelSize, maximumShift, alignChannel, experimentDir } }
    );
    return response.data;
  } catch (error) {
    console.error("Error starting Ashlar stitching:", error);
    throw error;
  }
};

export default apiExperimentControllerRunAshlarStitching;
