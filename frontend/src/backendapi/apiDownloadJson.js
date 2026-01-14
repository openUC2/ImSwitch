import createAxiosInstance from "../backendapi/createAxiosInstance";

const apiDownloadJson = async (filePath) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get(filePath, { responseType: "json" }); // Fetch JSON from provided path
    return response.data; // Return JSON data as an object
  } catch (error) {
    console.error(`Error downloading JSON from ${filePath}:`, error);
    throw error; // Throw error for caller to handle
  }
};

export default apiDownloadJson;

/*
// Example usage:

const fetchJsonData = () => {
  apiDownloadJson("/api/data.json") // Pass the JSON file path
    .then((data) => {
      setJsonData(data); // Store JSON data in state
    })
    .catch((err) => {
      setError("Failed to download JSON"); // Handle error
    });
};
*/
