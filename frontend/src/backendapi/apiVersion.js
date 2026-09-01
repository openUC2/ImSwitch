// Read the backend version: GET /imswitch/api/version -> { version }.
// Not controller-scoped — the route is registered directly on the API router
// (ImSwitchServer.py), hence the bare path.
import createAxiosInstance from "./createAxiosInstance";

const apiVersion = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/version");
  return response.data;
};

export default apiVersion;
