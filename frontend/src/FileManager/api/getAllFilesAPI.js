import { api } from "./api";

export const getAllFilesAPI = async () => {
  try {
    const response = await api.get("/");
    console.log("response", response);
    console.log("fetch url: ", `${api.defaults.baseURL}`);
    return response;
  } catch (error) {
    return error;
  }
};
