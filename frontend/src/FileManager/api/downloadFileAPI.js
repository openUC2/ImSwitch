import { api } from "./api";

export const downloadFile = async (files, hostname, port) => {
  if (files.length === 0) return;

  try {
    // build url based on api
    const fileQuery = files.map((file) => `${file.path}`).join("&");
    const url = `${hostname}:${port}/FileManager/download/${fileQuery}`; //${import.meta.env.VITE_API_BASE_URL}/download?${fileQuery}`;

    const link = document.createElement("a");
    link.href = url;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (error) {
    return error;
  }
};
