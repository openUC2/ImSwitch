import { api } from "./api";

export const downloadFile = async (files, hostname, port) => {
  if (files.length === 0) return;

  try {
    // Download each file individually
    for (const file of files) {
      // Remove leading slash if present to avoid double slashes
      const cleanPath = file.path.startsWith("/")
        ? file.path.slice(1)
        : file.path;
      const url = `${hostname}:${port}/imswitch/api/FileManager/download/${cleanPath}`;

      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", file.name);

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Small delay between downloads
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  } catch (error) {
    console.error("Download error:", error);
    return error;
  }
};
