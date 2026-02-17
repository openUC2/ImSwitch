import { api } from "./api";

export const downloadFile = async (files, hostname, port) => {
  if (files.length === 0) return { success: [], failed: [] };

  const result = { success: [], failed: [] };

  for (const file of files) {
    try {
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

      result.success.push(file.name);

      // Small delay between downloads
      await new Promise((resolve) => setTimeout(resolve, 100));
    } catch (error) {
      console.error(`Download failed for file "${file.name}":`, error);
      result.failed.push({ name: file.name, error: error.message });
    }
  }

  // Log summary
  if (result.failed.length > 0) {
    console.warn(
      `Download completed with errors: ${result.success.length} succeeded, ${result.failed.length} failed`,
      result.failed,
    );
  } else if (result.success.length > 0) {
    console.log(`All ${result.success.length} file(s) downloaded successfully`);
  }

  return result;
};
