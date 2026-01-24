/**
 * Formats disk usage data from decimal to percentage string
 * @param {number|string} data - Usage data (0.686940516168836 or already formatted string)
 * @returns {string} Formatted percentage string (e.g., "68.7%")
 */
export const formatDiskUsage = (data) => {
  if (typeof data === "number") {
    // Convert decimal to percentage (0.686940516168836 -> 68.7%)
    return `${(data * 100).toFixed(1)}%`;
  }
  // Return as-is if already formatted
  return data;
};

/**
 * Formats memory usage in bytes to human-readable format
 * @param {number} bytes - Memory in bytes
 * @returns {string} Formatted memory string (e.g., "2.5 GB")
 */
export const formatMemoryUsage = (bytes) => {
  if (typeof bytes !== "number" || bytes === 0) return "0 B";
  
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};