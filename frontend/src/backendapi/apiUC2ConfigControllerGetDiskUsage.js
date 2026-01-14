import createAxiosInstance from "./createAxiosInstance";
import { formatDiskUsage } from "../utils/formatUtils";

/**
 * Fetches disk usage from UC2ConfigController and returns both raw and formatted values
 * @returns {Promise<{raw: number, formatted: string, percent: number}>}
 * Example: {raw: 0.687, formatted: "68.7%", percent: 68.7}
 */
export default async function apiUC2ConfigControllerGetDiskUsage() {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/UC2ConfigController/getDiskUsage"
    );

    const rawData = response.data;

    // Handle different response formats
    let rawValue = 0;
    if (typeof rawData === "number") {
      rawValue = rawData;
    } else if (typeof rawData === "string") {
      // If it's a string like "0.687" or "68.7%"
      const cleaned = rawData.replace("%", "");
      const parsed = parseFloat(cleaned);
      // Check if it's already a percentage (>=1) or a decimal (<1)
      rawValue = parsed >= 1 ? parsed / 100 : parsed;
    } else if (rawData && rawData.disk_usage !== undefined) {
      rawValue =
        typeof rawData.disk_usage === "number"
          ? rawData.disk_usage
          : parseFloat(String(rawData.disk_usage).replace("%", ""));
      // Normalize if already percentage
      if (rawValue >= 1) rawValue = rawValue / 100;
    }

    // Return structured data
    return {
      raw: rawValue, // 0.687
      formatted: formatDiskUsage(rawValue), // "68.7%"
      percent: rawValue * 100, // 68.7
    };
  } catch (error) {
    console.error("Error fetching disk usage:", error);
    throw error;
  }
}
