// src/constants/sidebarColors.js
// Central color constants for sidebar and navigation
// Theme-aware colors that adapt to light/dark mode

/**
 * Get sidebar colors based on theme mode
 * @param {string} mode - 'light' or 'dark'
 * @returns {object} - Color configuration for sidebar elements
 */
export const getSidebarColors = (mode = "dark") => {
  const isDark = mode === "dark";

  return {
    essentials: isDark ? "#90caf9" : "#1565c0", // Light blue in dark, darker blue in light
    liveView: isDark ? "#1976d2" : "#0d47a1", // Medium blue in dark, deep blue in light
    apps: isDark ? "#43a047" : "#2e7d32", // Light green in dark, darker green in light
    fileManager: isDark ? "#1976d2" : "#0d47a1", // Same as liveView
    coding: isDark ? "#cd1616ff" : "#b71c1c", // Light red in dark, darker red in light
    system: isDark ? "#eba400cc" : "#e65100", // Light orange in dark, darker orange in light
    systemSettings: isDark ? "#90a4ae" : "#37474f", // Light gray in dark, dark gray in light
  };
};

// Legacy export for backward compatibility
// Note: This uses dark theme colors as default
const SIDEBAR_COLORS = getSidebarColors("dark");

export default SIDEBAR_COLORS;
