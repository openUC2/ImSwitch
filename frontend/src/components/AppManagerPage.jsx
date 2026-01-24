// src/components/AppManagerPage.jsx
// Simple page version of App Manager (instead of modal)
// Fallback implementation for plugin-based navigation

import React from "react";
import AppManager from "./AppManager/AppManager.jsx";

/**
 * Page wrapper for the App Manager component
 * Used when accessed as a regular plugin/page
 */
const AppManagerPage = ({ onNavigateToApp }) => {
  return (
    <div style={{ height: "100%", width: "100%", padding: "16px" }}>
      <AppManager onNavigateToApp={onNavigateToApp} />
    </div>
  );
};

export default AppManagerPage;
