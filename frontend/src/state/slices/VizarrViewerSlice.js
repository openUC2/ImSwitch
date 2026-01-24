/**
 * VizarrViewerSlice.js
 * 
 * Redux slice for managing the offline Vizarr viewer state.
 * Handles the current file URL, viewer visibility, and refresh triggers.
 */

import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // Whether the viewer panel is visible
  isOpen: false,
  
  // Current OME-Zarr URL being viewed
  currentUrl: "",
  
  // File name for display
  fileName: "",
  
  // Key to force refresh/re-render
  refreshKey: 0,
  
  // History of recently viewed files
  recentFiles: [],
  
  // Maximum number of recent files to keep
  maxRecentFiles: 10,
};

const vizarrViewerSlice = createSlice({
  name: "vizarrViewer",
  initialState,
  reducers: {
    /**
     * Opens the viewer with a specific OME-Zarr URL
     */
    openViewer: (state, action) => {
      const { url, fileName = "" } = action.payload;
      state.isOpen = true;
      state.currentUrl = url;
      state.fileName = fileName || url.split("/").pop() || "OME-Zarr";
      
      // Add to recent files (avoid duplicates)
      const existingIndex = state.recentFiles.findIndex(f => f.url === url);
      if (existingIndex !== -1) {
        state.recentFiles.splice(existingIndex, 1);
      }
      state.recentFiles.unshift({ url, fileName: state.fileName, timestamp: Date.now() });
      
      // Keep only the most recent files
      if (state.recentFiles.length > state.maxRecentFiles) {
        state.recentFiles = state.recentFiles.slice(0, state.maxRecentFiles);
      }
      
      console.log("[VizarrViewerSlice] Opening viewer with URL:", url);
    },
    
    /**
     * Closes the viewer
     */
    closeViewer: (state) => {
      state.isOpen = false;
      console.log("[VizarrViewerSlice] Closing viewer");
    },
    
    /**
     * Sets the current URL without changing visibility
     */
    setCurrentUrl: (state, action) => {
      const { url, fileName = "" } = action.payload;
      state.currentUrl = url;
      state.fileName = fileName || url.split("/").pop() || "OME-Zarr";
      console.log("[VizarrViewerSlice] Setting URL:", url);
    },
    
    /**
     * Clears the current URL
     */
    clearUrl: (state) => {
      state.currentUrl = "";
      state.fileName = "";
    },
    
    /**
     * Forces a refresh/re-render of the viewer
     */
    refreshViewer: (state) => {
      state.refreshKey += 1;
      console.log("[VizarrViewerSlice] Refreshing viewer, key:", state.refreshKey);
    },
    
    /**
     * Clears the recent files history
     */
    clearRecentFiles: (state) => {
      state.recentFiles = [];
    },
    
    /**
     * Removes a specific file from recent files
     */
    removeFromRecentFiles: (state, action) => {
      const urlToRemove = action.payload;
      state.recentFiles = state.recentFiles.filter(f => f.url !== urlToRemove);
    },
  },
});

// Export actions
export const {
  openViewer,
  closeViewer,
  setCurrentUrl,
  clearUrl,
  refreshViewer,
  clearRecentFiles,
  removeFromRecentFiles,
} = vizarrViewerSlice.actions;

// Selector to get the entire state
export const getVizarrViewerState = (state) => state.vizarrViewerState;

// Individual selectors
export const selectIsOpen = (state) => state.vizarrViewerState.isOpen;
export const selectCurrentUrl = (state) => state.vizarrViewerState.currentUrl;
export const selectFileName = (state) => state.vizarrViewerState.fileName;
export const selectRecentFiles = (state) => state.vizarrViewerState.recentFiles;

// Export reducer
export default vizarrViewerSlice.reducer;
