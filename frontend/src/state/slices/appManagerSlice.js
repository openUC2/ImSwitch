// src/state/slices/appManagerSlice.js
// Redux slice for managing enabled/disabled applications
// Handles user preferences for which apps appear in the navigation drawer

import { createSlice } from "@reduxjs/toolkit";
import { APP_REGISTRY, APP_CATEGORIES } from "../../constants/appRegistry";

// Initial state - essentials are always enabled, others start disabled
const getInitialEnabledApps = () => {
  const enabledApps = [];

  // Add all essential apps
  Object.values(APP_REGISTRY).forEach((app) => {
    if (app.essential) {
      enabledApps.push(app.id);
    }
  });

  return enabledApps;
};

const initialState = {
  // List of enabled app IDs
  enabledApps: getInitialEnabledApps(),

  // Search and filter state for app manager UI
  searchQuery: "",
  selectedCategory: "all", // 'all' or specific category

  // Statistics
  stats: {
    totalApps: Object.keys(APP_REGISTRY).length,
    enabledCount: getInitialEnabledApps().length,
    lastUpdated: Date.now(),
  },
};

const appManagerSlice = createSlice({
  name: "appManager",
  initialState,
  reducers: {
    /**
     * Enable a specific app
     */
    enableApp: (state, action) => {
      const appId = action.payload;
      if (!state.enabledApps.includes(appId)) {
        state.enabledApps.push(appId);
        state.stats.enabledCount = state.enabledApps.length;
        state.stats.lastUpdated = Date.now();
      }
    },

    /**
     * Disable a specific app (only if not essential)
     */
    disableApp: (state, action) => {
      const appId = action.payload;
      const app = APP_REGISTRY[appId];

      // Don't allow disabling essential apps
      if (app && !app.essential) {
        state.enabledApps = state.enabledApps.filter((id) => id !== appId);
        state.stats.enabledCount = state.enabledApps.length;
        state.stats.lastUpdated = Date.now();
      }
    },

    /**
     * Toggle app enabled state
     */
    toggleApp: (state, action) => {
      const appId = action.payload;
      const app = APP_REGISTRY[appId];

      if (!app) return;

      // Don't allow toggling essential apps
      if (app.essential) return;

      if (state.enabledApps.includes(appId)) {
        state.enabledApps = state.enabledApps.filter((id) => id !== appId);
      } else {
        state.enabledApps.push(appId);
      }

      state.stats.enabledCount = state.enabledApps.length;
      state.stats.lastUpdated = Date.now();
    },

    /**
     * Enable all apps in a category
     */
    enableCategory: (state, action) => {
      const category = action.payload;

      Object.values(APP_REGISTRY).forEach((app) => {
        if (app.category === category && !state.enabledApps.includes(app.id)) {
          state.enabledApps.push(app.id);
        }
      });

      state.stats.enabledCount = state.enabledApps.length;
      state.stats.lastUpdated = Date.now();
    },

    /**
     * Disable all apps in a category (except essentials)
     */
    disableCategory: (state, action) => {
      const category = action.payload;

      // Don't disable essentials category
      if (category === APP_CATEGORIES.ESSENTIALS) return;

      Object.values(APP_REGISTRY).forEach((app) => {
        if (app.category === category && !app.essential) {
          state.enabledApps = state.enabledApps.filter((id) => id !== app.id);
        }
      });

      state.stats.enabledCount = state.enabledApps.length;
      state.stats.lastUpdated = Date.now();
    },

    /**
     * Reset to default state (only essentials enabled)
     */
    resetToDefaults: (state) => {
      state.enabledApps = getInitialEnabledApps();
      state.stats.enabledCount = state.enabledApps.length;
      state.stats.lastUpdated = Date.now();
    },

    /**
     * Bulk update enabled apps
     */
    setEnabledApps: (state, action) => {
      const appIds = action.payload;

      // Always include essential apps
      const essentialAppIds = Object.values(APP_REGISTRY)
        .filter((app) => app.essential)
        .map((app) => app.id);

      // Combine essential apps with provided list
      state.enabledApps = [...new Set([...essentialAppIds, ...appIds])];
      state.stats.enabledCount = state.enabledApps.length;
      state.stats.lastUpdated = Date.now();
    },

    /**
     * Update search query for app manager UI
     */
    setSearchQuery: (state, action) => {
      state.searchQuery = action.payload;
    },

    /**
     * Set selected category filter
     */
    setSelectedCategory: (state, action) => {
      state.selectedCategory = action.payload;
    },

    /**
     * Clear search and filters
     */
    clearFilters: (state) => {
      state.searchQuery = "";
      state.selectedCategory = "all";
    },
  },
});

// Export actions
export const {
  enableApp,
  disableApp,
  toggleApp,
  enableCategory,
  disableCategory,
  resetToDefaults,
  setEnabledApps,
  setSearchQuery,
  setSelectedCategory,
  clearFilters,
} = appManagerSlice.actions;

// Selectors
export const selectEnabledApps = (state) => state.appManager.enabledApps;
export const selectSearchQuery = (state) => state.appManager.searchQuery;
export const selectSelectedCategory = (state) =>
  state.appManager.selectedCategory;
export const selectAppStats = (state) => state.appManager.stats;

// Computed selectors
export const selectEnabledAppObjects = (state) => {
  const enabledIds = state.appManager.enabledApps;
  return Object.values(APP_REGISTRY).filter((app) =>
    enabledIds.includes(app.id)
  );
};

export const selectIsAppEnabled = (appId) => (state) => {
  return state.appManager.enabledApps.includes(appId);
};

export const selectEnabledAppsByCategory = (category) => (state) => {
  const enabledIds = state.appManager.enabledApps;
  return Object.values(APP_REGISTRY).filter(
    (app) => enabledIds.includes(app.id) && app.category === category
  );
};

export default appManagerSlice.reducer;
