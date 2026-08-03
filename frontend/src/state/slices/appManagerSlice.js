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

/**
 * Look up an app by id across both the compile-time registry and the
 * runtime-discovered plugins. Without this, every reducer that guards on
 * `APP_REGISTRY[appId]` would silently ignore plugin ids.
 */
const findApp = (state, appId) =>
  APP_REGISTRY[appId] || (state.dynamicApps || []).find((a) => a.id === appId);

/**
 * redux-persist rehydrates this slice with autoMergeLevel1, which replaces the
 * whole `appManager` object with whatever was persisted. Anyone upgrading from
 * a build before plugins existed therefore gets a slice with these fields
 * missing, and the reducers below would throw on first use. Backfill them.
 */
const ensureArrays = (state) => {
  if (!Array.isArray(state.dynamicApps)) state.dynamicApps = [];
  if (!Array.isArray(state.seenDynamicApps)) state.seenDynamicApps = [];
  if (!Array.isArray(state.pluginErrors)) state.pluginErrors = [];
};

const initialState = {
  // List of enabled app IDs
  enabledApps: getInitialEnabledApps(),

  // Runtime-discovered plugins, mapped into the APP_REGISTRY shape by
  // makeRegistryEntryFromManifest(). Serializable only — icons are resolved by
  // name at render time. APP_REGISTRY itself is never mutated.
  dynamicApps: [],

  // Plugin ids we have already applied the "default enabled" rule to. Without
  // this, a plugin the user deliberately disabled would silently re-enable
  // itself on every page load.
  seenDynamicApps: [],

  // Plugins the backend could not load, and plugins gated off by
  // availableWidgets. Surfaced in the App Manager: a plugin that just fails to
  // appear is the failure mode we most need to avoid.
  pluginErrors: [],

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
      const app = findApp(state, appId);

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
      const app = findApp(state, appId);

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

    /**
     * Publish the plugins discovered at runtime.
     *
     * Payload: { apps, errors } where `apps` are entries produced by
     * makeRegistryEntryFromManifest() and `errors` is the backend's `errors`
     * array plus any disabled plugins.
     *
     * A newly seen plugin defaults to ENABLED. Rationale: the backend already
     * gated on availableWidgets, so a plugin that reached the frontend is one
     * the operator deliberately turned on for this instrument. The App Manager
     * toggle then controls visibility only, and — because ids are recorded in
     * seenDynamicApps — an explicit disable survives the next reload.
     */
    setDynamicApps: (state, action) => {
      const { apps = [], errors = [] } = action.payload || {};

      ensureArrays(state);
      state.dynamicApps = apps;
      state.pluginErrors = errors;

      apps.forEach((app) => {
        if (state.seenDynamicApps.includes(app.id)) return;
        state.seenDynamicApps.push(app.id);
        if (!state.enabledApps.includes(app.id)) {
          state.enabledApps.push(app.id);
        }
      });

      state.stats.enabledCount = state.enabledApps.length;
      state.stats.totalApps =
        Object.keys(APP_REGISTRY).length + state.dynamicApps.length;
      state.stats.lastUpdated = Date.now();
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
  setDynamicApps,
} = appManagerSlice.actions;

// Selectors
export const selectEnabledApps = (state) => state.appManager.enabledApps;

// Shared fallback. It MUST be a single module-level constant, not a fresh `[]`
// per call: these selectors are read through useSelector, which compares by
// reference. A new array each time makes every consumer re-render on every
// dispatched action and invalidates the useMemo chains built on top of them.
//
// The fallback is reached whenever redux-persist rehydrates an `appManager`
// saved before these fields existed — i.e. on every upgrade from a build
// without plugin support — so this is the normal path, not an edge case.
const EMPTY = Object.freeze([]);

// Runtime plugins. Returns the array straight out of state (stable reference)
// so useSelector does not re-render on every dispatch. Callers that need
// derived data — resolved icons, sorting — should useMemo over it.
export const selectDynamicApps = (state) => state.appManager.dynamicApps || EMPTY;
export const selectPluginErrors = (state) =>
  state.appManager.pluginErrors || EMPTY;
export const selectSearchQuery = (state) => state.appManager.searchQuery;
export const selectSelectedCategory = (state) =>
  state.appManager.selectedCategory;
export const selectAppStats = (state) => state.appManager.stats;

// Computed selectors
// Union of built-in apps and runtime plugins. Everything downstream of the
// App Manager should read this rather than APP_REGISTRY directly, otherwise
// plugins silently drop out of that view.
export const selectAllApps = (state) => [
  ...Object.values(APP_REGISTRY),
  ...(state.appManager.dynamicApps || []),
];

export const selectEnabledAppObjects = (state) => {
  const enabledIds = state.appManager.enabledApps;
  return selectAllApps(state).filter((app) => enabledIds.includes(app.id));
};

export const selectIsAppEnabled = (appId) => (state) => {
  return state.appManager.enabledApps.includes(appId);
};

export const selectEnabledAppsByCategory = (category) => (state) => {
  const enabledIds = state.appManager.enabledApps;
  return selectAllApps(state).filter(
    (app) => enabledIds.includes(app.id) && app.category === category
  );
};

export default appManagerSlice.reducer;
