// src/state/slices/LanguageSlice.js
// UI language selection. Persisted via the root persist whitelist in store.js.
// Kept free of imports from ../../i18n — i18n imports this slice's selector, so
// pulling LANGUAGES back in here would make the module cycle bite at init time.

import { createSlice } from "@reduxjs/toolkit";

const SUPPORTED = ["en", "de"];

// Start in the browser's language when we ship a catalog for it, else English.
const detectLanguage = () => {
  const tag = (navigator.language || "en").toLowerCase();
  return SUPPORTED.find((code) => tag.startsWith(code)) || "en";
};

const languageSlice = createSlice({
  name: "languageState",
  initialState: { language: detectLanguage() },
  reducers: {
    setLanguage: (state, action) => {
      state.language = action.payload;
    },
  },
});

export const { setLanguage } = languageSlice.actions;
export const getLanguage = (state) => state.languageState?.language || "en";
export default languageSlice.reducer;
