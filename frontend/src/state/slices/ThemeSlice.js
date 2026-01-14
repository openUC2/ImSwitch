import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  isDarkMode: true,
};

const themeSlice = createSlice({
  name: "themeState",
  initialState,
  reducers: {
    toggleTheme: (state) => {
      state.isDarkMode = !state.isDarkMode;
    },
    setDarkMode: (state, action) => {
      state.isDarkMode = action.payload;
    },
  },
});

export const { toggleTheme, setDarkMode } = themeSlice.actions;
export const getThemeState = (state) => state.themeState;
export default themeSlice.reducer;