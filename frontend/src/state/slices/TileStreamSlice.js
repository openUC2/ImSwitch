import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialTileStreamState = {
  tileViewImage: "",
};

// Create slice
const tileStreamSlice = createSlice({
  name: "tileStreamState",
  initialState: initialTileStreamState,
  reducers: {
    setTileViewImage: (state, action) => {
      state.tileViewImage = action.payload;
    },


    resetState: (state) => {
      console.log("resetState");
      return { ...initialTileStreamState }; // Reset to initial state
    },
  },
});


// Export actions from slice
export const {
  setTileViewImage,
  resetState,
} = tileStreamSlice.actions;

// Selector helper
export const getTileStreamState = (state) => state.tileStreamState;

// Export reducer from slice
export default tileStreamSlice.reducer;
