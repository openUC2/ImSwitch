// OmeZarrTileStreamSlice for handling incoming Zarr chunk updates

import { createSlice, createAction } from "@reduxjs/toolkit";

// We receive a dict like:
// { event: "zarr_chunk", path: rel_chunk, zarr: "/files/.../FastStageScan.ome.zarr" }

const initialZarrSlice = {
  zarrUrl: "",  // base URL for the OmeZarr dataset, e.g., "/files/.../FastStageScan.ome.zarr"
  version: 0,   // version to force reloading of data
  tilePath: "", // path to the current tile
};

/*
we get a dictionary with the following structure:
    sigZarrDict = {
        "event": "zarr_chunk",
        "path": rel_chunk,
        "zarr": str(self.getOmeZarrUrl())  # e.g. /files/…/FastStageScan.ome.zarr
    }
    This action is triggered when a new Zarr chunk arrives
*/

export const tileArrived = createAction("tile/chunk");

const omeZarrSlice = createSlice({
  name: "tile",
  initialState: initialZarrSlice,
  reducers: {
    // sets the base URL for the OmeZarr dataset
    setZarrUrl: (state, action) => {
        console.log("setZarrUrl", action.payload);
        state.zarrUrl = action.payload.zarr;        // URL to the file e.g. "https://localhost:8000/files/…/FastStageScan.ome.zarr"
        state.tilePath = action.payload.path || ""; // tile path to the current chunk, e.g. "0/1"
    },
  },
  extraReducers: (builder) => {
    // increments the version to force reloading the data
    builder.addCase(tileArrived, (state) => {
      state.version += 1; // bump version -> cache-bust
    });
  },
});

// Selector helper
export const getOmeZarrState = (state) => state.omeZarrState;


// export actions and reducer
export const { setZarrUrl } = omeZarrSlice.actions;
export default omeZarrSlice.reducer;