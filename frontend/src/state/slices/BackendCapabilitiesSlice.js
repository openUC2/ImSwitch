import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import apiGetAvailableControllers from "../../backendapi/apiGetAvailableControllers";

export const fetchAvailableControllers = createAsyncThunk(
  "backendCapabilities/fetchAvailableControllers",
  async (_, { rejectWithValue }) => {
    try {
      const controllers = await apiGetAvailableControllers();
      return Array.isArray(controllers) ? controllers : [];
    } catch (error) {
      return rejectWithValue(
        error?.message || "Failed to fetch available controllers",
      );
    }
  },
);

const initialState = {
  availableControllers: [],
  loading: false,
  error: null,
  lastUpdated: null,
};

const backendCapabilitiesSlice = createSlice({
  name: "backendCapabilities",
  initialState,
  reducers: {
    clearAvailableControllers: (state) => {
      state.availableControllers = [];
      state.loading = false;
      state.error = null;
      state.lastUpdated = null;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAvailableControllers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchAvailableControllers.fulfilled, (state, action) => {
        state.loading = false;
        state.availableControllers = action.payload;
        state.error = null;
        state.lastUpdated = Date.now();
      })
      .addCase(fetchAvailableControllers.rejected, (state, action) => {
        state.loading = false;
        state.availableControllers = [];
        state.error =
          action.payload ||
          action.error?.message ||
          "Failed to fetch available controllers";
      });
  },
});

export const { clearAvailableControllers } = backendCapabilitiesSlice.actions;

export const getBackendCapabilitiesState = (state) => state.backendCapabilities;
export const selectAvailableControllers = (state) =>
  state.backendCapabilities.availableControllers;
export const selectHasController = (controllerName) => (state) =>
  state.backendCapabilities.availableControllers.includes(controllerName);

export default backendCapabilitiesSlice.reducer;
