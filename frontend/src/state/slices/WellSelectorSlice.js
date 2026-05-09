import { createSlice } from '@reduxjs/toolkit';

// Define the initial state for wellSelectorState
const initialWellSelectorState = {
  mode: 'single',          // string: could be 'default', 'select', 'edit', etc.
  overlapWidth: 0.0,          // float: size of the overlap between the rasters
  overlapHeight: 0.0,          // float: size of the overlap between the rasters
  showOverlap: false,          // bool: show overlap or not 
  showShape: false,          // bool: show shape or not
  cameraTargetPosition: { x: 0.0, y: 0.0 },          // float: x position of the camera target 
  savedBoundingBox: null,          // object: bounding box of the selected wells
  layoutOffsetX: 0.0,          // float: global X offset added to all well layouts in micrometers
  layoutOffsetY: 0.0,          // float: global Y offset added to all well layouts in micrometers
  // Area select settings
  areaSelectSnakescan: false,  // bool: enable snakescan pattern for area select
  areaSelectOverlap: 0.0,      // float: overlap percentage (0.0 = no overlap, 0.1 = 10% overlap)
  // Cup select settings
  cupSelectShape: 'circle',    // string: 'circle' or 'rectangle' for well select shape
  cupSelectOverlap: 0.0,       // float: overlap percentage (0.0 = no overlap, 0.1 = 10% overlap)
  // Labware (Opentrons-style) selection state
  labwareLoadName: null,       // string|null: currently selected labware loadName
  selectedWellIds: [],         // array<string>: currently selected well IDs (e.g. ["A1","B3"])
  conditionLabels: {},         // object<string,string>: well_id -> free-text condition label
};

// Create wellSelectorState slice
const wellSelectorSlice = createSlice({
  name: 'wellSelectorState',
  initialState: initialWellSelectorState,
  reducers: {
    setMode: (state, action) => {
        console.log("setMode");
        state.mode = action.payload;
    },
    setOverlapWidth: (state, action) => {
        console.log("setOverlapWidth");
        state.overlapWidth = action.payload;
        if(isNaN(state.overlapWidth)){
            state.overlapWidth = 0;
        }
    },
    setOverlapHeight: (state, action) => {
        console.log("setOverlapHeight");
        state.overlapHeight = action.payload;
    },
    setMouseDownFlag: (state, action) => {
        console.log("setMouseDownFlag");
        state.mouseDownFlag = action.payload;
    },
    setShowOverlap: (state, action) => {
        console.log("setShowOverlap");
        state.showOverlap = action.payload;	
    },
    setShowOverlap: (state, action) => {
        console.log("setShowOverlap");
        state.showOverlap = action.payload;	
    },
    setShowShape: (state, action) => {
        console.log("setShowShape");
        state.showShape = action.payload;	
    },
    setCameraTargetPosition: (state, action) => {
        console.log("setCameraTargetPosition");
        state.cameraTargetPosition = action.payload;	
    },
    resetState: (state) => {
        console.log("resetState");
        return { ...initialWellSelectorState }; // Reset to initial state
    },
    setSavedBoundingBox: (state, action) => {
      state.savedBoundingBox = action.payload;
    },
    setLayoutOffsetX: (state, action) => {
      console.log("setLayoutOffsetX");
      state.layoutOffsetX = action.payload;
      if(isNaN(state.layoutOffsetX)){
        state.layoutOffsetX = 0;
      }
    },
    setLayoutOffsetY: (state, action) => {
      console.log("setLayoutOffsetY");
      state.layoutOffsetY = action.payload;
      if(isNaN(state.layoutOffsetY)){
        state.layoutOffsetY = 0;
      }
    },
    setAreaSelectSnakescan: (state, action) => {
      console.log("setAreaSelectSnakescan");
      state.areaSelectSnakescan = action.payload;
    },
    setAreaSelectOverlap: (state, action) => {
      console.log("setAreaSelectOverlap");
      state.areaSelectOverlap = action.payload;
      if(isNaN(state.areaSelectOverlap)){
        state.areaSelectOverlap = 0;
      }
    },
    setCupSelectShape: (state, action) => {
      console.log("setCupSelectShape");
      state.cupSelectShape = action.payload;
    },
    setCupSelectOverlap: (state, action) => {
      console.log("setCupSelectOverlap");
      state.cupSelectOverlap = action.payload;
      if(isNaN(state.cupSelectOverlap)){
        state.cupSelectOverlap = 0;
      }
    },
    // ── Labware selection actions ──────────────────────────────────────
    setLabwareLoadName: (state, action) => {
      state.labwareLoadName = action.payload || null;
    },
    setSelectedWellIds: (state, action) => {
      state.selectedWellIds = Array.isArray(action.payload) ? action.payload : [];
    },
    toggleSelectedWellId: (state, action) => {
      const id = action.payload;
      if (!id) return;
      const i = state.selectedWellIds.indexOf(id);
      if (i >= 0) state.selectedWellIds.splice(i, 1);
      else state.selectedWellIds.push(id);
    },
    clearSelectedWellIds: (state) => {
      state.selectedWellIds = [];
    },
    setConditionLabel: (state, action) => {
      const { wellId, label } = action.payload || {};
      if (!wellId) return;
      if (label == null || label === "") {
        delete state.conditionLabels[wellId];
      } else {
        state.conditionLabels[wellId] = String(label);
      }
    },
    setConditionLabels: (state, action) => {
      state.conditionLabels =
        action.payload && typeof action.payload === "object" ? { ...action.payload } : {};
    },
    clearConditionLabels: (state) => {
      state.conditionLabels = {};
    },
  },
});

// Export actions from  slice
export const { 
  setMode, 
  setOverlapWidth,
  setOverlapHeight,
  setShowOverlap,
  setShowShape,
  setCameraTargetPosition,
  resetState,
  setSavedBoundingBox,
  setLayoutOffsetX,
  setLayoutOffsetY,
  setAreaSelectSnakescan,
  setAreaSelectOverlap,
  setCupSelectShape,
  setCupSelectOverlap,
  setLabwareLoadName,
  setSelectedWellIds,
  toggleSelectedWellId,
  clearSelectedWellIds,
  setConditionLabel,
  setConditionLabels,
  clearConditionLabels,
} = wellSelectorSlice.actions;

// Selector helper
export const getWellSelectorState = (state) => state.wellSelectorState;

// Export reducer from  slice
export default wellSelectorSlice.reducer;
