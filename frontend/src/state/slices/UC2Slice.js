import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // Tab navigation
  tabIndex: 0,

  // Setup management
  availableSetups: [],
  selectedSetup: "",
  isDialogOpen: false,
  restartSoftware: false,

  // Serial communication
  serialPayload: "",
  serialLog: [],

  // Connection status (IMPORTANT: Two different types!)
  backendConnected: false, // Backend API reachable (enables UI functions)
  uc2Connected: false, // UC2 hardware connected to backend

  // Config file editing
  selectedFileForEdit: "",
  editorJson: null,
  editorJsonText: "",
  useAceEditor: false,

  // Current active configuration
  currentActiveFilename: null,
  isLoadingCurrentFilename: false,

  // File operations
  newFileName: "",
  setAsCurrentConfig: true,
  restartAfterSave: false,
  overwriteFile: false,

  // Loading states
  isLoadingFile: false,
  isSavingFile: false,
  isRestarting: false,
  isValidatingConfig: false,

  // Validation and preview
  validationResult: null,
  configPreview: null,
  showPreviewDialog: false,
};

const uc2Slice = createSlice({
  name: "uc2",
  initialState,
  reducers: {
    // Tab navigation
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Setup management
    setAvailableSetups: (state, action) => {
      state.availableSetups = action.payload;
    },
    setSelectedSetup: (state, action) => {
      state.selectedSetup = action.payload;
    },
    setIsDialogOpen: (state, action) => {
      state.isDialogOpen = action.payload;
    },
    setRestartSoftware: (state, action) => {
      state.restartSoftware = action.payload;
    },

    // Serial communication
    setSerialPayload: (state, action) => {
      state.serialPayload = action.payload;
    },
    addSerialLogEntry: (state, action) => {
      state.serialLog.push(action.payload);
    },
    clearSerialLog: (state) => {
      state.serialLog = [];
    },

    // Connection status setters
    setBackendConnected: (state, action) => {
      state.backendConnected = action.payload;
    },
    setUc2Connected: (state, action) => {
      state.uc2Connected = action.payload;
    },

    // Config file editing
    setSelectedFileForEdit: (state, action) => {
      state.selectedFileForEdit = action.payload;
    },
    setEditorJson: (state, action) => {
      state.editorJson = action.payload;
    },
    setEditorJsonText: (state, action) => {
      state.editorJsonText = action.payload;
    },
    setUseAceEditor: (state, action) => {
      state.useAceEditor = action.payload;
    },

    // Current active configuration
    setCurrentActiveFilename: (state, action) => {
      state.currentActiveFilename = action.payload;
    },
    setIsLoadingCurrentFilename: (state, action) => {
      state.isLoadingCurrentFilename = action.payload;
    },

    // File operations
    setNewFileName: (state, action) => {
      state.newFileName = action.payload;
    },
    setSetAsCurrentConfig: (state, action) => {
      state.setAsCurrentConfig = action.payload;
    },
    setRestartAfterSave: (state, action) => {
      state.restartAfterSave = action.payload;
    },
    setOverwriteFile: (state, action) => {
      state.overwriteFile = action.payload;
    },
    setSerialLog: (state, action) => {
      state.serialLog = action.payload;
    },

    // Loading states
    setIsLoadingFile: (state, action) => {
      state.isLoadingFile = action.payload;
    },
    setIsSavingFile: (state, action) => {
      state.isSavingFile = action.payload;
    },
    setIsRestarting: (state, action) => {
      state.isRestarting = action.payload;
    },
    setIsValidatingConfig: (state, action) => {
      state.isValidatingConfig = action.payload;
    },

    // Validation and preview
    setValidationResult: (state, action) => {
      state.validationResult = action.payload;
    },
    setConfigPreview: (state, action) => {
      state.configPreview = action.payload;
    },
    setShowPreviewDialog: (state, action) => {
      state.showPreviewDialog = action.payload;
    },

    // Reset actions
    resetFileOperations: (state) => {
      state.newFileName = "";
      state.setAsCurrentConfig = true;
      state.restartAfterSave = false;
      state.overwriteFile = false;
      state.selectedFileForEdit = "";
      state.editorJson = null;
      state.editorJsonText = "";
      state.useAceEditor = false;
      state.validationResult = null;
      state.configPreview = null;
      state.showPreviewDialog = false;
      state.isLoadingFile = false;
      state.isSavingFile = false;
      state.isValidatingConfig = false;
    },
  },
});

// Export actions
export const {
  setTabIndex,
  setAvailableSetups,
  setSelectedSetup,
  setIsDialogOpen,
  setRestartSoftware,
  setSerialPayload,
  addSerialLogEntry,
  clearSerialLog,
  setBackendConnected,
  setUc2Connected,
  setSelectedFileForEdit,
  setEditorJson,
  setEditorJsonText,
  setUseAceEditor,
  setCurrentActiveFilename,
  setIsLoadingCurrentFilename,
  setNewFileName,
  setSetAsCurrentConfig,
  setRestartAfterSave,
  setOverwriteFile,
  resetFileOperations,
  setSerialLog,
  setIsLoadingFile,
  setIsSavingFile,
  setIsRestarting,
  setIsValidatingConfig,
  setValidationResult,
  setConfigPreview,
  setShowPreviewDialog,
} = uc2Slice.actions;

// Export selector
export const getUc2State = (state) => state.uc2State;

// Export reducer
export default uc2Slice.reducer;
