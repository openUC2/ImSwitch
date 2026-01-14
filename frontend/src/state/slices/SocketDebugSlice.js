// SocketDebugSlice.js
import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for socket debug messages
const initialSocketDebugState = {
  messages: [], // Array of all received messages
  maxMessages: 100, // Keep last 100 messages to prevent memory issues
  filterImageUpdates: true, // Filter out image update signals by default
};

// Create socket debug slice
const socketDebugSlice = createSlice({
  name: "socketDebugState",
  initialState: initialSocketDebugState,
  reducers: {
    addMessage: (state, action) => {
      // action.payload should be the parsed signal data
      const message = action.payload;
      
      // Filter out image updates if enabled
      if (state.filterImageUpdates && 
          (message.name === "sigImageUpdated" || message.name === "sigUpdateImage")) {
        return;
      }
      
      // Add timestamp if not present
      const messageWithTimestamp = {
        ...message,
        timestamp: message.timestamp || Date.now(),
      };
      
      // Add to beginning of array (newest first)
      state.messages.unshift(messageWithTimestamp);
      
      // Keep only last N messages
      if (state.messages.length > state.maxMessages) {
        state.messages = state.messages.slice(0, state.maxMessages);
      }
    },
    
    clearMessages: (state) => {
      state.messages = [];
    },
    
    setFilterImageUpdates: (state, action) => {
      state.filterImageUpdates = action.payload;
    },
    
    setMaxMessages: (state, action) => {
      state.maxMessages = action.payload;
      // Trim if necessary
      if (state.messages.length > action.payload) {
        state.messages = state.messages.slice(0, action.payload);
      }
    },
    
    resetState: (state) => {
      return { ...initialSocketDebugState };
    },
  },
});

// Export actions
export const {
  addMessage,
  clearMessages,
  setFilterImageUpdates,
  setMaxMessages,
  resetState,
} = socketDebugSlice.actions;

// Selector helper
export const getSocketDebugState = (state) => state.socketDebugState;

// Export reducer
export default socketDebugSlice.reducer;
