// src/context/LiveWidgetContext.js
import React, { createContext, useState } from 'react';

// Create the context
export const LiveWidgetContext = createContext();

// Create a provider component
export const LiveWidgetProvider = ({ children }) => {
  // Define your global state here
  const [isStreamRunning, setStreamRunning] = useState(false);
  
  // Pass down the state and state-updating functions
  return (
    <LiveWidgetContext.Provider
      value={{
        isStreamRunning,
        setStreamRunning
            }}
    >
      {children}
    </LiveWidgetContext.Provider>
  );
};
