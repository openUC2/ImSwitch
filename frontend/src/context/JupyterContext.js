import React, { createContext, useContext, useState } from "react";

const JupyterContext = createContext();

export const JupyterProvider = ({ children }) => {
  const [jupyterUrl, setJupyterUrl] = useState(null);
  const [notebookData, setNotebookData] = useState(null);

  return (
    <JupyterContext.Provider value={{ jupyterUrl, setJupyterUrl, notebookData, setNotebookData }}>
      {children}
    </JupyterContext.Provider>
  );
};

export const useJupyter = () => useContext(JupyterContext);
