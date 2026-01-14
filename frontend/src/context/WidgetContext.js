import React, { useState, useContext } from "react";

const WidgetContext = React.createContext();

export function useWidgetContext() {
  return useContext(WidgetContext);
}

export function WidgetContextProvider({ children }) {
  const [sliderValue, setSliderValue] = useState(0);
  const [generic, setGeneric] = useState({ init: "init" });

  const handleGeneric = (e) => {
    const key = e[0];
    const value = e[1];
    setGeneric((prevState) => ({
      ...prevState,
      [key]: value,
    }));
  };

  return (
    <WidgetContext.Provider
      value={{
        sliderValue,
        setSliderValue,
        generic,
        handleGeneric,
      }}
    >
      {children}
    </WidgetContext.Provider>
  );
}
