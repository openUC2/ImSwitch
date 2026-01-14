// src/context/MCTContext.js
import React, { createContext, useState } from 'react';

// Create the context
export const MCTContext = createContext();

// Create the provider component
export const MCTProvider = ({ children }) => {
  const [timePeriod, setTimePeriod] = useState('5');
  const [numMeasurements, setNumMeasurements] = useState('1');
  const [zMin, setZMin] = useState('-100');
  const [zMax, setZMax] = useState('100');
  const [zSteps, setZSteps] = useState('0');
  const [zStackEnabled, setZStackEnabled] = useState(false);
  const [xMin, setXMin] = useState('-1000');
  const [xMax, setXMax] = useState('1000');
  const [xSteps, setXSteps] = useState('0');
  const [xStackEnabled, setXStackEnabled] = useState(false);
  const [yMin, setYMin] = useState('-1000');
  const [yMax, setYMax] = useState('1000');
  const [ySteps, setYSteps] = useState('0');
  const [yStackEnabled, setYStackEnabled] = useState(false);
  const [intensityLaser1, setIntensityLaser1] = useState(0);
  const [intensityLaser2, setIntensityLaser2] = useState(0);
  const [intensityLED, setIntensityLED] = useState(0);
  const [fileName, setFileName] = useState('MCT');
  const [isRunning, setIsRunning] = useState(false);

  return (
    <MCTContext.Provider
      value={{
        timePeriod,
        setTimePeriod,
        numMeasurements,
        setNumMeasurements,
        zMin,
        setZMin,
        zMax,
        setZMax,
        zSteps,
        setZSteps,
        zStackEnabled,
        setZStackEnabled,
        xMin,
        setXMin,
        xMax,
        setXMax,
        xSteps,
        setXSteps,
        xStackEnabled,
        setXStackEnabled,
        yMin,
        setYMin,
        yMax,
        setYMax,
        ySteps,
        setYSteps,
        yStackEnabled,
        setYStackEnabled,
        intensityLaser1,
        setIntensityLaser1,
        intensityLaser2,
        setIntensityLaser2,
        intensityLED,
        setIntensityLED,
        fileName,
        setFileName,
        isRunning,
        setIsRunning,
      }}
    >
      {children}
    </MCTContext.Provider>
  );
};
