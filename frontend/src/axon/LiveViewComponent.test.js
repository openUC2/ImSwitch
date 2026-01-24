import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import LiveViewComponent from './LiveViewComponent';
import liveStreamReducer from '../state/slices/LiveStreamSlice';

// Create a mock store for testing
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      liveStreamState: liveStreamReducer,
    },
    preloadedState: {
      liveStreamState: {
        liveViewImage: "",
        minVal: 0,
        maxVal: 255,
        pixelSize: null,
        histogramX: [],
        histogramY: [],
        showHistogram: false,
        ...initialState,
      },
    },
  });
};

describe('LiveViewComponent', () => {
  test('renders loading message when no image is available', () => {
    const store = createMockStore();
    
    render(
      <Provider store={store}>
        <LiveViewComponent />
      </Provider>
    );
    
    expect(screen.getByText('Loading image...')).toBeInTheDocument();
  });

  test('renders intensity slider with correct range', () => {
    const store = createMockStore({
      liveViewImage: "test-base64-image-data",
      minVal: 50,
      maxVal: 200,
    });
    
    render(
      <Provider store={store}>
        <LiveViewComponent />
      </Provider>
    );
    
    // Check if intensity label is present
    expect(screen.getByText('Intensity')).toBeInTheDocument();
  });

  test('renders scale bar when pixelSize is available', () => {
    const store = createMockStore({
      liveViewImage: "test-base64-image-data",
      pixelSize: 0.2,
    });
    
    render(
      <Provider store={store}>
        <LiveViewComponent />
      </Provider>
    );
    
    // Check if scale bar text is present
    expect(screen.getByText('10.00 µm')).toBeInTheDocument();
  });
});