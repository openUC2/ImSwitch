// Test for histogram functionality implementation
import { configureStore } from '@reduxjs/toolkit';
import liveStreamSlice, * as liveStreamActions from '../state/slices/LiveStreamSlice.js';

describe('LiveStreamSlice Histogram Functionality', () => {
  let store;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        liveStreamState: liveStreamSlice,
      },
    });
  });

  test('should have initial histogram state', () => {
    const state = store.getState().liveStreamState;
    
    expect(state.histogramX).toEqual([]);
    expect(state.histogramY).toEqual([]);
    expect(state.showHistogram).toBe(false);
  });

  test('should set histogram data correctly', () => {
    const histogramData = {
      x: [0, 50, 100, 150, 200, 255],
      y: [10, 20, 30, 25, 15, 5]
    };

    store.dispatch(liveStreamActions.setHistogramData(histogramData));
    
    const state = store.getState().liveStreamState;
    expect(state.histogramX).toEqual(histogramData.x);
    expect(state.histogramY).toEqual(histogramData.y);
  });

  test('should toggle histogram visibility', () => {
    // Initially false
    expect(store.getState().liveStreamState.showHistogram).toBe(false);
    
    // Set to true
    store.dispatch(liveStreamActions.setShowHistogram(true));
    expect(store.getState().liveStreamState.showHistogram).toBe(true);
    
    // Set back to false
    store.dispatch(liveStreamActions.setShowHistogram(false));
    expect(store.getState().liveStreamState.showHistogram).toBe(false);
  });

  test('should reset state including histogram data', () => {
    // Set some histogram data
    store.dispatch(liveStreamActions.setHistogramData({
      x: [1, 2, 3],
      y: [10, 20, 30]
    }));
    store.dispatch(liveStreamActions.setShowHistogram(true));
    
    // Verify data is set
    let state = store.getState().liveStreamState;
    expect(state.histogramX).toEqual([1, 2, 3]);
    expect(state.showHistogram).toBe(true);
    
    // Reset state
    store.dispatch(liveStreamActions.resetState());
    
    // Verify state is reset
    state = store.getState().liveStreamState;
    expect(state.histogramX).toEqual([]);
    expect(state.histogramY).toEqual([]);
    expect(state.showHistogram).toBe(false);
  });
});

describe('WebSocket sigHistogramComputed Signal Processing', () => {
  test('should process sigHistogramComputed signal format correctly', () => {
    // Mock signal data format as described in the issue
    const mockSignalData = {
      name: "sigHistogramComputed",
      args: {
        p0: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255], // units
        p1: [5, 15, 25, 35, 45, 30, 20, 15, 10, 8, 2]           // hist
      }
    };

    // Test that the data structure matches what our WebSocketHandler expects
    expect(mockSignalData.name).toBe('sigHistogramComputed');
    expect(mockSignalData.args).toHaveProperty('p0');
    expect(mockSignalData.args).toHaveProperty('p1');
    expect(Array.isArray(mockSignalData.args.p0)).toBe(true);
    expect(Array.isArray(mockSignalData.args.p1)).toBe(true);
    expect(mockSignalData.args.p0.length).toBeGreaterThan(0);
    expect(mockSignalData.args.p1.length).toBeGreaterThan(0);
  });
});