import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import FocusLockController from '../components/FocusLockController';
import focusLockReducer from '../state/slices/FocusLockSlice';

// Mock Chart.js
jest.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="focus-chart">Chart</div>
}));

// Mock WebSocket context
jest.mock('../context/WebSocketContext', () => ({
  useWebSocket: () => null
}));

// Create a test store
const createTestStore = () => {
  return configureStore({
    reducer: {
      focusLockState: focusLockReducer,
    },
  });
};

describe('FocusLockController', () => {
  test('renders without crashing', () => {
    const store = createTestStore();
    
    render(
      <Provider store={store}>
        <FocusLockController hostIP="http://localhost" hostPort="80" />
      </Provider>
    );
    
    expect(screen.getByText('Focus Lock Controller (Astigmatism-based)')).toBeInTheDocument();
    expect(screen.getByText('Focus Lock Status')).toBeInTheDocument();
    expect(screen.getByText('PI Controller Parameters')).toBeInTheDocument();
    expect(screen.getByText('Astigmatism Parameters')).toBeInTheDocument();
  });

  test('displays initial state correctly', () => {
    const store = createTestStore();
    
    render(
      <Provider store={store}>
        <FocusLockController hostIP="http://localhost" hostPort="80" />
      </Provider>
    );
    
    expect(screen.getByText(/Focus Lock Disabled/)).toBeInTheDocument();
    expect(screen.getByText(/Current Focus Value: 0.000/)).toBeInTheDocument();
  });
});
