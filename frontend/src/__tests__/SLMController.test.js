/**
 * SLM / DMD app: mounts every tab against a mocked backend and checks that the
 * status reaches the header, a pattern request carries its parameters, and the
 * app registry entry is gated on the SLMController.
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import liveStreamReducer from '../state/slices/LiveStreamSlice';
import { APP_REGISTRY, isAppAvailableForControllers } from '../constants/appRegistry';

jest.mock('../axon/LiveViewControlWrapper', () => (props) => (
  <div data-testid="live-view">{props.overlayContent}</div>
));
jest.mock('../backendapi/apiSettingsControllerGetDetectorNames', () => () =>
  Promise.resolve(['WidefieldCamera']));

const status = {
  name: 'SLM', backend: 'virtual', displayIndex: 0, width: 1280, height: 800, connected: true,
  error: '', seq: 1, source: 'init', settleTimeMs: 10, calibrated: false, calibration: {},
  affine: { a11: 1, a12: 0, tx: 0, a21: 0, a22: 1, ty: 0 },
  patterns: ['blank', 'checkerboard', 'disk', 'grating', 'grid'],
  acquisition: { running: false }, calibration_run: { running: false }, play: { running: false },
};
// CRA runs Jest with resetMocks: implementations must be (re)set in beforeEach.
const mockApi = {
  apiSLMGetStatus: jest.fn(), apiSLMShowPattern: jest.fn(), apiSLMListSequences: jest.fn(),
  apiSLMGetCalibrationPoints: jest.fn(), slmPreviewUrl: jest.fn(), slmSequenceFrameUrl: jest.fn(),
  slmAcquiredFrameUrl: jest.fn(),
};
beforeEach(() => {
  mockApi.apiSLMGetStatus.mockResolvedValue(status);
  mockApi.apiSLMShowPattern.mockResolvedValue(status);
  mockApi.apiSLMListSequences.mockResolvedValue([{ name: 'demo', count: 3, size: [1280, 800], frames: ['a', 'b', 'c'] }]);
  mockApi.apiSLMGetCalibrationPoints.mockResolvedValue([[640, 400], [320, 200], [960, 200]]);
  mockApi.slmPreviewUrl.mockReturnValue('http://x/preview.png');
  mockApi.slmSequenceFrameUrl.mockReturnValue('http://x/frame.png');
  mockApi.slmAcquiredFrameUrl.mockReturnValue('http://x/acq.png');
});
jest.mock('../backendapi/apiSLMController', () => mockApi);

const SLMController = require('../components/SLMController.jsx').default;

const renderApp = () => {
  const store = configureStore({ reducer: { liveStreamState: liveStreamReducer } });
  return render(<Provider store={store}><SLMController /></Provider>);
};

test('header shows the display status and a pattern request carries its parameters', async () => {
  renderApp();
  await screen.findByText('1280 × 800 px');
  expect(screen.getByText('virtual[0]')).toBeInTheDocument();
  expect(screen.getByText('not calibrated')).toBeInTheDocument();
  fireEvent.click(screen.getByText('Show'));
  await waitFor(() => expect(mockApi.apiSLMShowPattern).toHaveBeenCalled());
  const [kind, params] = mockApi.apiSLMShowPattern.mock.calls[0];
  expect(kind).toBe('grating');
  expect(params).toMatchObject({ period: 32, angle: 0, duty: 0.5 });
  fireEvent.click(screen.getByText('All off'));
  await waitFor(() => expect(mockApi.apiSLMShowPattern).toHaveBeenLastCalledWith('blank', { value: 0 }));
});

test('every tab mounts against the mocked backend', async () => {
  renderApp();
  await screen.findByText('1280 × 800 px');
  fireEvent.click(screen.getByText('Draw on camera'));
  expect(screen.getByText('Project')).toBeInTheDocument();
  expect(screen.getByTestId('live-view')).toBeInTheDocument();
  fireEvent.click(screen.getByText('Calibrate'));
  expect(screen.getByText('Run')).toBeInTheDocument();
  fireEvent.click(screen.getByText('Start'));
  await waitFor(() => expect(mockApi.apiSLMGetCalibrationPoints).toHaveBeenCalledWith(5));
  await screen.findByText(/Click spot 1\/3/);
  fireEvent.click(screen.getByText('Sequence'));
  await screen.findByText('demo');
  expect(screen.getByText('Upload images…')).toBeInTheDocument();
});

test('the registry gates the app on SLMController', () => {
  expect(APP_REGISTRY.slm.pluginId).toBe('SLM');
  expect(isAppAvailableForControllers(APP_REGISTRY.slm, ['SLMController'])).toBe(true);
  expect(isAppAvailableForControllers(APP_REGISTRY.slm, ['LaserController'])).toBe(false);
});
