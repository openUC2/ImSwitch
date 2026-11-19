/**
 * Arbitrary Points tab: the Sequence / Instant position switch.
 *
 * Sequence mode builds a point list from clicks on the camera view. Instant mode
 * sends the scanner to the clicked position at once (a one-point scan that holds
 * until Stop) and swaps the points list for an instant-position panel.
 * The live view is replaced by a stub that "clicks" the middle of a 640x480 image.
 */
/* act() flushes the fetch promise a click starts; waitFor alone would race the state update. */
/* eslint-disable testing-library/no-unnecessary-act */
import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';

import galvoReducer, { setScannerNames } from '../state/slices/GalvoScannerSlice';
import liveStreamReducer from '../state/slices/LiveStreamSlice';

jest.mock('../axon/LiveViewControlWrapper', () => ({ onClick, overlayContent }) => (
  <div>
    <button data-testid="cam-click" onClick={() => onClick(320, 240, 640, 480)}>cam</button>
    <button data-testid="cam-click-corner" onClick={() => onClick(640, 480, 640, 480)}>corner</button>
    {overlayContent}
  </div>
));
jest.mock('../components/GalvoAffineCalibrationWizard', () => () => <div />);

const makeStore = () =>
  configureStore({
    reducer: {
      galvoScannerState: galvoReducer,
      liveStreamState: liveStreamReducer,
      connectionSettingsState: () => ({ ip: 'http://10.0.0.5', apiPort: 8001 }),
    },
    middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
  });

const calls = () => global.fetch.mock.calls.map(([u]) => String(u));
const callsTo = (name) => calls().filter((u) => u.includes(`/${name}?`) || u.endsWith(`/${name}`));
const sentPoints = (url) => JSON.parse(new URL(url).searchParams.get('points'));

let routes;
const mount = async () => {
  const GalvoArbitraryPointsTab = require('../components/GalvoArbitraryPointsTab').default;
  const store = makeStore();
  store.dispatch(setScannerNames(['galvo']));
  await act(async () => {
    render(
      <Provider store={store}>
        <GalvoArbitraryPointsTab />
      </Provider>
    );
  });
  return store;
};

beforeEach(() => {
  routes = {
    getAffineTransform: { affine_transform: { a11: 1, a12: 0, tx: 0, a21: 0, a22: 1, ty: 0 } },
    startArbitraryScan: { status: 'points_set' },
    stopArbitraryScan: { status: 'stopped' },
  };
  global.fetch = jest.fn((url) => {
    const name = String(url).split('/').pop().split('?')[0];
    return Promise.resolve({ json: () => Promise.resolve(routes[name] ?? {}) });
  });
});

describe('Arbitrary points tab: sequence / instant switch', () => {
  test('sequence mode (default): a click adds a point and moves nothing', async () => {
    const store = await mount();
    fireEvent.click(screen.getByTestId('cam-click'));
    expect(store.getState().galvoScannerState.arbitraryPoints.points).toHaveLength(1);
    expect(callsTo('startArbitraryScan')).toHaveLength(0);
    expect(screen.getByText(/Click to add points \(1\/256\)/)).toBeInTheDocument();
  });

  test('instant mode: the points list is replaced and a click sends one point at once', async () => {
    const store = await mount();
    expect(screen.getByText(/^Points \(/)).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    expect(screen.queryByText(/^Points \(/)).toBeNull();
    expect(screen.getByTestId('galvo-instant-readout')).toHaveTextContent(/Click the camera view/);

    await act(async () => { fireEvent.click(screen.getByTestId('cam-click')); });

    const [url] = callsTo('startArbitraryScan');
    // middle of the image -> middle of the 12-bit DAC range; already DAC space, so no affine on the backend
    expect(sentPoints(url)).toEqual([{ x: 2048, y: 2048, dwell_us: 500, laser_intensity: 128 }]);
    expect(new URL(url).searchParams.get('apply_affine')).toBe('false');
    expect(new URL(url).searchParams.get('laser_trigger')).toBe('AUTO');

    await waitFor(() =>
      expect(screen.getByTestId('galvo-instant-readout')).toHaveTextContent('(2048, 2048)'));
    expect(screen.getByTestId('galvo-instant-marker')).toBeInTheDocument();
    expect(screen.getByText('Holding point')).toBeInTheDocument();
    // the sequence list is untouched
    expect(store.getState().galvoScannerState.arbitraryPoints.points).toHaveLength(0);
  });

  test('instant mode honours the affine calibration', async () => {
    routes.getAffineTransform = { affine_transform: { a11: 2, a12: 0, tx: 100, a21: 0, a22: 2, ty: 50 } };
    await mount();
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    await act(async () => { fireEvent.click(screen.getByTestId('cam-click')); });
    // galvo = A * sensor + b, with sensor (320, 240)
    expect(sentPoints(callsTo('startArbitraryScan')[0])[0]).toMatchObject({ x: 740, y: 530 });
  });

  test('the laser trigger and dwell settings travel with the point', async () => {
    const store = await mount();
    const { setLaserTrigger, setDefaultDwellUs } = require('../state/slices/GalvoScannerSlice');
    act(() => {
      store.dispatch(setLaserTrigger('LOW'));
      store.dispatch(setDefaultDwellUs(2500));
    });
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    await act(async () => { fireEvent.click(screen.getByTestId('cam-click-corner')); });
    const [url] = callsTo('startArbitraryScan');
    expect(new URL(url).searchParams.get('laser_trigger')).toBe('LOW');
    expect(sentPoints(url)[0]).toMatchObject({ x: 4095, y: 4095, dwell_us: 2500 });
  });

  test('a refused move shows the error and does not mark the beam as held', async () => {
    routes.startArbitraryScan = { error: 'Points not acknowledged: No response received' };
    const store = await mount();
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    await act(async () => { fireEvent.click(screen.getByTestId('cam-click')); });

    await waitFor(() =>
      expect(store.getState().galvoScannerState.error).toMatch(/Points not acknowledged/));
    expect(screen.queryByTestId('galvo-instant-marker')).toBeNull();
    expect(store.getState().galvoScannerState.arbitraryPoints.running).toBe(false);
  });

  test('typed coordinates + Go move the scanner', async () => {
    await mount();
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    fireEvent.change(screen.getByTestId('galvo-instant-x'), { target: { value: '1000' } });
    fireEvent.change(screen.getByTestId('galvo-instant-y'), { target: { value: '3000' } });
    await act(async () => { fireEvent.click(screen.getByTestId('galvo-instant-go')); });
    expect(sentPoints(callsTo('startArbitraryScan')[0])[0]).toMatchObject({ x: 1000, y: 3000 });
  });

  test('Stop and park stops the scan and clears the marker', async () => {
    const store = await mount();
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    await act(async () => { fireEvent.click(screen.getByTestId('cam-click')); });
    await screen.findByTestId('galvo-instant-marker');

    await act(async () => { fireEvent.click(screen.getByTestId('galvo-instant-stop')); });
    expect(callsTo('stopArbitraryScan')).toHaveLength(1);
    expect(screen.queryByTestId('galvo-instant-marker')).toBeNull();
    expect(store.getState().galvoScannerState.arbitraryPoints.running).toBe(false);
  });

  test('switching mode while a point is held stops it', async () => {
    const store = await mount();
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    await act(async () => { fireEvent.click(screen.getByTestId('cam-click')); });
    await screen.findByTestId('galvo-instant-marker');

    await act(async () => { fireEvent.click(screen.getByTestId('galvo-mode-sequence')); });
    expect(callsTo('stopArbitraryScan')).toHaveLength(1);
    expect(screen.queryByTestId('galvo-instant-marker')).toBeNull();
    expect(screen.getByText(/^Points \(/)).toBeInTheDocument();
    expect(store.getState().galvoScannerState.arbitraryPoints.running).toBe(false);
  });

  test('switching mode with nothing running sends no stop', async () => {
    await mount();
    fireEvent.click(screen.getByTestId('galvo-mode-instant'));
    fireEvent.click(screen.getByTestId('galvo-mode-sequence'));
    expect(callsTo('stopArbitraryScan')).toHaveLength(0);
  });
});
