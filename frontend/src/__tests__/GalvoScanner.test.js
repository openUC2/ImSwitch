/**
 * Galvo raster tab: window editing (width/centre), drag/resize in the scan
 * preview, camera backdrop placement through the affine, µm labels from the
 * camera<->scanner calibration, and the FLIM LABS web UI link.
 */
import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';

import galvoReducer, {
  setAxisWindow,
  setScanArea,
  moveScanArea,
  setCameraCalibration,
  setCalibrationComplete,
  cancelCalibration,
  setXRange,
  getScanPhysical,
} from '../state/slices/GalvoScannerSlice';
import GalvoScanPreview from '../components/GalvoScanPreview';
import { flimWebUiUrl } from '../components/FlimLabsPanel';

// The raster tab pulls in the Arbitrary Points tab (live view) and the FLIM
// panel; neither is under test here.
jest.mock('../components/GalvoArbitraryPointsTab', () => () => <div data-testid="arb-tab" />);
jest.mock('../components/FlimLabsPanel', () => {
  const actual = jest.requireActual('../components/FlimLabsPanel');
  const Mock = () => <div data-testid="flim-panel" />;
  Mock.flimWebUiUrl = actual.flimWebUiUrl;
  return { __esModule: true, default: Mock, flimWebUiUrl: actual.flimWebUiUrl };
});
jest.mock('../backendapi/apiFlimLabs', () => ({
  apiFlimGetStatus: jest.fn(() => Promise.resolve({ available: false })),
}));

// jsdom has no PointerEvent; without it fireEvent.pointerMove drops clientX/Y.
if (typeof window !== 'undefined' && !window.PointerEvent) {
  window.PointerEvent = class PointerEvent extends window.MouseEvent {
    constructor(type, init = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 0;
    }
  };
}

const makeStore = () =>
  configureStore({
    reducer: {
      galvoScannerState: galvoReducer,
      connectionSettingsState: () => ({ ip: 'http://10.0.0.5', apiPort: 8001 }),
    },
  });

// ---------------------------------------------------------------------------
// Slice: width / centre / drag semantics
// ---------------------------------------------------------------------------
describe('GalvoScannerSlice window editing', () => {
  const cfg = (store) => store.getState().galvoScannerState.config;

  test('changing the width keeps the centre', () => {
    const store = makeStore();
    store.dispatch(setXRange([500, 3500])); // centre 2000
    store.dispatch(setAxisWindow({ axis: 'x', width: 1000 }));
    expect([cfg(store).x_min, cfg(store).x_max]).toEqual([1500, 2500]);
  });

  test('changing the centre keeps the width', () => {
    const store = makeStore();
    store.dispatch(setXRange([500, 3500]));
    store.dispatch(setAxisWindow({ axis: 'x', center: 1000 }));
    expect([cfg(store).x_min, cfg(store).x_max]).toEqual([0, 3000]);
  });

  test('window slides instead of shrinking at the DAC limits', () => {
    const store = makeStore();
    store.dispatch(setAxisWindow({ axis: 'y', center: 4000, width: 1000 }));
    expect([cfg(store).y_min, cfg(store).y_max]).toEqual([3095, 4095]);
    store.dispatch(setAxisWindow({ axis: 'y', center: 10 }));
    expect([cfg(store).y_min, cfg(store).y_max]).toEqual([0, 1000]);
  });

  test('width is capped at the full range', () => {
    const store = makeStore();
    store.dispatch(setAxisWindow({ axis: 'x', width: 9999 }));
    expect([cfg(store).x_min, cfg(store).x_max]).toEqual([0, 4095]);
  });

  test('moveScanArea shifts both axes and stops at the edges', () => {
    const store = makeStore();
    store.dispatch(setScanArea({ x_min: 100, x_max: 600, y_min: 200, y_max: 400 }));
    store.dispatch(moveScanArea({ dx: 50, dy: -300 }));
    expect(cfg(store)).toMatchObject({ x_min: 150, x_max: 650, y_min: 0, y_max: 200 });
  });

  test('setScanArea clamps, orders and never collapses to zero size', () => {
    const store = makeStore();
    store.dispatch(setScanArea({ x_min: 5000, x_max: -3, y_min: 700, y_max: 700 }));
    expect(cfg(store)).toMatchObject({ x_min: 0, x_max: 4095, y_min: 700, y_max: 701 });
  });
});

describe('getScanPhysical', () => {
  test('is null while uncalibrated', () => {
    const store = makeStore();
    expect(getScanPhysical(store.getState())).toBeNull();
  });

  test('derives FOV and pixel size from µm per DAC count and the live config', () => {
    const store = makeStore();
    store.dispatch(setScanArea({ x_min: 500, x_max: 3500, y_min: 1000, y_max: 2000 }));
    store.dispatch(setCameraCalibration({ calibrated: true, umPerDacX: 0.125, umPerDacY: 0.25 }));
    const p = getScanPhysical(store.getState());
    expect(p.fovUmX).toBeCloseTo(375);
    expect(p.fovUmY).toBeCloseTo(250);
    expect(p.pixelUmX).toBeCloseTo(375 / 256);
    expect(p.fullScaleUmX).toBeCloseTo(512);
    // ...and it follows the sliders without another backend round-trip
    store.dispatch(setAxisWindow({ axis: 'x', width: 1000 }));
    expect(getScanPhysical(store.getState()).fovUmX).toBeCloseTo(125);
  });
});

// ---------------------------------------------------------------------------
// Preview: drag / resize / backdrop transform
// ---------------------------------------------------------------------------
describe('GalvoScanPreview', () => {
  const identity = { a11: 1, a12: 0, tx: 0, a21: 0, a22: 1, ty: 0 };
  const baseConfig = { x_min: 1000, x_max: 2000, y_min: 1000, y_max: 2000, nx: 8, ny: 8, bidirectional: false };
  // size 320, padding 25 -> 270 px for 4096 counts
  const k = 270 / 4096;

  test('dragging the rectangle moves the scan area by the pointer delta in DAC counts', () => {
    const onAreaChange = jest.fn();
    render(<GalvoScanPreview config={baseConfig} affine={identity} onAreaChange={onAreaChange} size={320} />);
    const area = screen.getByTestId('galvo-scan-area');
    fireEvent.pointerDown(area, { clientX: 100, clientY: 100, pointerId: 1 });
    // 27 px ~ 409.6 counts; jsdom has no CTM so canvas px == client px
    fireEvent.pointerMove(screen.getByTestId('galvo-scan-preview'), { clientX: 127, clientY: 100, pointerId: 1 });
    expect(onAreaChange).toHaveBeenCalled();
    const next = onAreaChange.mock.calls[0][0];
    expect(next.x_min).toBe(1000 + Math.round(27 / k));
    expect(next.x_max - next.x_min).toBe(1000);
    expect(next.y_min).toBe(1000);
  });

  test('moving past the edge stops at 4095 and keeps the size', () => {
    const onAreaChange = jest.fn();
    render(<GalvoScanPreview config={baseConfig} affine={identity} onAreaChange={onAreaChange} size={320} />);
    fireEvent.pointerDown(screen.getByTestId('galvo-scan-area'), { clientX: 100, clientY: 100, pointerId: 1 });
    fireEvent.pointerMove(screen.getByTestId('galvo-scan-preview'), { clientX: 900, clientY: 100, pointerId: 1 });
    const next = onAreaChange.mock.calls.pop()[0];
    expect(next).toMatchObject({ x_min: 3095, x_max: 4095 });
  });

  test('pulling the south-east handle resizes without moving the origin', () => {
    const onAreaChange = jest.fn();
    render(<GalvoScanPreview config={baseConfig} affine={identity} onAreaChange={onAreaChange} size={320} />);
    fireEvent.pointerDown(screen.getByTestId('galvo-scan-handle-se'), { clientX: 200, clientY: 200, pointerId: 1 });
    fireEvent.pointerMove(screen.getByTestId('galvo-scan-preview'), { clientX: 227, clientY: 213, pointerId: 1 });
    const next = onAreaChange.mock.calls.pop()[0];
    expect(next.x_min).toBe(1000);
    expect(next.y_min).toBe(1000);
    expect(next.x_max).toBe(2000 + Math.round(27 / k));
    expect(next.y_max).toBe(2000 + Math.round(13 / k));
  });

  test('a handle cannot cross the opposite edge', () => {
    const onAreaChange = jest.fn();
    render(<GalvoScanPreview config={baseConfig} affine={identity} onAreaChange={onAreaChange} size={320} />);
    fireEvent.pointerDown(screen.getByTestId('galvo-scan-handle-nw'), { clientX: 100, clientY: 100, pointerId: 1 });
    fireEvent.pointerMove(screen.getByTestId('galvo-scan-preview'), { clientX: 300, clientY: 300, pointerId: 1 });
    const next = onAreaChange.mock.calls.pop()[0];
    expect(next).toMatchObject({ x_min: 1999, y_min: 1999, x_max: 2000, y_max: 2000 });
  });

  test('nothing is reported after the pointer is released', () => {
    const onAreaChange = jest.fn();
    render(<GalvoScanPreview config={baseConfig} affine={identity} onAreaChange={onAreaChange} size={320} />);
    const svg = screen.getByTestId('galvo-scan-preview');
    fireEvent.pointerDown(screen.getByTestId('galvo-scan-area'), { clientX: 100, clientY: 100, pointerId: 1 });
    fireEvent.pointerUp(svg, { pointerId: 1 });
    fireEvent.pointerMove(svg, { clientX: 150, clientY: 100, pointerId: 1 });
    expect(onAreaChange).not.toHaveBeenCalled();
  });

  test('uncalibrated backdrop is stretched over the full DAC range', () => {
    const background = {
      image: 'data:image/png;base64,AAAA', width: 200, height: 100,
      frameWidth: 800, frameHeight: 400, subsampling: 4,
    };
    render(<GalvoScanPreview config={baseConfig} affine={identity} background={background} size={320} />);
    const img = screen.getByTestId('galvo-preview-background');
    const m = img.getAttribute('transform').match(/matrix\(([^)]+)\)/)[1].split(' ').map(Number);
    // 200 image px * a  must span the 270 px DAC square: a = 4095/800 * 4 * k
    expect(m[0] * 200).toBeCloseTo(270 * (4095 / 4096), 1);
    expect(m[3] * 100).toBeCloseTo(270 * (4095 / 4096), 1);
    expect(m[1]).toBeCloseTo(0);
    expect(m[2]).toBeCloseTo(0);
    expect(m[4]).toBeCloseTo(25, 3);
    expect(m[5]).toBeCloseTo(25, 3);
  });

  test('calibrated backdrop goes through the affine (camera px -> DAC -> canvas)', () => {
    // 1 camera px = 2 DAC counts, camera origin at DAC (100, 300)
    const affine = { a11: 2, a12: 0, tx: 100, a21: 0, a22: 2, ty: 300 };
    const background = {
      image: 'data:image/png;base64,AAAA', width: 100, height: 50,
      frameWidth: 400, frameHeight: 200, subsampling: 4,
    };
    render(<GalvoScanPreview config={baseConfig} affine={affine} background={background} size={320} />);
    const img = screen.getByTestId('galvo-preview-background');
    const m = img.getAttribute('transform').match(/matrix\(([^)]+)\)/)[1].split(' ').map(Number);
    expect(m[0]).toBeCloseTo(k * 4 * 2, 4);
    expect(m[3]).toBeCloseTo(k * 4 * 2, 4);
    expect(m[4]).toBeCloseTo(25 + k * 100, 3);
    expect(m[5]).toBeCloseTo(25 + k * 300, 3);
  });

  test('labels switch to micrometres once physical sizes are known', () => {
    const physical = { fovUmX: 125, fovUmY: 125, fullScaleUmX: 512, fullScaleUmY: 512, umPerDacX: 0.125, umPerDacY: 0.125 };
    render(<GalvoScanPreview config={baseConfig} affine={identity} physical={physical} size={320} />);
    expect(screen.getByText(/DAC 0–4095 ≈ 512 µm × 512 µm/)).toBeInTheDocument();
    expect(screen.getByText(/1000×1000 ≈ 125 µm × 125 µm/)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// FLIM LABS web UI link
// ---------------------------------------------------------------------------
describe('flimWebUiUrl', () => {
  test('uses the ImSwitch host with the FLIM port, never the backend\'s localhost', () => {
    expect(flimWebUiUrl('http://10.0.0.5', 'http://localhost:5249')).toBe('http://10.0.0.5:5249');
  });
  test('falls back to 5249 without a server url and drops an https scheme', () => {
    expect(flimWebUiUrl('https://scope.local', undefined)).toBe('http://scope.local:5249');
  });
  test('keeps a non-default port from the server url', () => {
    expect(flimWebUiUrl('http://10.0.0.5', 'http://127.0.0.1:6000')).toBe('http://10.0.0.5:6000');
  });
  test('tolerates a bare hostname', () => {
    expect(flimWebUiUrl('10.0.0.5:8001', null)).toBe('http://10.0.0.5:5249');
  });
});

// ---------------------------------------------------------------------------
// Raster tab integration
// ---------------------------------------------------------------------------
describe('GalvoScannerController raster tab', () => {
  const routes = {};
  beforeEach(() => {
    Object.keys(routes).forEach((k) => delete routes[k]);
    routes.getGalvoScannerNames = ['galvo'];
    routes.getGalvoScannerConfig = { config: { x_min: 500, x_max: 3500, y_min: 500, y_max: 3500 } };
    routes.getGalvoScannerStatus = { running: false };
    routes.getGalvoParkConfig = { park_x: 2048, park_y: 2048, park_on_stop: true };
    routes.getAffineTransform = { affine_transform: { a11: 1, a12: 0, tx: 0, a21: 0, a22: 1, ty: 0 } };
    routes.getGalvoCameraCalibration = {
      calibrated: false, detectorName: 'WidefieldCamera', cameraDetectors: ['WidefieldCamera'],
      frameWidth: 640, frameHeight: 480, pixelSizeUmX: 0.5, pixelSizeUmY: 0.5,
      hint: 'Scanner is not calibrated to the camera: run the 3-point affine wizard (Arbitrary Points tab).',
    };
    routes.snapGalvoCameraBackground = {
      detectorName: 'WidefieldCamera', image: 'data:image/png;base64,AAAA',
      width: 160, height: 120, frameWidth: 640, frameHeight: 480, subsampling: 4,
    };
    global.fetch = jest.fn((url) => {
      const name = String(url).split('/').pop().split('?')[0];
      return Promise.resolve({ json: () => Promise.resolve(routes[name] ?? {}) });
    });
  });

  // Mount inside act() so the mount-time fetches settle before assertions.
  const mountController = async () => {
    const GalvoScannerController = require('../components/GalvoScannerController').default;
    const store = makeStore();
    // eslint-disable-next-line testing-library/no-unnecessary-act
    await act(async () => {
      render(
        <Provider store={store}>
          <GalvoScannerController />
        </Provider>
      );
    });
    return store;
  };

  test('shows the calibration panel pointing at the wizard and switches to it', async () => {
    const store = await mountController();
    expect(screen.getByText('Not calibrated')).toBeInTheDocument();
    expect(screen.getByText(/run the 3-point affine wizard/)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('galvo-open-wizard'));
    expect(store.getState().galvoScannerState.activeTab).toBe(1);
    expect(store.getState().galvoScannerState.calibration.active).toBe(true);
    // The wizard (on the other tab) finishes -> back to the raster tab
    act(() => { store.dispatch(setCalibrationComplete()); });
    expect(store.getState().galvoScannerState.activeTab).toBe(0);
  });

  test('cancelling the wizard stays on the Arbitrary Points tab', async () => {
    const store = await mountController();
    fireEvent.click(screen.getByTestId('galvo-open-wizard'));
    act(() => { store.dispatch(cancelCalibration()); });
    expect(store.getState().galvoScannerState.activeTab).toBe(1);
  });

  test('width and centre fields drive the window', async () => {
    const store = await mountController();
    fireEvent.change(screen.getByTestId('galvo-x-width'), { target: { value: '1000' } });
    expect(store.getState().galvoScannerState.config).toMatchObject({ x_min: 1500, x_max: 2500 });
    fireEvent.change(screen.getByTestId('galvo-x-center'), { target: { value: '600' } });
    expect(store.getState().galvoScannerState.config).toMatchObject({ x_min: 100, x_max: 1100 });
  });

  test('snap shows the camera behind the preview and clear removes it', async () => {
    await mountController();
    expect(screen.queryByTestId('galvo-preview-background')).toBeNull();
    fireEvent.click(screen.getByTestId('galvo-snap-background'));
    expect(await screen.findByTestId('galvo-preview-background')).toBeInTheDocument();
    expect(global.fetch.mock.calls.some(([u]) => String(u).includes('snapGalvoCameraBackground?maxDim=1024') && String(u).includes('detectorName=WidefieldCamera'))).toBe(true);
    expect(screen.getByText(/Uncalibrated: camera frame is assumed/)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('galvo-clear-background'));
    expect(screen.queryByTestId('galvo-preview-background')).toBeNull();
  });

  test('calibrated: µm labels appear in the range panel', async () => {
    routes.getAffineTransform = { affine_transform: { a11: 8, a12: 0, tx: 0, a21: 0, a22: 8, ty: 0 } };
    routes.getGalvoCameraCalibration = {
      calibrated: true, detectorName: 'WidefieldCamera', cameraDetectors: ['WidefieldCamera'],
      frameWidth: 640, frameHeight: 480, pixelSizeUmX: 0.5, pixelSizeUmY: 0.5,
      umPerDacX: 0.0625, umPerDacY: 0.0625, rotationDeg: 0, flim: {},
    };
    await mountController();
    expect(screen.getByText('Calibrated')).toBeInTheDocument();
    // 3000 counts * 0.0625 = 187.5 -> "188 µm"
    expect(screen.getAllByText(/≈ 188 µm/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Scan field: 188 µm × 188 µm/)).toBeInTheDocument();
  });
});
