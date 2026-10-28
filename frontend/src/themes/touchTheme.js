import { createTheme } from "@mui/material/styles";

/**
 * Touch overrides layered on top of the desktop themes.
 *
 * The desktop themes are tuned for density (7 px spacing, small buttons).
 * With a finger that density turns into mis-taps, so when the touch UI is on
 * (see hooks/useDeviceProfile.js) the interactive controls get ~40 px minimum
 * hit areas. Typography and spacing are left alone so layouts do not reflow
 * wholesale — only the things you press grow.
 */
const touchOverrides = {
  components: {
    MuiButton: {
      styleOverrides: {
        root: { minHeight: 40, touchAction: "manipulation" },
        sizeSmall: { minHeight: 36 },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: { minWidth: 40, minHeight: 40, touchAction: "manipulation" },
        sizeSmall: { minWidth: 36, minHeight: 36 },
      },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: { minHeight: 40, minWidth: 40, touchAction: "manipulation" },
        sizeSmall: { minHeight: 36 },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { height: 36 },
        sizeSmall: { height: 32 },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: { minHeight: 48 },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: { minHeight: 44 },
      },
    },
    MuiCheckbox: {
      styleOverrides: {
        root: { padding: 10 },
      },
    },
    MuiRadio: {
      styleOverrides: {
        root: { padding: 10 },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: { padding: 10 },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: { padding: "18px 0" },
        thumb: { width: 28, height: 28 },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        inputSizeSmall: { paddingTop: 10, paddingBottom: 10 },
      },
    },
    MuiTooltip: {
      // Long-press shows a tooltip; keep it up long enough to read.
      defaultProps: { enterTouchDelay: 500, leaveTouchDelay: 3000 },
    },
  },
};

const cache = new WeakMap();

/** Returns `baseTheme` with touch-sized controls (memoised per base theme). */
export function withTouchOverrides(baseTheme) {
  if (!cache.has(baseTheme)) {
    cache.set(baseTheme, createTheme(baseTheme, touchOverrides));
  }
  return cache.get(baseTheme);
}
