// Dedicated dark theme for the /mobile kiosk UI shown on the Raspberry Pi
// DSI touchscreen (chromium kiosk mode). Bambu Lab / Formlabs-inspired:
// near-black surfaces, one strong brand accent, generous touch targets.
// Deliberately separate from themes/darkTheme.js — the desktop theme is
// density-optimised (smaller fonts/paddings), which is exactly wrong for a
// 7" touchscreen, so the two must never share component overrides.
import { createTheme } from "@mui/material/styles";

export const kioskColors = {
  background: "#0b0f14",
  surface: "#151b23",
  surfaceRaised: "#1c242e",
  accent: "#85b918", // openUC2 lime
  accentBlue: "#4aa8ff",
  danger: "#ef5350",
  warning: "#ffb74d",
  success: "#66bb6a",
};

const mobileTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: kioskColors.accent, contrastText: "#0b0f14" },
    secondary: { main: kioskColors.accentBlue },
    success: { main: kioskColors.success },
    warning: { main: kioskColors.warning },
    error: { main: kioskColors.danger },
    background: {
      default: kioskColors.background,
      paper: kioskColors.surface,
    },
    divider: "rgba(255,255,255,0.08)",
    text: { primary: "#e6ebf2", secondary: "#93a1b3" },
  },
  shape: { borderRadius: 14 },
  typography: {
    fontFamily: "Roboto, system-ui, sans-serif",
    fontSize: 15,
    fontWeightBold: 700,
    h4: { fontWeight: 800, letterSpacing: "-0.02em" },
    h5: { fontWeight: 700, letterSpacing: "-0.01em" },
    h6: { fontWeight: 700 },
    button: { textTransform: "none", fontWeight: 700 },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: `
        @font-face {
          font-family: 'Roboto';
          font-style: normal;
          font-display: swap;
          font-weight: 400;
          src: local('Roboto'),
               url('${process.env.PUBLIC_URL}/fonts/Roboto-Regular.ttf') format('truetype');
        }
        @font-face {
          font-family: 'Roboto';
          font-style: normal;
          font-display: swap;
          font-weight: 700;
          src: local('Roboto Bold'),
               url('${process.env.PUBLIC_URL}/fonts/Roboto-Bold.ttf') format('truetype');
        }
        html, body, #root { height: 100%; }
        body {
          overscroll-behavior: none;
          touch-action: manipulation;
        }
      `,
    },
    MuiButton: {
      styleOverrides: {
        root: {
          minHeight: 56,
          padding: "12px 22px",
          borderRadius: 14,
          touchAction: "manipulation",
        },
        sizeLarge: { minHeight: 68, fontSize: "1.05rem" },
        sizeSmall: { minHeight: 44, padding: "8px 16px" },
      },
    },
    MuiIconButton: {
      styleOverrides: { root: { minWidth: 48, minHeight: 48 } },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: { minHeight: 52, minWidth: 60, textTransform: "none", fontWeight: 700 },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: { height: 10, padding: "18px 0" },
        thumb: { width: 30, height: 30 },
      },
    },
    // Kill MUI's dark-mode elevation tint so the flat near-black surfaces
    // and 1px borders read cleanly (the tint drifts Paper lighter per level).
    MuiPaper: {
      styleOverrides: { root: { backgroundImage: "none" } },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          border: "1px solid rgba(255,255,255,0.08)",
        },
      },
    },
    MuiChip: {
      styleOverrides: { root: { fontWeight: 700 } },
    },
    MuiDialogActions: {
      styleOverrides: { root: { padding: 16, gap: 8 } },
    },
  },
});

export default mobileTheme;
