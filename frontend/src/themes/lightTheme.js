import { createTheme } from "@mui/material/styles";

/**
 * ImSwitch Light Theme
 * Material-UI theme configuration for light mode
 * Follows Copilot Instructions for generic UI components
 */
export const lightTheme = createTheme({
  palette: {
    mode: "light",
  },
  typography: {
    fontFamily: "Roboto",
    fontWeightBold: 700,
    // Scale down all text by 10% for better density
    fontSize: 14,
  },
  spacing: 7, // Default is 8px, reduce to 7px for tighter layout
  components: {
    // Scale down buttons and form elements
    MuiButton: {
      styleOverrides: {
        root: {
          fontSize: "0.8rem",
          padding: "6px 14px", // Slightly smaller padding
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          "& .MuiInputBase-root": {
            fontSize: "0.85rem",
          },
        },
      },
    },
    MuiTypography: {
      styleOverrides: {
        root: {
          fontSize: "0.9em", // Scale down all typography
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          padding: "14px", // Slightly smaller card padding
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontSize: "0.8rem",
          minHeight: 42, // Slightly smaller tabs
        },
      },
    },
    MuiCssBaseline: {
      styleOverrides: `
        @font-face {
          font-family: 'Roboto';
          font-style: normal;
          font-display: swap;
          font-weight: 400;
          src: local('Roboto'),
               url('/imswitch/ui/fonts/Roboto-Regular.ttf') format('truetype');
        }
        @font-face {
          font-family: 'Roboto';
          font-style: normal;
          font-display: swap;
          font-weight: 700;
          src: local('Roboto Bold'),
               url('/imswitch/ui/fonts/Roboto-Bold.ttf') format('truetype');
        }
      `,
    },
  },
});
