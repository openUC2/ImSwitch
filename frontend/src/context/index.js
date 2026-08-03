// ─────────────────────────────────────────────────────────────────────────────
//  Public context surface for plugins.
//
//  Exposed to Module Federation as `host_app/contexts`. A plugin does:
//
//      import { useWebSocket, useWidgetContext } from "host_app/contexts";
//
//  and gets the *host's* context objects — not copies — so the hooks resolve
//  against the providers already mounted in App.jsx.
//
//  Note that most of what a plugin needs does NOT come from here. The Redux
//  store reaches it through the shared react-redux singleton (useSelector /
//  useDispatch work with no import from the host at all), and the MUI theme
//  through the shared @mui/material singleton (useTheme). This barrel is only
//  for the contexts ImSwitch defines itself, which federation cannot share
//  automatically because they are application modules rather than npm packages.
//
//  Adding an export here widens a published surface — see the stable-surface
//  table in docs/plugins/DECISIONS.md before doing so.
// ─────────────────────────────────────────────────────────────────────────────

// Live backend connection (Socket.IO). `useWebSocket` returns the host's
// already-connected client; a plugin should never open its own.
export { WebSocketProvider, useWebSocket } from "./WebSocketContext";

// Per-widget UI state shared between the shell and the active widget.
export { WidgetContextProvider, useWidgetContext } from "./WidgetContext";

// Live-view widget registry.
export { LiveWidgetContext, LiveWidgetProvider } from "./LiveWidgetContext";

// Jupyter kernel/session plumbing.
export { JupyterProvider, useJupyter } from "./JupyterContext";

// Progressive-web-app install/update state.
export { PWAProvider, usePWA } from "./PWAContext";

// The base MUI theme object. Prefer `useTheme()` from @mui/material inside a
// component — that reads the live theme including the user's dark-mode choice.
// This export is for the rare case where a plugin needs the raw default.
export { default as baseTheme } from "./ThemeContext";
