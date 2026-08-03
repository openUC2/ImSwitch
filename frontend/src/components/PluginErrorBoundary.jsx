// ─────────────────────────────────────────────────────────────────────────────
//  Error boundary for a federated plugin widget.
//
//  A plugin runs third-party code inside the operator's browser. Without a
//  boundary, one throwing widget unmounts the whole React tree and the
//  microscope UI goes blank — with no indication that a plugin caused it.
//
//  This catches both classes of failure:
//    * load failures rejected by loadRemote() and re-thrown by <Suspense>
//    * render/runtime errors thrown by the plugin's own component
//
//  Must be a class component: there is still no hook equivalent of
//  componentDidCatch.
// ─────────────────────────────────────────────────────────────────────────────
import React from "react";

import { Alert, AlertTitle, Box, Button, Typography } from "@mui/material";

class PluginErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Keep the component stack in the console — the inline card deliberately
    // shows only the message, since the stack is meaningless to an operator
    // but essential to the plugin author.
    console.error(
      `[plugin:${this.props.pluginName}] crashed while rendering`,
      error,
      info?.componentStack,
    );
  }

  componentDidUpdate(prevProps) {
    // Let the user get back to a working state by switching away and back,
    // and give "Retry" something to do after a transient network failure.
    if (prevProps.pluginName !== this.props.pluginName && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    const { error } = this.state;
    const { pluginName, children } = this.props;

    if (!error) return children;

    return (
      <Box sx={{ p: 2 }}>
        <Alert
          severity="error"
          action={
            <Button
              color="inherit"
              size="small"
              onClick={() => this.setState({ error: null })}
            >
              Retry
            </Button>
          }
        >
          <AlertTitle>Plugin “{pluginName}” could not be displayed</AlertTitle>
          <Typography
            variant="body2"
            component="pre"
            sx={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontFamily: "monospace",
              m: 0,
            }}
          >
            {error.message || String(error)}
          </Typography>
          <Typography variant="caption" sx={{ display: "block", mt: 1 }}>
            The rest of ImSwitch is unaffected. See the browser console for the
            full stack trace.
          </Typography>
        </Alert>
      </Box>
    );
  }
}

export default PluginErrorBoundary;
