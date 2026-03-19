import React from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Button,
  Box,
  Typography,
  Paper,
  Chip,
  Card,
  CardContent,
} from "@mui/material";
import * as socketDebugSlice from "../state/slices/SocketDebugSlice.js";

const SocketView = () => {
  const dispatch = useDispatch();

  // Get messages from Redux
  const socketDebugState = useSelector(socketDebugSlice.getSocketDebugState);
  const { messages, filterImageUpdates } = socketDebugState;

  const handleClearMessages = () => {
    dispatch(socketDebugSlice.clearMessages());
  };

  const toggleImageFilter = () => {
    dispatch(socketDebugSlice.setFilterImageUpdates(!filterImageUpdates));
  };

  return (
    <Box
      sx={{
        p: 3,
        display: "flex",
        flexDirection: "column",
        height: "100%",
        width: "100%",
        minHeight: 0,
      }}
    >
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Socket Debug View
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Monitor and debug WebSocket connections in real-time
        </Typography>
      </Box>

      <Box sx={{ mb: 2, display: "flex", gap: 2, alignItems: "center" }}>
        <Chip
          label={`${messages.length} messages`}
          color="primary"
          size="small"
        />
        <Button variant="outlined" size="small" onClick={handleClearMessages}>
          Clear Messages
        </Button>
        <Button
          variant={filterImageUpdates ? "contained" : "outlined"}
          size="small"
          onClick={toggleImageFilter}
        >
          {filterImageUpdates ? "Showing: No Images" : "Showing: All"}
        </Button>
        <Box sx={{ flexGrow: 1 }} />
      </Box>

      <Card
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          width: "100%",
        }}
      >
        <CardContent
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "auto",
            width: "100%",
            minHeight: 0,
          }}
        >
          <Paper
            sx={{
              flex: 1,
              overflow: "auto",
              p: 2,
              backgroundColor: "background.default",
              width: "100%",
            }}
          >
            {messages.length === 0 ? (
              <Typography
                color="text.secondary"
                sx={{ textAlign: "center", py: 4 }}
              >
                No messages received yet. Waiting for socket signals...
              </Typography>
            ) : (
              messages.map((message, index) => (
                <Paper
                  key={index}
                  elevation={1}
                  sx={{
                    mb: 1,
                    p: 1.5,
                    borderLeft: 3,
                    borderColor: "primary.main",
                    backgroundColor: "background.paper",
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 0.5,
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: "bold" }}>
                      {message.name}
                    </Typography>
                    {message.timestamp && (
                      <Typography variant="caption" color="text.secondary">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </Typography>
                    )}
                  </Box>
                  <Typography
                    variant="body2"
                    component="pre"
                    sx={{
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      fontFamily: "monospace",
                      fontSize: "0.75rem",
                      color: "text.secondary",
                      m: 0,
                    }}
                  >
                    {JSON.stringify(message.args, null, 2)}
                  </Typography>
                </Paper>
              ))
            )}
          </Paper>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SocketView;
