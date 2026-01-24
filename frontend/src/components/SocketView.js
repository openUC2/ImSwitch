import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { Button, Box, Typography, Paper, Chip } from "@mui/material";
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
    <Box sx={{ p: 2 }}>
      <Box sx={{ mb: 2, display: "flex", gap: 2, alignItems: "center" }}>
        <Typography variant="h6">Socket Debug View</Typography>
        <Chip 
          label={`${messages.length} messages`} 
          color="primary" 
          size="small" 
        />
        <Button 
          variant="outlined" 
          size="small" 
          onClick={handleClearMessages}
        >
          Clear Messages
        </Button>
        <Button 
          variant={filterImageUpdates ? "contained" : "outlined"}
          size="small" 
          onClick={toggleImageFilter}
        >
          {filterImageUpdates ? "Showing: No Images" : "Showing: All"}
        </Button>
      </Box>
      
      <Paper 
        sx={{ 
          maxHeight: "600px", 
          overflow: "auto", 
          p: 2,
          backgroundColor: "background.default"
        }}
      >
        {messages.length === 0 ? (
          <Typography color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
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
                backgroundColor: "background.paper"
              }}
            >
              <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
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
                  m: 0
                }}
              >
                {JSON.stringify(message.args, null, 2)}
              </Typography>
            </Paper>
          ))
        )}
      </Paper>
    </Box>
  );
};

export default SocketView;