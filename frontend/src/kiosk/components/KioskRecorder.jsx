/**
 * KioskRecorder.jsx
 * 
 * Simplified Recording Controls for Kiosk Mode
 * - Start/Stop/Pause recording
 * - Current recording status
 * - Touch-optimized buttons
 */

import { useState } from "react";
import {
  Box,
  Button,
  Paper,
  Typography,
  Stack,
  Chip,
} from "@mui/material";
import {
  FiberManualRecord as RecordIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  PlayArrow as ResumeIcon,
} from "@mui/icons-material";

const KioskRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [duration, setDuration] = useState("00:00:00");
  const [frameCount, setFrameCount] = useState(0);

  const handleStartRecording = () => {
    // TODO: Integrate with actual recorder API
    setIsRecording(true);
    setIsPaused(false);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    setIsPaused(false);
  };

  const handlePauseResume = () => {
    setIsPaused(!isPaused);
  };

  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: 2,
        p: 2,
      }}
    >
      {/* Status Display */}
      <Paper
        elevation={2}
        sx={{
          p: 2,
          textAlign: "center",
          backgroundColor: isRecording
            ? isPaused
              ? "warning.dark"
              : "error.dark"
            : "background.paper",
        }}
      >
        <Typography variant="h5" gutterBottom>
          Recording Status
        </Typography>
        <Chip
          label={
            isRecording
              ? isPaused
                ? "PAUSED"
                : "RECORDING"
              : "STOPPED"
          }
          color={isRecording ? (isPaused ? "warning" : "error") : "default"}
          size="large"
          sx={{ fontSize: "1.2rem", py: 2, px: 3 }}
        />
      </Paper>

      {/* Control Buttons */}
      <Stack spacing={2}>
        {!isRecording ? (
          <Button
            variant="contained"
            color="error"
            size="large"
            startIcon={<RecordIcon />}
            onClick={handleStartRecording}
            fullWidth
            sx={{
              py: 2,
              fontSize: "1.2rem",
            }}
          >
            Start Recording
          </Button>
        ) : (
          <>
            <Button
              variant="contained"
              color="warning"
              size="large"
              startIcon={isPaused ? <ResumeIcon /> : <PauseIcon />}
              onClick={handlePauseResume}
              fullWidth
              sx={{
                py: 2,
                fontSize: "1.2rem",
              }}
            >
              {isPaused ? "Resume" : "Pause"}
            </Button>
            <Button
              variant="contained"
              color="primary"
              size="large"
              startIcon={<StopIcon />}
              onClick={handleStopRecording}
              fullWidth
              sx={{
                py: 2,
                fontSize: "1.2rem",
              }}
            >
              Stop Recording
            </Button>
          </>
        )}
      </Stack>

      {/* Recording Info */}
      {/* Recording Info */}
      {isRecording && (
        <Paper
          elevation={1}
          sx={{
            p: 2,
            mt: "auto",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Duration: {duration}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Frames: {frameCount}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default KioskRecorder;
