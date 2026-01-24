import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { Box, Card, CardContent, Typography } from "@mui/material";
import { Videocam as VideocamIcon } from "@mui/icons-material";
import { useWebSocket } from "../context/WebSocketContext";
import * as stageOffsetCalibrationSlice from "../state/slices/StageOffsetCalibrationSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import apiSettingsControllerGetDetectorNames from "../backendapi/apiSettingsControllerGetDetectorNames";

const LiveStreamTile = ({ width = 200, height = 150 }) => {
  const socket = useWebSocket();
  const dispatch = useDispatch();
  
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  
  // Access Redux state for image display
  const stageOffsetState = useSelector(stageOffsetCalibrationSlice.getStageOffsetCalibrationState);
  const imageUrls = stageOffsetState.imageUrls;
  const detectors = stageOffsetState.detectors;

  // Fetch the list of detectors from the server
  useEffect(() => {
    const fetchDetectorNames = async () => {
      try {
        const data = await apiSettingsControllerGetDetectorNames();
        dispatch(stageOffsetCalibrationSlice.setDetectors(data || []));
      } catch (error) {
        console.error("Error fetching detector names:", error);
      }
    };

    fetchDetectorNames();
  }, [dispatch]);

  // Handle socket signals for live stream
  useEffect(() => {
    if (!socket) return;
    const handleSignal = (rawData) => {
      try {
        const jdata = JSON.parse(rawData);
        if (jdata.name === "sigUpdateImage") {
          const detectorName = jdata.detectorname;
          const imgSrc = `data:image/jpeg;base64,${jdata.image}`;
          dispatch(stageOffsetCalibrationSlice.updateImageUrl({
            detector: detectorName,
            url: imgSrc
          }));
        }
      } catch (error) {
        console.error("Error parsing signal data:", error);
      }
    };
    socket.on("signal", handleSignal);
    return () => {
      socket.off("signal", handleSignal);
    };
  }, [socket, dispatch]);

  const hasImage = detectors.length > 0 && imageUrls[detectors[0]];

  return (
    <Card 
      elevation={3} 
      sx={{ 
        width: width, 
        height: height,
        position: 'relative',
        bgcolor: 'background.paper',
        border: '2px solid',
        borderColor: 'primary.main',
        borderRadius: 2,
      }}
    >
      <CardContent sx={{ p: 1, height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <VideocamIcon sx={{ fontSize: 16, mr: 0.5, color: 'primary.main' }} />
          <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
            Live Stream
          </Typography>
        </Box>
        
        {/* Image or placeholder */}
        <Box sx={{ 
          flex: 1, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          bgcolor: 'grey.100',
          borderRadius: 1,
          overflow: 'hidden'
        }}>
          {hasImage ? (
            <img
              src={imageUrls[detectors[0]]}
              alt="Live Stream"
              style={{ 
                width: '100%', 
                height: '100%', 
                objectFit: 'cover',
                borderRadius: '4px'
              }}
            />
          ) : (
            <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
              <VideocamIcon sx={{ fontSize: 32, mb: 1, opacity: 0.5 }} />
              <Typography variant="caption" display="block">
                No stream
              </Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default LiveStreamTile;