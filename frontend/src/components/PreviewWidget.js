import React from 'react';
import { Paper, Typography, Slider, TextField, Switch } from '@mui/material';

const PreviewWidget = ({ hostIP, hostPort, title }) => {
    const videoRef = useRef(null);
    const streamUrl = `${hostIP}:${hostPort}/imswitch/api/RecordingController/video_feeder`
    // simply display the stream from the camera under 
    return (
        <img
          style={{ width: "100%", height: "auto" }}
          autoPlay
          src={streamUrl}
          ref={videoRef}
        ></img>
    );
};

export default PreviewWidget;
