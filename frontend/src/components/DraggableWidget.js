import React from 'react';
import { Paper, Typography, Slider, TextField, Switch } from '@mui/material';

const DraggableWidget = ({ title }) => {

    const handleSliderChange = (event, newValue) => {
        // Handle the new slider value here
        console.log(newValue);
    };
    
    return (
        <Paper style={{ padding: 20, margin: 10 }}>
            <Typography variant="h6" style={{ pointerEvents: 'auto' }}>{title}</Typography>
            <Slider
                className="no-drag"
                onChange={handleSliderChange}
                defaultValue={30}
                aria-labelledby="continuous-slider"
                style={{ pointerEvents: 'auto' }}
                draggable={false}
            />
            <TextField
                className="no-drag"
                label="Input"
                variant="outlined"
                size="small"
                style={{ marginTop: 10, pointerEvents: 'auto' }}
                draggable={false}
            />
            <Switch
                className="no-drag"
                defaultChecked
                style={{ pointerEvents: 'auto' }}
                draggable={false}
            />
        </Paper>
    );
};

export default DraggableWidget;
