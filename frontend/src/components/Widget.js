import React from 'react';
import { Grid, Paper, Typography, Slider, TextField, Switch } from '@mui/material';

const Widget = ({ title }) => {

    const handleSliderChange = (event, newValue) => {
        console.log(newValue);
    };
    
    return (
        <Paper style={{ padding: 20, margin: 10 }}>
            <Typography variant="h6">{title}</Typography>
            <Slider
                onChange={handleSliderChange}
                defaultValue={30}
                aria-labelledby="continuous-slider"
                style={{ marginBottom: 10 }}
            />
            <TextField
                label="Input"
                variant="outlined"
                size="small"
                style={{ marginBottom: 10 }}
            />
            <Switch defaultChecked />
        </Paper>
    );
};

const FixedGrid = () => {
    return (
        <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={4}>
                <Widget title="Widget 1" />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
                <Widget title="Widget 2" />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
                <Widget title="Widget 3" />
            </Grid>
            {/* Add more widgets as needed */}
        </Grid>
    );
};

export default FixedGrid;
