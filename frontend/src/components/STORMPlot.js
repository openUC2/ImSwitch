// src/components/STORMPlot.js
import React, { useRef, useEffect, useCallback } from 'react';
import { Box, Typography, FormControlLabel, Checkbox, TextField, Grid } from '@mui/material';

const STORMPlot = ({ data, width = 400, height = 300, title = "STORM Localizations" }) => {
  const canvasRef = useRef(null);
  const [showDensity, setShowDensity] = React.useState(false);
  const [pointSize, setPointSize] = React.useState(1);
  const [colormap, setColormap] = React.useState('hot');

  // Draw the plot
  const drawPlot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Find bounds of data
    const xCoords = data.map(d => d.x || 0);
    const yCoords = data.map(d => d.y || 0);
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    // Draw background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw points
    if (showDensity) {
      // Create density map
      const binSize = 5;
      const binsX = Math.ceil(canvas.width / binSize);
      const binsY = Math.ceil(canvas.height / binSize);
      const densityMap = new Array(binsX * binsY).fill(0);

      // Populate density map
      data.forEach(point => {
        const x = ((point.x - minX) / rangeX) * (canvas.width - 1);
        const y = ((point.y - minY) / rangeY) * (canvas.height - 1);
        const binX = Math.floor(x / binSize);
        const binY = Math.floor(y / binSize);
        if (binX >= 0 && binX < binsX && binY >= 0 && binY < binsY) {
          densityMap[binY * binsX + binX]++;
        }
      });

      const maxDensity = Math.max(...densityMap);

      // Draw density map
      for (let i = 0; i < densityMap.length; i++) {
        const density = densityMap[i];
        if (density > 0) {
          const x = (i % binsX) * binSize;
          const y = Math.floor(i / binsX) * binSize;
          const intensity = density / maxDensity;
          
          // Apply colormap
          let r, g, b;
          if (colormap === 'hot') {
            r = Math.floor(intensity * 255);
            g = Math.floor(Math.max(0, (intensity - 0.33) * 255 / 0.67));
            b = Math.floor(Math.max(0, (intensity - 0.67) * 255 / 0.33));
          } else if (colormap === 'viridis') {
            // Simplified viridis colormap
            r = Math.floor(intensity * 68 + (1 - intensity) * 68);
            g = Math.floor(intensity * 255 + (1 - intensity) * 1);
            b = Math.floor(intensity * 84 + (1 - intensity) * 84);
          } else {
            // Default grayscale
            const gray = Math.floor(intensity * 255);
            r = g = b = gray;
          }
          
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(x, y, binSize, binSize);
        }
      }
    } else {
      // Draw individual points
      ctx.fillStyle = '#ffffff';
      data.forEach(point => {
        const x = ((point.x - minX) / rangeX) * (canvas.width - 1);
        const y = ((point.y - minY) / rangeY) * (canvas.height - 1);
        
        ctx.beginPath();
        ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw axes labels
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.fillText(`Min X: ${minX.toFixed(1)}`, 5, canvas.height - 25);
    ctx.fillText(`Max X: ${maxX.toFixed(1)}`, canvas.width - 80, canvas.height - 25);
    ctx.fillText(`Min Y: ${minY.toFixed(1)}`, 5, 15);
    ctx.fillText(`Max Y: ${maxY.toFixed(1)}`, 5, canvas.height - 5);
    ctx.fillText(`N: ${data.length}`, canvas.width - 60, 15);
  }, [data, showDensity, pointSize, colormap]);

  useEffect(() => {
    drawPlot();
  }, [drawPlot]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={6}>
          <FormControlLabel
            control={
              <Checkbox
                checked={showDensity}
                onChange={(e) => setShowDensity(e.target.checked)}
                size="small"
              />
            }
            label="Density Mode"
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="Point Size"
            type="number"
            value={pointSize}
            onChange={(e) => setPointSize(Math.max(1, parseInt(e.target.value) || 1))}
            inputProps={{ min: 1, max: 10 }}
            size="small"
            disabled={showDensity}
          />
        </Grid>
      </Grid>

      <Box sx={{ textAlign: 'center', border: '1px solid #ddd' }}>
        {data && data.length > 0 ? (
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{ maxWidth: '100%', height: 'auto', backgroundColor: '#000' }}
          />
        ) : (
          <Box
            sx={{
              width: width,
              height: height,
              backgroundColor: '#f5f5f5',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              maxWidth: '100%'
            }}
          >
            <Typography color="textSecondary">
              No localization data available
            </Typography>
          </Box>
        )}
      </Box>

      {data && data.length > 0 && (
        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
          Showing {data.length} localizations
        </Typography>
      )}
    </Box>
  );
};

export default STORMPlot;