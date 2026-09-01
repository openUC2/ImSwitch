// Bambu-style status tile: small icon+label caption on top, big value below.
import { Box, Paper, Typography } from "@mui/material";

const StatusTile = ({ icon, label, value, valueColor = "text.primary", onClick, sx }) => (
  <Paper
    variant="outlined"
    onClick={onClick}
    sx={{
      p: 2,
      borderRadius: 3,
      display: "flex",
      flexDirection: "column",
      gap: 0.75,
      flex: 1,
      minWidth: 120,
      cursor: onClick ? "pointer" : "default",
      ...sx,
    }}
  >
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        color: "text.secondary",
        "& svg": { fontSize: 18 },
      }}
    >
      {icon}
      <Typography variant="caption" sx={{ fontWeight: 700, letterSpacing: "0.5px" }}>
        {label}
      </Typography>
    </Box>
    <Typography
      variant="h5"
      sx={{ color: valueColor, fontVariantNumeric: "tabular-nums", lineHeight: 1.2 }}
    >
      {value}
    </Typography>
  </Paper>
);

export default StatusTile;
