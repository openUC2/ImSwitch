import { Box, Typography } from "@mui/material";

const ConnectionDot = ({ ok, label, labelOn = "Online", labelOff = "Offline" }) => (
  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 1 }}>
    <Box
      sx={{
        width: 10,
        height: 10,
        borderRadius: "50%",
        flexShrink: 0,
        bgcolor: ok ? "success.main" : "error.main",
        boxShadow: (theme) =>
          ok ? `0 0 8px ${theme.palette.success.main}` : "none",
      }}
    />
    <Typography variant="body2" color="text.secondary">
      {label ? `${label}: ` : ""}
      {ok ? labelOn : labelOff}
    </Typography>
  </Box>
);

export default ConnectionDot;
