// Standard kiosk page scaffold: fixed header row (title/subtitle/action) and
// a single scrollable content region — the shell itself never scrolls.
import { Box, Typography } from "@mui/material";

const MobilePage = ({ title, subtitle, action, children, disablePadding = false }) => (
  <Box
    sx={{
      height: "100%",
      display: "flex",
      flexDirection: "column",
      minWidth: 0,
    }}
  >
    <Box
      sx={{
        px: 3,
        py: 2,
        flexShrink: 0,
        display: "flex",
        alignItems: "center",
        gap: 2,
        borderBottom: "1px solid",
        borderColor: "divider",
      }}
    >
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography variant="h5" noWrap>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary" noWrap>
            {subtitle}
          </Typography>
        )}
      </Box>
      {action}
    </Box>
    <Box
      sx={{
        flex: 1,
        minHeight: 0,
        overflowY: "auto",
        p: disablePadding ? 0 : 3,
      }}
    >
      {children}
    </Box>
  </Box>
);

export default MobilePage;
