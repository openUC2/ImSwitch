// Dashed placeholder frame for explanatory images/renderings that will be
// dropped in later (sample loading photos, objective renderings, ...).
import { Box, Typography } from "@mui/material";
import ImageOutlinedIcon from "@mui/icons-material/ImageOutlined";

const PlaceholderImage = ({ label = "Illustration", height = 140, sx }) => (
  <Box
    sx={{
      height,
      borderRadius: 3,
      border: "2px dashed",
      borderColor: "divider",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: 1,
      color: "text.disabled",
      bgcolor: "rgba(255,255,255,0.02)",
      ...sx,
    }}
  >
    <ImageOutlinedIcon />
    <Typography variant="caption" align="center" sx={{ px: 2 }}>
      {label}
    </Typography>
  </Box>
);

export default PlaceholderImage;
