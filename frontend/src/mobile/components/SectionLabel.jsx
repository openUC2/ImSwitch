import { Typography } from "@mui/material";

const SectionLabel = ({ children, sx }) => (
  <Typography
    variant="subtitle2"
    sx={{
      textTransform: "uppercase",
      letterSpacing: "0.8px",
      fontWeight: 700,
      color: "text.secondary",
      mb: 1.5,
      ...sx,
    }}
  >
    {children}
  </Typography>
);

export default SectionLabel;
