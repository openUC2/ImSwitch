// Large stacked icon+label button for primary touch actions.
import { Button, Typography } from "@mui/material";

const TouchButton = ({ icon, label, caption, sx, ...buttonProps }) => (
  <Button
    variant="outlined"
    size="large"
    sx={{
      flexDirection: "column",
      gap: 0.5,
      minHeight: 92,
      px: 2,
      lineHeight: 1.2,
      "& svg": { fontSize: 30 },
      ...sx,
    }}
    {...buttonProps}
  >
    {icon}
    <span>{label}</span>
    {caption && (
      <Typography variant="caption" color="text.secondary" component="span">
        {caption}
      </Typography>
    )}
  </Button>
);

export default TouchButton;
