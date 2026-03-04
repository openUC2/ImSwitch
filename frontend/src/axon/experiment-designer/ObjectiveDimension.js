import React from "react";
import { Box, Typography } from "@mui/material";
import ObjectiveSwitcher from "../../components/ObjectiveSwitcher";

/**
 * ObjectiveDimension – renders the ObjectiveSwitcher inside the experiment
 * designer's dimension panel so the user can switch objectives without
 * leaving the experiment setup flow.
 */
const ObjectiveDimension = () => {
  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
        Objective Lens
      </Typography>
      <ObjectiveSwitcher />
    </Box>
  );
};

export default ObjectiveDimension;
