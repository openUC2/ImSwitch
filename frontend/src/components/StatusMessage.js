import React, { useEffect } from "react";
import { Alert, Snackbar, Box, CircularProgress } from "@mui/material";

const StatusMessage = ({ 
  message, 
  type = "info", 
  showProgress = false,
  onClose,
  autoHideDuration = 6000 
}) => {
  const [open, setOpen] = React.useState(false);

  useEffect(() => {
    if (message) {
      setOpen(true);
    }
  }, [message]);

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpen(false);
    if (onClose) {
      onClose();
    }
  };

  if (!message) return null;

  return (
    <Snackbar 
      open={open} 
      autoHideDuration={type === "error" ? null : autoHideDuration}
      onClose={handleClose}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
    >
      <Alert 
        onClose={handleClose} 
        severity={type} 
        sx={{ 
          width: '100%',
          minWidth: '300px',
          alignItems: 'center'
        }}
        action={
          showProgress ? (
            <Box sx={{ display: 'flex', alignItems: 'center', mr: 1 }}>
              <CircularProgress size={20} color="inherit" />
            </Box>
          ) : undefined
        }
      >
        {message}
      </Alert>
    </Snackbar>
  );
};

export default StatusMessage;