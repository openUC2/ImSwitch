// Touch-friendly confirmation dialog: two equal thumb-sized buttons.
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";

const ConfirmDialog = ({
  open,
  title,
  text,
  confirmLabel = "Confirm",
  danger = false,
  onCancel,
  onConfirm,
}) => (
  <Dialog open={open} onClose={onCancel} fullWidth maxWidth="xs">
    <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
      {danger && <WarningAmberRoundedIcon color="warning" />}
      {title}
    </DialogTitle>
    <DialogContent>
      <DialogContentText>{text}</DialogContentText>
    </DialogContent>
    <DialogActions>
      <Button size="large" onClick={onCancel} sx={{ flex: 1 }}>
        Cancel
      </Button>
      <Button
        size="large"
        variant="contained"
        color={danger ? "error" : "primary"}
        onClick={onConfirm}
        sx={{ flex: 1 }}
      >
        {confirmLabel}
      </Button>
    </DialogActions>
  </Dialog>
);

export default ConfirmDialog;
