// Kiosk objective switcher: one big card per turret slot with a clear
// "INSERTED" state (Formlabs-style), tap the other card to swap. Slots are
// 0-based, matching ObjectiveController.moveToObjective.
import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Card,
  CardActionArea,
  Chip,
  CircularProgress,
  Typography,
} from "@mui/material";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import { enqueueSnackbar } from "notistack";

import MobilePage from "../components/MobilePage";
import PlaceholderImage from "../components/PlaceholderImage";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice";
import fetchObjectiveControllerGetStatus from "../../middleware/fetchObjectiveControllerGetStatus";
import apiObjectiveControllerMoveToObjective from "../../backendapi/apiObjectiveControllerMoveToObjective";

const SlotCard = ({ slot, name, magnification, na, pixelSize, configured, inserted, pending, disabled, onSelect }) => (
  <Card
    variant="outlined"
    sx={{
      flex: 1,
      minWidth: 260,
      maxWidth: 420,
      borderRadius: 3,
      borderWidth: 2,
      borderColor: inserted ? "primary.main" : "divider",
      opacity: configured ? 1 : 0.45,
      position: "relative",
    }}
  >
    <CardActionArea
      disabled={disabled || !configured || inserted}
      onClick={onSelect}
      sx={{ p: 2.5, height: "100%" }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
        <Typography variant="h6" sx={{ flex: 1 }}>
          Slot {slot + 1}
          {name ? ` · ${name}` : ""}
        </Typography>
        {inserted && (
          <Chip
            icon={<CheckCircleRoundedIcon />}
            label="INSERTED"
            color="primary"
            size="small"
          />
        )}
        {!configured && <Chip label="Not configured" size="small" variant="outlined" />}
      </Box>

      {/* Placeholder for a photo/rendering of the mounted objective */}
      <PlaceholderImage label={`Objective photo — slot ${slot + 1}`} height={120} sx={{ mb: 1.5 }} />

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", color: "text.secondary" }}>
        <Typography variant="body2">
          Mag: <strong>{magnification ? `${magnification}×` : "—"}</strong>
        </Typography>
        <Typography variant="body2">
          NA: <strong>{na || "—"}</strong>
        </Typography>
        <Typography variant="body2">
          Pixel: <strong>{pixelSize ? `${pixelSize} µm` : "—"}</strong>
        </Typography>
      </Box>

      {!inserted && configured && !disabled && (
        <Typography variant="body2" color="primary" sx={{ mt: 1.5, fontWeight: 700 }}>
          Tap to insert
        </Typography>
      )}
    </CardActionArea>

    {pending && (
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "rgba(0,0,0,0.55)",
          borderRadius: "inherit",
        }}
      >
        <CircularProgress />
      </Box>
    )}
  </Card>
);

const MobileObjectivePage = () => {
  const dispatch = useDispatch();
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const [pendingSlot, setPendingSlot] = useState(null);

  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
  }, [dispatch]);

  const switchTo = async (slot) => {
    setPendingSlot(slot);
    try {
      await apiObjectiveControllerMoveToObjective(slot);
      // sigObjectiveChanged refreshes the slice too; this fetch covers setups
      // where the signal is not emitted.
      fetchObjectiveControllerGetStatus(dispatch);
      enqueueSnackbar(`Objective slot ${slot + 1} inserted`, { variant: "success" });
    } catch (error) {
      enqueueSnackbar(error?.message || "Objective switch failed", { variant: "error" });
    } finally {
      setPendingSlot(null);
    }
  };

  const currentSlot = objectiveState.currentObjective;
  const switching = pendingSlot !== null;

  return (
    <MobilePage
      title="Objective"
      subtitle={
        currentSlot != null
          ? `Current: ${objectiveState.availableObjectivesNames?.[currentSlot] || `slot ${currentSlot + 1}`}`
          : "Objective position unknown — switch once to calibrate"
      }
    >
      <Box sx={{ display: "flex", gap: 2.5, flexWrap: "wrap" }}>
        {[0, 1].map((slot) => (
          <SlotCard
            key={slot}
            slot={slot}
            name={objectiveState.availableObjectivesNames?.[slot]}
            magnification={objectiveState.availableObjectiveMagnifications?.[slot]}
            na={objectiveState.availableObjectiveNAs?.[slot]}
            pixelSize={objectiveState.availableObjectivePixelSizes?.[slot]}
            configured={objectiveState.slotConfigured?.[slot] !== false}
            inserted={currentSlot === slot}
            pending={pendingSlot === slot}
            disabled={switching}
            onSelect={() => switchTo(slot)}
          />
        ))}
      </Box>
      <Typography variant="caption" color="text.disabled" sx={{ display: "block", mt: 2.5 }}>
        Switching moves the turret and adjusts Z for the stored parfocality offset.
        Keep hands clear of the objective area while switching.
      </Typography>
    </MobilePage>
  );
};

export default MobileObjectivePage;
