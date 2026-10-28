// src/components/ObjectiveController.js
// Objective turret page.
//
//   Turret        side view of the revolver + the two "put in beam" buttons
//   Live & jog    camera stream with turret (A) / focus (Z) step controls,
//                 so a stored position can be set while looking at the sample
//   Slot cards    per-slot optics and stored positions (A, par-focal Z)
//   Maintenance   homing, switch speed, calibration wizard
//
// All backend calls go through useObjectiveActions (shared with the compact
// ObjectiveSwitcher in the Live View). Anything that moves hardware or
// overwrites a stored value without showing it first asks for confirmation.
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Paper,
  TextField,
  Typography,
} from "@mui/material";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import AutoFixHighRoundedIcon from "@mui/icons-material/AutoFixHighRounded";
import HomeRoundedIcon from "@mui/icons-material/HomeRounded";
import SpeedRoundedIcon from "@mui/icons-material/SpeedRounded";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import ObjectiveCalibrationWizard from "./ObjectiveCalibrationWizard";
import ObjectiveTurretView from "./ObjectiveTurretView";
import ObjectiveSlotButtons from "./objective/ObjectiveSlotButtons.jsx";
import ObjectiveSlotCard from "./objective/ObjectiveSlotCard.jsx";
import ObjectiveJogPanel from "./objective/ObjectiveJogPanel.jsx";
import useObjectiveActions from "./objective/useObjectiveActions.js";
import { getSlots, fmt, opticsLabel } from "./objective/objectiveSlots.js";
import ConfirmDialog from "../mobile/components/ConfirmDialog.jsx";
import * as positionSlice from "../state/slices/PositionSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import useDeviceProfile from "../hooks/useDeviceProfile";

const Section = ({ title, action, children, sx }) => (
  <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, minWidth: 0, ...sx }}>
    {(title || action) && (
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 700, flex: 1 }}>
          {title}
        </Typography>
        {action}
      </Box>
    )}
    {children}
  </Paper>
);

const ObjectiveController = () => {
  const { ip: hostIP, apiPort: hostPort } = useSelector(getConnectionSettingsState);
  const position = useSelector(positionSlice.getPositionState);
  const { compactLayout } = useDeviceProfile();
  const actions = useObjectiveActions();
  const { objectiveState: obj, pendingSlot, isSwitching, refresh } = actions;

  const [withZ, setWithZ] = useState(true);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [speedDraft, setSpeedDraft] = useState("");

  useEffect(() => {
    refresh();
  }, [hostIP, hostPort, refresh]);

  const slots = getSlots(obj);
  const current =
    obj.currentObjective === 0 || obj.currentObjective === 1
      ? obj.currentObjective
      : null;
  const hasMotor = obj.hasMotor !== false;
  const moveSpeed = obj.moveSpeed ?? 20000;

  const runConfirmed = async () => {
    const pending = confirm;
    setConfirm(null);
    await pending?.onConfirm?.();
  };

  // "Use current": read the live position first so the dialog can show the
  // exact value that will be stored.
  const askUseCurrent = async (slotIndex, kind) => {
    let value;
    try {
      const az = await actions.readCurrentAZ();
      value = kind === "a" ? az.a : az.z;
    } catch (e) {
      actions.notify(`Could not read the stage position: ${e.message || e}`, "error");
      return;
    }
    const slotLabel = `slot ${slotIndex + 1}`;
    setConfirm(
      kind === "a"
        ? {
            title: `Store turret position for ${slotLabel}?`,
            text: `The current revolver position A = ${fmt(value)} µm becomes the position that puts ${slotLabel} in the beam.`,
            confirmLabel: "Store A",
            onConfirm: () => actions.storePosition(`x${slotIndex}`, value),
          }
        : {
            title: `Store focus for ${slotLabel}?`,
            text: `The current Z = ${fmt(value)} µm becomes the par-focal focus of ${slotLabel}. With Z leveling on, switching objectives moves Z by the difference between the two stored foci.`,
            confirmLabel: "Store Z",
            onConfirm: () => actions.storePosition(`z${slotIndex}`, value),
          },
    );
  };

  const askHome = () =>
    setConfirm({
      title: "Home the objective turret?",
      text: "The revolver (A axis) drives to its endstop and re-references its position. Make sure nothing obstructs the turret.",
      confirmLabel: "Home turret",
      danger: true,
      onConfirm: actions.home,
    });

  const speedValue = Number(speedDraft);
  const speedValid = speedDraft !== "" && Number.isFinite(speedValue) && speedValue > 0;
  const askSpeed = () =>
    setConfirm({
      title: "Change the switch speed?",
      text: `The turret will move at ${fmt(speedValue, 0)} steps/s when switching objectives (currently ${fmt(moveSpeed, 0)}).`,
      confirmLabel: "Set speed",
      onConfirm: async () => {
        await actions.saveMoveSpeed(speedValue);
        setSpeedDraft("");
      },
    });

  const currentSlot = current !== null ? slots[current] : null;
  const status = isSwitching ? (
    <Chip
      color="warning"
      icon={<CircularProgress size={14} color="inherit" />}
      label={`Switching to slot ${pendingSlot + 1}…`}
    />
  ) : currentSlot ? (
    <Chip
      color="primary"
      label={`${currentSlot.label} in beam · ${opticsLabel(currentSlot)}`}
    />
  ) : (
    <Chip variant="outlined" label="Objective unknown" />
  );

  return (
    <Box
      sx={{
        width: "100%",
        maxWidth: 1500,
        mx: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 2,
        pb: 2,
      }}
    >
      {/* header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mr: 1 }}>
          Objective turret
        </Typography>
        {status}
        <Box sx={{ flex: 1 }} />
        <Button startIcon={<HelpOutlineRoundedIcon />} onClick={() => setShowHelp((v) => !v)}>
          How it works
        </Button>
        <Button startIcon={<RefreshRoundedIcon />} onClick={refresh}>
          Refresh
        </Button>
        <Button
          variant="contained"
          color="success"
          startIcon={<AutoFixHighRoundedIcon />}
          onClick={() => setWizardOpen(true)}
        >
          Calibration wizard
        </Button>
      </Box>

      <Collapse in={showHelp} unmountOnExit>
        <Alert severity="info" onClose={() => setShowHelp(false)}>
          Two objectives sit on a motorised revolver. Switching slides the
          revolver along its <strong>A axis</strong> to the slot's stored{" "}
          <strong>turret position</strong>; with <strong>Z leveling</strong> on,
          the focus also moves by the difference between the two stored{" "}
          <strong>par-focal Z</strong> values, so the sample stays in focus. To
          set them: switch to a slot, centre it with the turret controls, focus
          on the sample, then press <em>Use current</em> for A and Z on that
          slot's card — or let the calibration wizard walk you through it.
        </Alert>
      </Collapse>

      {/* turret + live view */}
      <Box
        sx={{
          display: "grid",
          gap: 2,
          gridTemplateColumns: {
            xs: "minmax(0, 1fr)",
            lg: "minmax(0, 5fr) minmax(0, 6fr)",
          },
        }}
      >
        <Section title="Turret">
          {/* On a short screen the to-scale drawing would fill the view. */}
          <Box sx={{ maxWidth: compactLayout ? 420 : "none", mx: "auto" }}>
            <ObjectiveTurretView />
          </Box>
          <Box sx={{ mt: 2 }}>
            <ObjectiveSlotButtons
              slots={slots}
              currentSlot={current}
              pendingSlot={pendingSlot}
              onSelect={(slot) => actions.switchTo(slot, { withZ })}
              withZ={withZ}
              onWithZChange={setWithZ}
            />
          </Box>
        </Section>

        <Section title="Live view">
          <Box
            sx={{
              height: compactLayout ? 240 : 340,
              mb: 2,
              borderRadius: 1,
              overflow: "hidden",
            }}
          >
            <LiveViewControlWrapper fill />
          </Box>
          <ObjectiveJogPanel position={position} hasMotor={hasMotor} onJog={actions.jog} />
        </Section>
      </Box>

      {/* per-slot configuration */}
      <Box
        sx={{
          display: "grid",
          gap: 2,
          gridTemplateColumns: { xs: "minmax(0, 1fr)", md: "repeat(2, minmax(0, 1fr))" },
        }}
      >
        {slots.map((slot) => (
          <ObjectiveSlotCard
            key={slot.index}
            slot={slot}
            inBeam={current === slot.index && !isSwitching}
            hasMotor={hasMotor}
            onSaveOptics={(optics) => actions.saveOptics(slot.index, optics)}
            onStorePosition={(kind, value) =>
              actions.storePosition(`${kind === "a" ? "x" : "z"}${slot.index}`, value)
            }
            onUseCurrent={(kind) => askUseCurrent(slot.index, kind)}
          />
        ))}
      </Box>

      {/* maintenance */}
      <Section title="Maintenance">
        <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "flex-start" }}>
          {hasMotor && (
            <Box sx={{ flex: "1 1 260px" }}>
              <Typography variant="body2" sx={{ color: "text.secondary", mb: 1 }}>
                Re-reference the revolver against its endstop, e.g. after a
                power cycle or a stall.
              </Typography>
              <Button variant="outlined" startIcon={<HomeRoundedIcon />} onClick={askHome}>
                Home turret
              </Button>
            </Box>
          )}
          <Box sx={{ flex: "1 1 320px" }}>
            <Typography variant="body2" sx={{ color: "text.secondary", mb: 1 }}>
              Switch speed — currently <strong>{fmt(moveSpeed, 0)}</strong> steps/s.
            </Typography>
            <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
              <TextField
                label="New speed"
                type="number"
                size="small"
                value={speedDraft}
                placeholder={String(moveSpeed)}
                onChange={(e) => setSpeedDraft(e.target.value)}
                inputProps={{
                  min: 1,
                  step: 1000,
                  "data-keypad-presets": "5000,10000,20000,40000",
                }}
                InputProps={{ endAdornment: <Typography variant="caption">steps/s</Typography> }}
                sx={{ width: 200 }}
              />
              <Button
                variant="contained"
                startIcon={<SpeedRoundedIcon />}
                onClick={askSpeed}
                disabled={!speedValid}
              >
                Apply
              </Button>
            </Box>
          </Box>
        </Box>
      </Section>

      <ConfirmDialog
        open={Boolean(confirm)}
        title={confirm?.title}
        text={confirm?.text}
        confirmLabel={confirm?.confirmLabel}
        danger={confirm?.danger}
        onCancel={() => setConfirm(null)}
        onConfirm={runConfirmed}
      />

      <ObjectiveCalibrationWizard open={wizardOpen} onClose={() => setWizardOpen(false)} />
    </Box>
  );
};

export default ObjectiveController;
