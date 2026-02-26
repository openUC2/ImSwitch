import React, { useRef, useEffect, useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  IconButton,
  Chip,
  LinearProgress,
  Slider,
  Alert,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import UndoIcon from "@mui/icons-material/Undo";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import RefreshIcon from "@mui/icons-material/Refresh";

import * as overviewRegSlice from "../state/slices/OverviewRegistrationSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";

import apiGetOverviewRegistrationConfig from "../backendapi/apiGetOverviewRegistrationConfig.js";
import apiSnapOverviewImage from "../backendapi/apiSnapOverviewImage.js";
import apiRegisterOverviewSlide from "../backendapi/apiRegisterOverviewSlide.js";
import apiGetOverviewOverlayData from "../backendapi/apiGetOverviewOverlayData.js";
import apiRefreshOverviewSlideImage from "../backendapi/apiRefreshOverviewSlideImage.js";

//##################################################################################
// Wizard step labels
const WIZARD_STEPS = [
  "Select Slide",
  "Align & Snap",
  "Mark Corners",
  "Save Registration",
];

// Corner convention labels
const CORNER_LABELS = [
  "1: Top-Left",
  "2: Top-Right",
  "3: Bottom-Right",
  "4: Bottom-Left",
];

const CORNER_COLORS = ["#ff4444", "#44ff44", "#4444ff", "#ffaa00"];

//##################################################################################
const OverviewRegistrationWizard = () => {
  const dispatch = useDispatch();
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const streamImgRef = useRef(null);

  // Redux state
  const regState = useSelector(
    overviewRegSlice.getOverviewRegistrationState
  );
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  // Local state for stream
  const [streamUrl, setStreamUrl] = useState("");
  const [streamError, setStreamError] = useState(false);

  // Local state for image natural dimensions (for coordinate mapping)
  const [imgNaturalSize, setImgNaturalSize] = useState({ w: 0, h: 0 });

  //##################################################################################
  // Load config on wizard open
  useEffect(() => {
    if (regState.wizardOpen) {
      loadConfig();
    }
  }, [regState.wizardOpen]);

  //##################################################################################
  // Build MJPEG stream URL
  useEffect(() => {
    if (regState.wizardOpen && regState.cameraAvailable) {
      const base = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
      setStreamUrl(
        `${base}/PixelCalibrationController/overviewStream?startStream=true`
      );
      setStreamError(false);
    }
  }, [
    regState.wizardOpen,
    regState.cameraAvailable,
    connectionSettings.ip,
    connectionSettings.apiPort,
  ]);

  //##################################################################################
  const loadConfig = async () => {
    dispatch(overviewRegSlice.setIsLoading(true));
    try {
      const config = await apiGetOverviewRegistrationConfig(
        regState.layoutName
      );
      dispatch(overviewRegSlice.setConfig(config));
      dispatch(overviewRegSlice.setError(null));
    } catch (e) {
      dispatch(
        overviewRegSlice.setError("Failed to load wizard config: " + e.message)
      );
    }
    dispatch(overviewRegSlice.setIsLoading(false));
  };

  //##################################################################################
  const handleClose = () => {
    dispatch(overviewRegSlice.setWizardOpen(false));
    dispatch(overviewRegSlice.resetWizard());
  };

  //##################################################################################
  const handleSlotSelect = (slotId) => {
    dispatch(overviewRegSlice.setCurrentSlotId(slotId));
    dispatch(overviewRegSlice.clearSnapshot());
    dispatch(overviewRegSlice.setWizardStep(1)); // Go to Align & Snap
  };

  //##################################################################################
  const handleSnapImage = async () => {
    dispatch(overviewRegSlice.setIsLoading(true));
    try {
      const result = await apiSnapOverviewImage(
        regState.currentSlotId,
        regState.cameraName
      );
      dispatch(overviewRegSlice.setSnapshot(result));
      dispatch(overviewRegSlice.setCornerPickingActive(true));
      dispatch(overviewRegSlice.resetPickedCorners());
      dispatch(overviewRegSlice.setWizardStep(2)); // Go to Mark Corners
    } catch (e) {
      dispatch(
        overviewRegSlice.setError("Failed to snap image: " + e.message)
      );
    }
    dispatch(overviewRegSlice.setIsLoading(false));
  };

  //##################################################################################
  // Handle click on snapshot image for corner picking
  const handleImageClick = useCallback(
    (e) => {
      if (!regState.cornerPickingActive) return;
      if (regState.pickedCorners.length >= 4) return;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();

      // Get click position relative to canvas
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      // Map display coordinates to image pixel coordinates
      const scaleX = imgNaturalSize.w / rect.width;
      const scaleY = imgNaturalSize.h / rect.height;

      const imgX = clickX * scaleX;
      const imgY = clickY * scaleY;

      dispatch(overviewRegSlice.addPickedCorner({ x: imgX, y: imgY }));
    },
    [
      regState.cornerPickingActive,
      regState.pickedCorners.length,
      imgNaturalSize,
    ]
  );

  //##################################################################################
  // Draw corner picking overlay on canvas
  useEffect(() => {
    if (regState.wizardStep !== 2) return;
    drawCornerOverlay();
  }, [regState.pickedCorners, regState.snapshotImage, regState.wizardStep]);

  const drawCornerOverlay = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Load snapshot image
    const img = new Image();
    img.onload = () => {
      // Set canvas size to match display area
      const parent = canvas.parentElement;
      const maxWidth = parent ? parent.clientWidth : 640;
      const aspect = img.naturalHeight / img.naturalWidth;
      canvas.width = maxWidth;
      canvas.height = maxWidth * aspect;

      setImgNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });

      // Draw image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const scaleX = canvas.width / img.naturalWidth;
      const scaleY = canvas.height / img.naturalHeight;

      // Draw guide: current slot rectangle projected roughly in center
      // (just a visual hint, not exact projection)
      drawGuideHint(ctx, canvas.width, canvas.height);

      // Draw picked corners
      regState.pickedCorners.forEach((pt, idx) => {
        const px = pt.x * scaleX;
        const py = pt.y * scaleY;

        // Crosshair
        ctx.strokeStyle = CORNER_COLORS[idx];
        ctx.lineWidth = 2;
        const size = 15;
        ctx.beginPath();
        ctx.moveTo(px - size, py);
        ctx.lineTo(px + size, py);
        ctx.moveTo(px, py - size);
        ctx.lineTo(px, py + size);
        ctx.stroke();

        // Circle
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, 2 * Math.PI);
        ctx.stroke();

        // Label
        ctx.fillStyle = CORNER_COLORS[idx];
        ctx.font = "bold 14px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${idx + 1}`, px + 12, py - 4);
      });

      // Draw connecting polygon if we have points
      if (regState.pickedCorners.length >= 2) {
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        const first = regState.pickedCorners[0];
        ctx.moveTo(first.x * scaleX, first.y * scaleY);
        for (let i = 1; i < regState.pickedCorners.length; i++) {
          const pt = regState.pickedCorners[i];
          ctx.lineTo(pt.x * scaleX, pt.y * scaleY);
        }
        if (regState.pickedCorners.length === 4) {
          ctx.closePath();
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw next corner hint
      if (regState.pickedCorners.length < 4) {
        const nextLabel = CORNER_LABELS[regState.pickedCorners.length];
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(0, canvas.height - 30, canvas.width, 30);
        ctx.fillStyle = "#ffffff";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          `Click to place: ${nextLabel}`,
          canvas.width / 2,
          canvas.height - 15
        );
      }
    };
    img.src = `data:${regState.snapshotMimeType};base64,${regState.snapshotImage}`;
  };

  //##################################################################################
  const drawGuideHint = (ctx, canvasW, canvasH) => {
    // Draw corner numbering guide in top-right
    const guideW = 80;
    const guideH = 50;
    const guideX = canvasW - guideW - 10;
    const guideY = 10;

    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(guideX, guideY, guideW, guideH);
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.lineWidth = 1;
    ctx.strokeRect(guideX + 5, guideY + 5, guideW - 10, guideH - 10);

    // Corner numbers
    const corners = [
      { x: guideX + 10, y: guideY + 15, label: "1" },
      { x: guideX + guideW - 15, y: guideY + 15, label: "2" },
      { x: guideX + guideW - 15, y: guideY + guideH - 10, label: "3" },
      { x: guideX + 10, y: guideY + guideH - 10, label: "4" },
    ];
    corners.forEach((c, i) => {
      ctx.fillStyle = CORNER_COLORS[i];
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(c.label, c.x, c.y);
    });
  };

  //##################################################################################
  const handleConfirmCorners = async () => {
    if (regState.pickedCorners.length !== 4) return;

    // Find current slot definition
    const slot = regState.slots.find(
      (s) => s.slotId === regState.currentSlotId
    );
    if (!slot) {
      dispatch(overviewRegSlice.setError("Slot definition not found"));
      return;
    }

    dispatch(overviewRegSlice.setIsLoading(true));
    try {
      const result = await apiRegisterOverviewSlide({
        cameraName: regState.cameraName,
        layoutName: regState.layoutName,
        slotId: regState.currentSlotId,
        slotName: slot.name,
        snapshotId: regState.snapshotId,
        snapshotTimestamp: regState.snapshotTimestamp,
        imageWidth: regState.snapshotWidth,
        imageHeight: regState.snapshotHeight,
        cornersPx: regState.pickedCorners,
        slotStageCorners: slot.corners.map((c) => ({ x: c.x, y: c.y })),
      });

      dispatch(overviewRegSlice.setLastRegistrationResult(result));
      dispatch(
        overviewRegSlice.updateSlideStatus({
          slotId: regState.currentSlotId,
          data: {
            complete: true,
            slotId: regState.currentSlotId,
            slotName: slot.name,
            snapshotId: regState.snapshotId,
            reprojectionError: result.reprojectionError,
            hasOverlayImage: result.hasOverlayImage,
            updatedAt: result.createdAt,
          },
        })
      );
      dispatch(overviewRegSlice.setWizardStep(3)); // Go to Save/Summary
    } catch (e) {
      dispatch(
        overviewRegSlice.setError("Registration failed: " + e.message)
      );
    }
    dispatch(overviewRegSlice.setIsLoading(false));
  };

  //##################################################################################
  const handleContinueToNextSlide = () => {
    const currentIdx = parseInt(regState.currentSlotId);
    const totalSlots = regState.slots.length;

    if (currentIdx < totalSlots) {
      // Move to next slide
      dispatch(overviewRegSlice.setCurrentSlotId(String(currentIdx + 1)));
      dispatch(overviewRegSlice.clearSnapshot());
      dispatch(overviewRegSlice.setWizardStep(1));
    } else {
      // All done – show finish
      dispatch(overviewRegSlice.setWizardStep(4));
    }
  };

  //##################################################################################
  const handleFinish = async () => {
    // Load overlay data for WellSelector
    try {
      const overlayData = await apiGetOverviewOverlayData(
        regState.cameraName,
        regState.layoutName
      );
      dispatch(overviewRegSlice.setOverlayData(overlayData));
      dispatch(overviewRegSlice.setOverlayEnabled(true));
    } catch (e) {
      console.warn("Failed to load overlay data:", e);
    }
    dispatch(overviewRegSlice.setWizardOpen(false));
    dispatch(overviewRegSlice.setWizardStep(0));
  };

  //##################################################################################
  const handleRefreshSlide = async (slotId) => {
    dispatch(overviewRegSlice.setIsLoading(true));
    try {
      await apiRefreshOverviewSlideImage(
        slotId,
        regState.cameraName,
        regState.layoutName
      );
      // Reload overlay data
      const overlayData = await apiGetOverviewOverlayData(
        regState.cameraName,
        regState.layoutName
      );
      dispatch(overviewRegSlice.setOverlayData(overlayData));
    } catch (e) {
      dispatch(
        overviewRegSlice.setError("Refresh failed: " + e.message)
      );
    }
    dispatch(overviewRegSlice.setIsLoading(false));
  };

  //##################################################################################
  const getCompletedCount = () => {
    return Object.values(regState.slideStatus).filter((s) => s.complete).length;
  };

  //##################################################################################
  // Render helpers

  const renderSlideSelector = () => (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Select a slide slot to register. Complete all 4 slides for full
        coverage.
      </Typography>
      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 2 }}>
        {regState.slots.map((slot) => {
          const status = regState.slideStatus[slot.slotId];
          const isComplete = status?.complete;
          return (
            <Box
              key={slot.slotId}
              sx={{
                border: "2px solid",
                borderColor: isComplete ? "success.main" : "grey.500",
                borderRadius: 2,
                p: 2,
                minWidth: 120,
                textAlign: "center",
                cursor: "pointer",
                "&:hover": { borderColor: "primary.main", bgcolor: "action.hover" },
              }}
              onClick={() => handleSlotSelect(slot.slotId)}
            >
              {isComplete ? (
                <CheckCircleIcon color="success" />
              ) : (
                <RadioButtonUncheckedIcon color="disabled" />
              )}
              <Typography variant="body1" sx={{ fontWeight: "bold", mt: 1 }}>
                {slot.name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Slot {slot.slotId}
              </Typography>
              {isComplete && (
                <Box sx={{ mt: 1 }}>
                  <IconButton
                    size="small"
                    title="Snap new image (reuse registration)"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRefreshSlide(slot.slotId);
                    }}
                  >
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Box>
              )}
            </Box>
          );
        })}
      </Box>
      <Typography variant="body2" color="text.secondary">
        Progress: {getCompletedCount()} / {regState.slots.length} slides
        registered
      </Typography>
      {getCompletedCount() === regState.slots.length && (
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" color="success" onClick={handleFinish}>
            Finish & Enable Overlay
          </Button>
        </Box>
      )}
    </Box>
  );

  //##################################################################################
  const renderLiveAlignStep = () => {
    const slot = regState.slots.find(
      (s) => s.slotId === regState.currentSlotId
    );
    return (
      <Box>
        <Typography variant="subtitle1" sx={{ mb: 1 }}>
          Align <strong>{slot?.name || `Slide ${regState.currentSlotId}`}</strong>{" "}
          in the overview camera view, then snap an image.
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Manually move the XYZ stage until the physical slide appears centered
          and aligned in the live stream below. The guide rectangle shows the
          approximate target area.
        </Typography>

        {/* Live MJPEG stream */}
        <Box
          sx={{
            position: "relative",
            width: "100%",
            maxWidth: 640,
            margin: "0 auto",
            border: "1px solid #444",
            borderRadius: 1,
            overflow: "hidden",
            bgcolor: "#000",
          }}
        >
          {streamUrl && !streamError ? (
            <img
              ref={streamImgRef}
              src={streamUrl}
              alt="Overview camera live stream"
              style={{ width: "100%", display: "block" }}
              onError={() => setStreamError(true)}
            />
          ) : (
            <Box
              sx={{
                width: "100%",
                height: 300,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Typography color="error">
                {streamError
                  ? "Stream connection failed. Check overview camera."
                  : "No camera stream available."}
              </Typography>
            </Box>
          )}

          {/* Guide rectangle overlay (CSS overlay on live stream) */}
          <Box
            sx={{
              position: "absolute",
              top: "15%",
              left: "15%",
              width: "70%",
              height: "70%",
              border: "2px dashed rgba(255,255,0,0.7)",
              borderRadius: 1,
              pointerEvents: "none",
              display: "flex",
              alignItems: "flex-start",
              justifyContent: "center",
            }}
          >
            <Typography
              sx={{
                color: "rgba(255,255,0,0.9)",
                bgcolor: "rgba(0,0,0,0.5)",
                px: 1,
                py: 0.5,
                fontSize: 12,
                borderRadius: "0 0 4px 4px",
              }}
            >
              Guide: {slot?.name || "Slide"} – align slide here
            </Typography>
            {/* Corner number hints */}
            <Box sx={{ position: "absolute", top: 2, left: 4 }}>
              <Typography sx={{ color: CORNER_COLORS[0], fontSize: 11, fontWeight: "bold" }}>1</Typography>
            </Box>
            <Box sx={{ position: "absolute", top: 2, right: 4 }}>
              <Typography sx={{ color: CORNER_COLORS[1], fontSize: 11, fontWeight: "bold" }}>2</Typography>
            </Box>
            <Box sx={{ position: "absolute", bottom: 2, right: 4 }}>
              <Typography sx={{ color: CORNER_COLORS[2], fontSize: 11, fontWeight: "bold" }}>3</Typography>
            </Box>
            <Box sx={{ position: "absolute", bottom: 2, left: 4 }}>
              <Typography sx={{ color: CORNER_COLORS[3], fontSize: 11, fontWeight: "bold" }}>4</Typography>
            </Box>
          </Box>
        </Box>

        <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<CameraAltIcon />}
            onClick={handleSnapImage}
            disabled={regState.isLoading}
          >
            Snap Image
          </Button>
          <Button
            variant="outlined"
            onClick={() => dispatch(overviewRegSlice.setWizardStep(0))}
          >
            Back to Slide Selection
          </Button>
        </Box>
      </Box>
    );
  };

  //##################################################################################
  const renderCornerPickStep = () => (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: 1 }}>
        Click the 4 slide corners in order: TL → TR → BR → BL
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Click directly on the snapped image to place corner markers. The
        corners must follow the order shown in the guide (top-right corner of
        image).
      </Typography>

      {/* Corner status chips */}
      <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
        {CORNER_LABELS.map((label, idx) => (
          <Chip
            key={idx}
            label={label}
            size="small"
            sx={{
              bgcolor:
                idx < regState.pickedCorners.length
                  ? CORNER_COLORS[idx]
                  : undefined,
              color:
                idx < regState.pickedCorners.length ? "#000" : undefined,
              fontWeight:
                idx === regState.pickedCorners.length ? "bold" : "normal",
              border:
                idx === regState.pickedCorners.length
                  ? "2px solid white"
                  : undefined,
            }}
          />
        ))}
      </Box>

      {/* Snapshot canvas for corner picking */}
      <Box
        sx={{
          width: "100%",
          maxWidth: 640,
          margin: "0 auto",
          border: "1px solid #444",
          borderRadius: 1,
          overflow: "hidden",
          cursor:
            regState.pickedCorners.length < 4 ? "crosshair" : "default",
        }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleImageClick}
          style={{ width: "100%", display: "block" }}
        />
      </Box>

      {/* Actions */}
      <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
        <Button
          variant="outlined"
          startIcon={<UndoIcon />}
          onClick={() => dispatch(overviewRegSlice.undoLastCorner())}
          disabled={regState.pickedCorners.length === 0}
        >
          Undo
        </Button>
        <Button
          variant="outlined"
          startIcon={<RestartAltIcon />}
          onClick={() => dispatch(overviewRegSlice.resetPickedCorners())}
          disabled={regState.pickedCorners.length === 0}
        >
          Reset
        </Button>
        <Button
          variant="contained"
          color="primary"
          startIcon={<CheckCircleIcon />}
          onClick={handleConfirmCorners}
          disabled={
            regState.pickedCorners.length !== 4 || regState.isLoading
          }
        >
          Confirm Corners
        </Button>
        <Button
          variant="outlined"
          onClick={() => {
            dispatch(overviewRegSlice.clearSnapshot());
            dispatch(overviewRegSlice.setWizardStep(1));
          }}
        >
          Re-snap
        </Button>
      </Box>
    </Box>
  );

  //##################################################################################
  const renderSaveStep = () => {
    const result = regState.lastRegistrationResult;
    return (
      <Box>
        <Alert severity="success" sx={{ mb: 2 }}>
          Registration saved for Slide {regState.currentSlotId}!
        </Alert>
        {result && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2">
              Reprojection error:{" "}
              <strong>{result.reprojectionError?.toFixed(2)} µm</strong>
            </Typography>
            <Typography variant="body2">
              Overlay image:{" "}
              {result.hasOverlayImage ? "✓ Created" : "✗ Not created"}
            </Typography>
          </Box>
        )}
        <Box sx={{ display: "flex", gap: 1 }}>
          <Button variant="contained" onClick={handleContinueToNextSlide}>
            {parseInt(regState.currentSlotId) < regState.slots.length
              ? `Continue to Slide ${parseInt(regState.currentSlotId) + 1}`
              : "Finish Wizard"}
          </Button>
          <Button
            variant="outlined"
            onClick={() => {
              dispatch(overviewRegSlice.clearSnapshot());
              dispatch(overviewRegSlice.setWizardStep(0));
            }}
          >
            Back to Overview
          </Button>
        </Box>
      </Box>
    );
  };

  //##################################################################################
  const renderFinishStep = () => (
    <Box>
      <Alert severity="success" sx={{ mb: 2 }}>
        All slides registered! ({getCompletedCount()}/{regState.slots.length})
      </Alert>
      <Typography variant="body2" sx={{ mb: 2 }}>
        The overlay will be enabled on the WellSelector canvas. You can toggle
        visibility and adjust opacity from the WellPlate controls.
      </Typography>
      <Typography variant="body2" sx={{ mb: 2 }}>
        To update a single slide overlay later, use the refresh button (↻)
        on the slide card without redoing the full wizard.
      </Typography>
      <Button variant="contained" color="success" onClick={handleFinish}>
        Finish & Enable Overlay
      </Button>
    </Box>
  );

  //##################################################################################
  const renderStepContent = () => {
    switch (regState.wizardStep) {
      case 0:
        return renderSlideSelector();
      case 1:
        return renderLiveAlignStep();
      case 2:
        return renderCornerPickStep();
      case 3:
        return renderSaveStep();
      case 4:
        return renderFinishStep();
      default:
        return renderSlideSelector();
    }
  };

  //##################################################################################
  // Map wizard step to stepper index
  const getActiveStepperIndex = () => {
    if (regState.wizardStep === 0) return 0;
    if (regState.wizardStep === 1) return 1;
    if (regState.wizardStep === 2) return 2;
    if (regState.wizardStep >= 3) return 3;
    return 0;
  };

  //##################################################################################
  return (
    <Dialog
      open={regState.wizardOpen}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: "60vh", bgcolor: "#1e1e1e", color: "#fff" },
      }}
    >
      <DialogTitle
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Box>
          <Typography variant="h6">Overview Overlay Wizard</Typography>
          <Typography variant="caption" color="text.secondary">
            {regState.layoutName} – Manual slide registration
          </Typography>
        </Box>
        <IconButton onClick={handleClose} sx={{ color: "#fff" }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers sx={{ p: 3 }}>
        {/* Stepper */}
        <Stepper
          activeStep={getActiveStepperIndex()}
          sx={{ mb: 3 }}
          alternativeLabel
        >
          {WIZARD_STEPS.map((label) => (
            <Step key={label}>
              <StepLabel
                sx={{
                  "& .MuiStepLabel-label": { color: "#ccc", fontSize: 12 },
                  "& .MuiStepLabel-label.Mui-active": { color: "#fff" },
                }}
              >
                {label}
              </StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Loading bar */}
        {regState.isLoading && <LinearProgress sx={{ mb: 2 }} />}

        {/* Error */}
        {regState.error && (
          <Alert
            severity="error"
            sx={{ mb: 2 }}
            onClose={() => dispatch(overviewRegSlice.setError(null))}
          >
            {regState.error}
          </Alert>
        )}

        {/* Camera not available warning */}
        {!regState.cameraAvailable && regState.wizardOpen && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Overview camera not detected. The wizard requires an observation
            camera configured in SetupInfo.PixelCalibration.ObservationCamera.
          </Alert>
        )}

        {/* Step content */}
        {renderStepContent()}
      </DialogContent>
    </Dialog>
  );
};

//##################################################################################
export default OverviewRegistrationWizard;
