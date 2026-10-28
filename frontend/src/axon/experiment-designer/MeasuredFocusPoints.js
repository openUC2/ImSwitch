// Measured focus points: one expandable table per region, with go-to,
// nudge-Z, re-measure and delete. Lifted verbatim out of FocusMapDimension —
// it was already a component in all but name, and at ~430 lines it was a third
// of that file.
import React, { useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Chip,
  CircularProgress,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import RefreshIcon from "@mui/icons-material/Refresh";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";

import { useTheme } from "@mui/material/styles";

import * as focusMapSlice from "../../state/slices/FocusMapSlice";
import { colorForValue } from "./FocusMapVisualization";

// Rows rendered per region before "show more" — these tables get long.
const POINTS_PAGE_SIZE = 20;

const MeasuredFocusPoints = ({
  groupEntries,
  ui,
  showMeasuredPoints,
  setShowMeasuredPoints,
  highlightedPoint,
  dispatch,
  handleGoToPoint,
  handleDeleteMeasuredPoint,
  handleAutofocusAtPoint,
  handleRefitGroup,
  handleSetCurrentZ,
  handleSetCurrentXYZ,
  handleStepZ,
  goToInProgress,
  highlightPoint,
  clearHighlight,
}) => {
  const theme = useTheme();
  // State that exists only for this table, and now lives with it.
  const [editingPointZ, setEditingPointZ] = useState(null);
  const [editingPointXY, setEditingPointXY] = useState(null);
  const [visiblePointCount, setVisiblePointCount] = useState({});

  return (
    <>
          {groupEntries.length > 0 && (
            <Accordion
              expanded={showMeasuredPoints}
              onChange={() => setShowMeasuredPoints(!showMeasuredPoints)}
              variant="outlined"
              sx={{ mb: 2 }}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="body2">
                  Measured Focus Points
                  <Chip
                    label={`${groupEntries.reduce(
                      (sum, [, r]) => sum + (r.points?.length || 0),
                      0
                    )} point(s)`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mb: 1, display: "block" }}
                >
                  Points measured by autofocus during focus map computation. Use "Go To" to move
                  the stage, edit Z to fine-tune, and "Refit" to update the surface.
                </Typography>

                {groupEntries.map(([groupId, result]) => {
                  const pts = result.points || [];
                  if (pts.length === 0) return null;

                  // Compute Z range summary for the group
                  const zValues = pts.map((p) => p.z).filter((z) => z != null && isFinite(z));
                  const zMin = zValues.length > 0 ? Math.min(...zValues) : 0;
                  const zMax = zValues.length > 0 ? Math.max(...zValues) : 0;
                  const maxVisible = visiblePointCount[groupId] || POINTS_PAGE_SIZE;
                  const visiblePts = pts.slice(0, maxVisible);
                  const hasMore = pts.length > maxVisible;

                  return (
                    <Box key={groupId} sx={{ mb: 2 }}>
                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                          mb: 0.5,
                        }}
                      >
                        <Typography variant="body2" fontWeight={500}>
                          {result.group_name || groupId}
                        </Typography>
                        <Chip
                          label={`${pts.length} pts`}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          label={`Z: ${zMin.toFixed(1)} – ${zMax.toFixed(1)} µm`}
                          size="small"
                          variant="outlined"
                          color="info"
                        />
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<RefreshIcon />}
                          onClick={() => handleRefitGroup(groupId, pts)}
                          disabled={ui.isComputing}
                        >
                          Refit
                        </Button>
                      </Box>
                      <TableContainer
                        component={Paper}
                        variant="outlined"
                        sx={{ maxHeight: 300 }}
                      >
                        <Table size="small" stickyHeader>
                          <TableHead>
                            <TableRow>
                              <TableCell>#</TableCell>
                              <TableCell>X (µm)</TableCell>
                              <TableCell>Y (µm)</TableCell>
                              <TableCell>Z (µm)</TableCell>
                              <TableCell align="right">Actions</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {visiblePts.map((pt, idx) => {
                              const isEditing =
                                editingPointZ?.groupId === groupId &&
                                editingPointZ?.pointIndex === idx;
                              const goToKey = `${groupId}-${idx}`;
                              const isMoving = goToInProgress === goToKey;

                              const isHighlighted =
                                highlightedPoint?.source === "measured" &&
                                highlightedPoint?.groupId === groupId &&
                                highlightedPoint?.index === idx;
                              // Same colour ramp as the heatmap preview so the
                              // row chip visually matches the drawn point.
                              const zColor = colorForValue(
                                zMax > zMin ? (pt.z - zMin) / (zMax - zMin) : 0.5
                              );

                              return (
                                <TableRow
                                  key={idx}
                                  hover
                                  selected={isHighlighted}
                                  onMouseEnter={() => highlightPoint("measured", groupId, idx)}
                                  onMouseLeave={clearHighlight}
                                >
                                  <TableCell>
                                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                                      <Box
                                        sx={{
                                          width: 10,
                                          height: 10,
                                          borderRadius: "50%",
                                          bgcolor: zColor,
                                          border: `1px solid ${theme.palette.text.primary}`,
                                          flexShrink: 0,
                                        }}
                                      />
                                      {idx + 1}
                                    </Box>
                                  </TableCell>
                                  {/* Editable X */}
                                  <TableCell>
                                    {editingPointXY?.groupId === groupId &&
                                     editingPointXY?.pointIndex === idx &&
                                     editingPointXY?.field === "x" ? (
                                      <TextField
                                        type="number"
                                        size="small"
                                        variant="standard"
                                        value={editingPointXY.value}
                                        onChange={(e) =>
                                          setEditingPointXY({
                                            ...editingPointXY,
                                            value: parseFloat(e.target.value) || 0,
                                          })
                                        }
                                        onBlur={() => {
                                          const updatedPts = [...pts];
                                          updatedPts[idx] = { ...pt, x: editingPointXY.value };
                                          dispatch(
                                            focusMapSlice.updateFocusMapGroupResult({
                                              groupId,
                                              result: { ...result, points: updatedPts },
                                            })
                                          );
                                          setEditingPointXY(null);
                                        }}
                                        onKeyDown={(e) => { if (e.key === "Enter") e.target.blur(); }}
                                        autoFocus
                                        sx={{ width: 80 }}
                                      />
                                    ) : (
                                      <Box
                                        sx={{ display: "flex", alignItems: "center", gap: 0.5, cursor: "pointer", "&:hover": { color: "primary.main" } }}
                                        onClick={() => setEditingPointXY({ groupId, pointIndex: idx, field: "x", value: pt.x })}
                                      >
                                        {pt.x?.toFixed(1)}
                                        <EditIcon fontSize="inherit" sx={{ opacity: 0.4 }} />
                                      </Box>
                                    )}
                                  </TableCell>
                                  {/* Editable Y */}
                                  <TableCell>
                                    {editingPointXY?.groupId === groupId &&
                                     editingPointXY?.pointIndex === idx &&
                                     editingPointXY?.field === "y" ? (
                                      <TextField
                                        type="number"
                                        size="small"
                                        variant="standard"
                                        value={editingPointXY.value}
                                        onChange={(e) =>
                                          setEditingPointXY({
                                            ...editingPointXY,
                                            value: parseFloat(e.target.value) || 0,
                                          })
                                        }
                                        onBlur={() => {
                                          const updatedPts = [...pts];
                                          updatedPts[idx] = { ...pt, y: editingPointXY.value };
                                          dispatch(
                                            focusMapSlice.updateFocusMapGroupResult({
                                              groupId,
                                              result: { ...result, points: updatedPts },
                                            })
                                          );
                                          setEditingPointXY(null);
                                        }}
                                        onKeyDown={(e) => { if (e.key === "Enter") e.target.blur(); }}
                                        autoFocus
                                        sx={{ width: 80 }}
                                      />
                                    ) : (
                                      <Box
                                        sx={{ display: "flex", alignItems: "center", gap: 0.5, cursor: "pointer", "&:hover": { color: "primary.main" } }}
                                        onClick={() => setEditingPointXY({ groupId, pointIndex: idx, field: "y", value: pt.y })}
                                      >
                                        {pt.y?.toFixed(1)}
                                        <EditIcon fontSize="inherit" sx={{ opacity: 0.4 }} />
                                      </Box>
                                    )}
                                  </TableCell>
                                  <TableCell>
                                    {isEditing ? (
                                      <TextField
                                        type="number"
                                        size="small"
                                        variant="standard"
                                        value={editingPointZ.z}
                                        onChange={(e) =>
                                          setEditingPointZ({
                                            ...editingPointZ,
                                            z: parseFloat(e.target.value) || 0,
                                          })
                                        }
                                        onBlur={() => {
                                          // Save edited Z back into the result points
                                          // (local only, use Refit to apply)
                                          const updatedPts = [...pts];
                                          updatedPts[idx] = {
                                            ...pt,
                                            z: editingPointZ.z,
                                          };
                                          dispatch(
                                            focusMapSlice.updateFocusMapGroupResult({
                                              groupId,
                                              result: {
                                                ...result,
                                                points: updatedPts,
                                              },
                                            })
                                          );
                                          setEditingPointZ(null);
                                        }}
                                        onKeyDown={(e) => {
                                          if (e.key === "Enter") e.target.blur();
                                        }}
                                        autoFocus
                                        sx={{ width: 80 }}
                                      />
                                    ) : (
                                      <Box
                                        sx={{
                                          display: "flex",
                                          alignItems: "center",
                                          gap: 0.5,
                                          cursor: "pointer",
                                          "&:hover": {
                                            color: "primary.main",
                                          },
                                        }}
                                        onClick={() =>
                                          setEditingPointZ({
                                            groupId,
                                            pointIndex: idx,
                                            z: pt.z,
                                          })
                                        }
                                      >
                                        {pt.z?.toFixed(2)}
                                        <EditIcon
                                          fontSize="inherit"
                                          sx={{ opacity: 0.4 }}
                                        />
                                      </Box>
                                    )}
                                  </TableCell>
                                  <TableCell align="right">
                                    <Box sx={{ display: "flex", gap: 0.25, justifyContent: "flex-end" }}>
                                      <Tooltip title="Move stage to this XYZ position">
                                        <span>
                                          <IconButton
                                            size="small"
                                            color="primary"
                                            onClick={() =>
                                              handleGoToPoint(pt, groupId, idx)
                                            }
                                            disabled={isMoving}
                                          >
                                            {isMoving ? (
                                              <CircularProgress size={16} />
                                            ) : (
                                              <MyLocationIcon fontSize="small" />
                                            )}
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                      <Tooltip title="Run autofocus at this XY, update Z">
                                        <span>
                                          <IconButton
                                            size="small"
                                            color="secondary"
                                            onClick={() =>
                                              handleAutofocusAtPoint(pt, groupId, idx)
                                            }
                                            disabled={isMoving}
                                          >
                                            <CenterFocusStrongIcon fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                      <Tooltip title="Step Z up (+5 µm)">
                                        <span>
                                          <IconButton
                                            size="small"
                                            onClick={() =>
                                              handleStepZ(pt, groupId, idx, "up")
                                            }
                                            disabled={isMoving}
                                          >
                                            <ArrowUpwardIcon fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                      <Tooltip title="Step Z down (−5 µm)">
                                        <span>
                                          <IconButton
                                            size="small"
                                            onClick={() =>
                                              handleStepZ(pt, groupId, idx, "down")
                                            }
                                            disabled={isMoving}
                                          >
                                            <ArrowDownwardIcon fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                      <Tooltip title="Set this point's Z to current stage Z">
                                        <span>
                                          <IconButton
                                            size="small"
                                            color="success"
                                            onClick={() =>
                                              handleSetCurrentZ(pt, groupId, idx)
                                            }
                                            disabled={isMoving}
                                          >
                                            <GpsFixedIcon fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                      <Tooltip title="Set this point's XYZ to current stage position">
                                        <span>
                                          <IconButton
                                            size="small"
                                            color="info"
                                            onClick={() =>
                                              handleSetCurrentXYZ(pt, groupId, idx)
                                            }
                                            disabled={isMoving}
                                          >
                                            <MyLocationIcon fontSize="small" sx={{ color: theme.palette.info.main }} />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                      <Tooltip title="Delete this point and refit the surface from the remaining points">
                                        <span>
                                          <IconButton
                                            size="small"
                                            color="error"
                                            onClick={() =>
                                              handleDeleteMeasuredPoint(groupId, idx)
                                            }
                                            disabled={isMoving || ui.isComputing}
                                          >
                                            <DeleteIcon fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                    </Box>
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                      {/* Pagination controls for large point lists */}
                      {(hasMore || maxVisible > POINTS_PAGE_SIZE) && (
                        <Box sx={{ display: "flex", gap: 1, mt: 0.5, alignItems: "center" }}>
                          <Typography variant="caption" color="text.secondary">
                            Showing {Math.min(maxVisible, pts.length)} of {pts.length} points
                          </Typography>
                          {hasMore && (
                            <Button
                              size="small"
                              variant="text"
                              onClick={() =>
                                setVisiblePointCount((prev) => ({
                                  ...prev,
                                  [groupId]: (prev[groupId] || POINTS_PAGE_SIZE) + POINTS_PAGE_SIZE,
                                }))
                              }
                            >
                              Show More
                            </Button>
                          )}
                          {maxVisible > POINTS_PAGE_SIZE && (
                            <Button
                              size="small"
                              variant="text"
                              onClick={() =>
                                setVisiblePointCount((prev) => ({
                                  ...prev,
                                  [groupId]: POINTS_PAGE_SIZE,
                                }))
                              }
                            >
                              Collapse
                            </Button>
                          )}
                        </Box>
                      )}
                    </Box>
                  );
                })}
              </AccordionDetails>
            </Accordion>
          )}
    </>
  );
};

export default MeasuredFocusPoints;
