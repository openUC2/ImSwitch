// src/components/aufgaben/AufgabenOutline.jsx
// The job as an indented outline. Containers show their branches (Do / Else /
// While waiting). Rows carry the validation issues of their step, whether it
// asks first, moves the stage or switches light on, and which step runs now.
// Steps move by drag and drop or with the toolbar.
import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, IconButton, Tooltip, Typography } from "@mui/material";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import FormatIndentIncreaseIcon from "@mui/icons-material/FormatIndentIncrease";
import FormatIndentDecreaseIcon from "@mui/icons-material/FormatIndentDecrease";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import LightbulbOutlinedIcon from "@mui/icons-material/LightbulbOutlined";
import OpenWithIcon from "@mui/icons-material/OpenWith";

import * as A from "../../state/slices/AufgabenSlice";
import {
  BRANCH_TITLES,
  branchesOf,
  catalogById,
  issuesFor,
  paramSummary,
} from "../../utils/aufgabenJob";
import { DRAG_TASK } from "./AufgabenPalette";
import { useT } from "../../i18n";

const DRAG_NODE = "application/x-aufgaben-node";
const INDENT = 18;

// Drop handlers: a step or palette task lands at (parentId, branch, index).
const useDrop = (parentId, branch, index) => {
  const dispatch = useDispatch();
  return {
    onDragOver: (e) => e.preventDefault(),
    onDrop: (e) => {
      e.preventDefault();
      e.stopPropagation();
      const id = e.dataTransfer.getData(DRAG_NODE);
      const task = e.dataTransfer.getData(DRAG_TASK);
      if (id) dispatch(A.placeNode({ id, parentId, branch, index }));
      else if (task) dispatch(A.addNodeAt({ task, parentId, branch, index }));
    },
  };
};

const Markers = ({ node, entry, issues }) => {
  const t = useT();
  const errors = issues.filter((i) => i.level === "error");
  const warnings = issues.filter((i) => i.level !== "error");
  const flags = entry?.flags || {};
  const tip = (list) => list.map((i) => i.message).join("\n");
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, ml: "auto", pl: 1 }}>
      {flags.ask_first && (
        <Typography variant="caption" color="warning.main">
          {t("asks first")}
        </Typography>
      )}
      {(flags.moves_xy || flags.moves_z) && (
        <Tooltip title={t("Moves the stage")}>
          <OpenWithIcon sx={{ fontSize: 15, color: "text.secondary" }} />
        </Tooltip>
      )}
      {flags.emits_light && (
        <Tooltip title={t("Switches light on")}>
          <LightbulbOutlinedIcon sx={{ fontSize: 15, color: "text.secondary" }} />
        </Tooltip>
      )}
      {node.enabled === false && (
        <Tooltip title={t("Disabled")}>
          <VisibilityOffIcon sx={{ fontSize: 15, color: "text.disabled" }} />
        </Tooltip>
      )}
      {warnings.length > 0 && (
        <Tooltip title={<span style={{ whiteSpace: "pre-line" }}>{tip(warnings)}</span>}>
          <WarningAmberIcon sx={{ fontSize: 16, color: "warning.main" }} />
        </Tooltip>
      )}
      {errors.length > 0 && (
        <Tooltip title={<span style={{ whiteSpace: "pre-line" }}>{tip(errors)}</span>}>
          <ErrorOutlineIcon sx={{ fontSize: 16, color: "error.main" }} aria-label={t("Error")} />
        </Tooltip>
      )}
    </Box>
  );
};

const BranchDrop = ({ parentId, branch, depth, title, empty }) => {
  const t = useT();
  const drop = useDrop(parentId, branch, 0);
  return (
    <Box
      {...drop}
      sx={{ pl: `${depth * INDENT + 8}px`, py: 0.25, color: "text.secondary", fontSize: 12 }}
    >
      {title}
      {empty && (
        <Box component="span" sx={{ ml: 1, fontStyle: "italic", color: "text.disabled" }}>
          {t("empty: drop steps here")}
        </Box>
      )}
    </Box>
  );
};

const NodeRow = ({ node, depth, parentId, branch, index, ctx }) => {
  const dispatch = useDispatch();
  const t = useT();
  const entry = ctx.byId[node.task];
  const issues = issuesFor(ctx.validation, node.id);
  const selected = ctx.selectedId === node.id;
  const running = ctx.runningId === node.id;
  const drop = useDrop(parentId, branch, index + 1);
  const branches = branchesOf(entry).length
    ? branchesOf(entry)
    : ["body", "orelse", "idle"].filter((b) => node[b]?.length);
  return (
    <>
      <Box
        draggable
        onDragStart={(e) => {
          e.stopPropagation();
          e.dataTransfer.setData(DRAG_NODE, node.id);
        }}
        {...drop}
        onClick={() => dispatch(A.selectNode(node.id))}
        data-testid={`aufgaben-node-${node.id}`}
        sx={{
          display: "flex",
          alignItems: "baseline",
          gap: 1,
          pl: `${depth * INDENT + 8}px`,
          pr: 1,
          py: 0.5,
          cursor: "pointer",
          borderLeft: 3,
          borderColor: running ? "primary.main" : "transparent",
          bgcolor: selected ? "action.selected" : "transparent",
          opacity: node.enabled === false ? 0.5 : 1,
          "&:hover": { bgcolor: selected ? "action.selected" : "action.hover" },
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: branches.length ? 600 : 400 }}>
          {node.label || t(entry?.title || node.task)}
        </Typography>
        <Typography variant="caption" sx={{ fontFamily: "monospace", color: "text.secondary" }}>
          {node.id}
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          noWrap
          sx={{ minWidth: 0, flexShrink: 1 }}
        >
          {paramSummary(node.params)}
        </Typography>
        <Markers node={node} entry={entry} issues={issues} />
      </Box>
      {branches.map((name) => (
        <React.Fragment key={name}>
          {(branches.length > 1 || !(node[name] || []).length) && (
            <BranchDrop
              parentId={node.id}
              branch={name}
              depth={depth + 1}
              title={branches.length > 1 ? t(BRANCH_TITLES[name]) : ""}
              empty={!(node[name] || []).length}
            />
          )}
          <NodeList nodes={node[name] || []} depth={depth + 1} parentId={node.id} branch={name} ctx={ctx} />
        </React.Fragment>
      ))}
    </>
  );
};

const NodeList = ({ nodes, depth, parentId, branch, ctx }) =>
  nodes.map((node, index) => (
    <NodeRow
      key={node.id}
      node={node}
      depth={depth}
      parentId={parentId}
      branch={branch}
      index={index}
      ctx={ctx}
    />
  ));

const Toolbar = ({ selected }) => {
  const t = useT();
  const dispatch = useDispatch();
  const act = (action) => () => selected && dispatch(action);
  const buttons = [
    ["Move up", ArrowUpwardIcon, A.moveNode({ id: selected, delta: -1 })],
    ["Move down", ArrowDownwardIcon, A.moveNode({ id: selected, delta: 1 })],
    ["Into the step above", FormatIndentIncreaseIcon, A.indentNode(selected)],
    ["Out of its container", FormatIndentDecreaseIcon, A.outdentNode(selected)],
    ["Duplicate", ContentCopyIcon, A.duplicateNode(selected)],
    ["Delete", DeleteOutlineIcon, A.removeNode(selected)],
  ];
  return (
    <Box sx={{ display: "flex", borderBottom: 1, borderColor: "divider", mb: 0.5 }}>
      {buttons.map(([title, Icon, action]) => (
        <Tooltip key={title} title={t(title)}>
          <span>
            <IconButton size="small" disabled={!selected} onClick={act(action)} aria-label={t(title)}>
              <Icon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      ))}
    </Box>
  );
};

const AufgabenOutline = () => {
  const t = useT();
  const state = useSelector(A.getAufgabenState);
  const { job, catalog, selectedId, validation, status } = state;
  const ctx = {
    byId: catalogById(catalog),
    selectedId,
    validation,
    runningId: A.isActive(status) ? status.node : null,
  };
  const endDrop = useDrop(null, null, job.tasks.length);
  const jobIssues = issuesFor(validation, null);
  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, height: "100%" }}>
      <Toolbar selected={selectedId} />
      {jobIssues.map((issue, i) => (
        <Typography key={i} variant="caption" color={issue.level === "error" ? "error" : "warning.main"}>
          {issue.message}
        </Typography>
      ))}
      <Box sx={{ overflowY: "auto", flex: 1 }}>
        <NodeList nodes={job.tasks} depth={0} parentId={null} branch={null} ctx={ctx} />
        <Box
          {...endDrop}
          sx={{ minHeight: 48, px: 1, py: 1, color: "text.disabled", fontSize: 12 }}
        >
          {job.tasks.length === 0
            ? t("Click or drag steps from the left to build the job.")
            : t("Drop here to add at the end")}
        </Box>
      </Box>
    </Box>
  );
};

export default AufgabenOutline;
