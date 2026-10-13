// src/components/aufgaben/AufgabenInspector.jsx
// Right-hand panel of the editor: the selected step's parameters, the job's
// resources (positions, channels, plans shared by several steps) and the job
// settings (name, storage, variables, run parameters).
import React, { useEffect, useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Collapse,
  FormControlLabel,
  FormGroup,
  IconButton,
  Menu,
  MenuItem,
  Stack,
  Switch,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";

import * as A from "../../state/slices/AufgabenSlice";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions";
import {
  RESOURCE_TEMPLATES,
  catalogById,
  findNode,
  isValidId,
  issuesFor,
  paramFields,
  parseNumberInput,
  walk,
} from "../../utils/aufgabenJob";
import { CommitField, JsonField, ParamField, SectionTitle } from "./AufgabenFields";
import { useT } from "../../i18n";

const ON_ERROR = ["abort", "skip", "continue_loop", "retry"];

const freeName = (resources, base) => {
  if (!resources[base]) return base;
  for (let n = 2; ; n += 1) if (!resources[`${base}_${n}`]) return `${base}_${n}`;
};

// Illumination sources of this setup, from the catalogue's device enums.
const illuminationSources = (byId) =>
  byId.set_illumination?.schema?.properties?.source?.enum || [];

// -- step ----------------------------------------------------------------------

const ErrorPolicy = ({ node, onPatch }) => {
  const t = useT();
  const retry = typeof node.on_error === "object" ? node.on_error : null;
  const mode = retry ? "retry" : node.on_error || "abort";
  return (
    <Stack direction="row" spacing={1}>
      <TextField
        select
        size="small"
        label={t("On error")}
        value={mode}
        sx={{ minWidth: 150 }}
        onChange={(e) =>
          onPatch({
            on_error: e.target.value === "retry" ? { retry: 1, then: "abort" } : e.target.value,
          })
        }
      >
        {ON_ERROR.map((value) => (
          <MenuItem key={value} value={value}>
            {t(value.replace("_", " "))}
          </MenuItem>
        ))}
      </TextField>
      {retry && (
        <>
          <CommitField
            label={t("Retries")}
            value={retry.retry}
            onCommit={(text) => onPatch({ on_error: { ...retry, retry: Math.max(1, parseInt(text, 10) || 1) } })}
          />
          <TextField
            select
            size="small"
            fullWidth
            label={t("Then")}
            value={retry.then}
            onChange={(e) => onPatch({ on_error: { ...retry, then: e.target.value } })}
          >
            {ON_ERROR.slice(0, 3).map((value) => (
              <MenuItem key={value} value={value}>
                {t(value.replace("_", " "))}
              </MenuItem>
            ))}
          </TextField>
        </>
      )}
    </Stack>
  );
};

const Advanced = ({ node, job, byId, onPatch, onParam }) => {
  const t = useT();
  const loops = [...walk(job.tasks)].filter((e) => byId[e.node.task]?.loop).map((e) => e.node.id);
  const every = node.every || null;
  return (
    <Stack spacing={1.5} sx={{ mt: 1 }}>
      {typeof node.enabled === "string" ? (
        <CommitField
          label={t("Enabled when")}
          value={node.enabled}
          onCommit={(text) => onPatch({ enabled: text.trim() === "" ? true : text })}
        />
      ) : (
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={node.enabled !== false}
              onChange={(e) => onPatch({ enabled: e.target.checked })}
            />
          }
          label={t("Enabled")}
        />
      )}
      <ErrorPolicy node={node} onPatch={onPatch} />
      <Stack direction="row" spacing={1} alignItems="center">
        <CommitField
          label={t("Only every n-th iteration")}
          value={every?.nth}
          onCommit={(text) => {
            const nth = parseInt(text, 10);
            onPatch({ every: nth >= 1 ? { ...(every || {}), nth } : null });
          }}
        />
        <TextField
          select
          size="small"
          fullWidth
          label={t("of loop")}
          value={every?.of || ""}
          disabled={!every}
          onChange={(e) => onPatch({ every: { ...every, of: e.target.value || null } })}
        >
          <MenuItem value="">
            <em>{t("innermost")}</em>
          </MenuItem>
          {loops.map((id) => (
            <MenuItem key={id} value={id}>
              {id}
            </MenuItem>
          ))}
        </TextField>
      </Stack>
      <TextField
        select
        size="small"
        label={t("Defaults from resource")}
        value={(node.params?.$from || "").replace(/^@/, "")}
        helperText={t("Parameters not set here are taken from this resource")}
        onChange={(e) => onParam("$from", e.target.value ? `@${e.target.value}` : undefined)}
      >
        <MenuItem value="">
          <em>{t("(none)")}</em>
        </MenuItem>
        {Object.keys(job.resources || {}).map((name) => (
          <MenuItem key={name} value={name}>
            @{name}
          </MenuItem>
        ))}
      </TextField>
    </Stack>
  );
};

const StepPanel = ({ node, entry, job, byId, validation }) => {
  const t = useT();
  const dispatch = useDispatch();
  const [idError, setIdError] = useState("");
  const [advanced, setAdvanced] = useState(false);
  useEffect(() => setIdError(""), [node.id]);
  const fields = useMemo(() => paramFields(entry), [entry]);
  const issues = issuesFor(validation, node.id);
  const onParam = (name, value) => dispatch(A.setNodeParam({ id: node.id, name, value }));
  const onPatch = (patch) => dispatch(A.updateNode({ id: node.id, patch }));
  const onCreateResource = (kind, param) => {
    const name = freeName(job.resources || {}, kind);
    dispatch(A.setResource({ name, resource: RESOURCE_TEMPLATES[kind] || { kind } }));
    onParam(param, `@${name}`);
  };
  return (
    <Box>
      <Typography variant="subtitle1">{t(entry?.title || node.task)}</Typography>
      {entry?.description && (
        <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-line" }}>
          {entry.description}
        </Typography>
      )}
      {!entry && <Alert severity="error">{t("This microscope has no step '{task}'.", { task: node.task })}</Alert>}
      {issues.map((issue, i) => (
        <Alert key={i} severity={issue.level === "error" ? "error" : "warning"} sx={{ mt: 1, py: 0 }}>
          {issue.message}
        </Alert>
      ))}
      <Stack direction="row" spacing={1} sx={{ mt: 1.5 }}>
        <CommitField
          label={t("Id")}
          value={node.id}
          error={!!idError}
          helperText={idError || " "}
          InputProps={{ sx: { fontFamily: "monospace" } }}
          onCommit={(text) => {
            if (!isValidId(text)) return setIdError(t("Letters, digits and _; not starting with a digit"));
            if (findNode(job, text)) return setIdError(t("Another step has this id"));
            setIdError("");
            dispatch(A.renameNode({ id: node.id, newId: text }));
          }}
        />
        <CommitField
          label={t("Label")}
          value={node.label}
          helperText=" "
          onCommit={(text) => onPatch({ label: text || null })}
        />
      </Stack>
      {fields.length > 0 && <SectionTitle>{t("Parameters")}</SectionTitle>}
      <Stack spacing={1.5}>
        {fields.map((field) => (
          <ParamField
            key={field.name}
            field={field}
            value={node.params?.[field.name]}
            job={job}
            onChange={(value) => onParam(field.name, value)}
            onCreateResource={onCreateResource}
          />
        ))}
      </Stack>
      <Button
        size="small"
        sx={{ mt: 2 }}
        onClick={() => setAdvanced(!advanced)}
        endIcon={advanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
      >
        {t("Run conditions and errors")}
      </Button>
      <Collapse in={advanced}>
        <Advanced node={node} job={job} byId={byId} onPatch={onPatch} onParam={onParam} />
      </Collapse>
    </Box>
  );
};

// -- resources --------------------------------------------------------------------

const NumberCell = ({ value, onChange, width = 90 }) => (
  <TableCell sx={{ px: 0.5 }}>
    <CommitField
      value={value}
      sx={{ width }}
      inputProps={{ style: { padding: "4px 6px" } }}
      onCommit={(text) => onChange(parseNumberInput(text))}
    />
  </TableCell>
);

const PositionsEditor = ({ resource, onChange }) => {
  const t = useT();
  const [error, setError] = useState("");
  const positions = resource.positions || [];
  const set = (list) => onChange({ ...resource, positions: list });
  const update = (i, patch) => set(positions.map((p, j) => (j === i ? { ...p, ...patch } : p)));
  const addCurrent = async () => {
    try {
      const all = await apiPositionerControllerGetPositions();
      const stage = Object.values(all || {})[0] || {};
      set([...positions, { x: stage.X ?? 0, y: stage.Y ?? 0, z: stage.Z ?? null, name: `p${positions.length}` }]);
      setError("");
    } catch (e) {
      setError(t("Could not read the stage position"));
    }
  };
  return (
    <Box>
      <Table size="small">
        <TableHead>
          <TableRow>
            {["Name", "X µm", "Y µm", "Z µm", ""].map((h) => (
              <TableCell key={h} sx={{ px: 0.5 }}>
                {t(h)}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((p, i) => (
            <TableRow key={i}>
              <TableCell sx={{ px: 0.5 }}>
                <CommitField
                  value={p.name}
                  sx={{ width: 80 }}
                  inputProps={{ style: { padding: "4px 6px" } }}
                  onCommit={(text) => update(i, { name: text || null })}
                />
              </TableCell>
              <NumberCell value={p.x} onChange={(x) => update(i, { x })} />
              <NumberCell value={p.y} onChange={(y) => update(i, { y })} />
              <NumberCell value={p.z} onChange={(z) => update(i, { z: z ?? null })} />
              <TableCell sx={{ px: 0 }}>
                <IconButton size="small" aria-label={t("Remove")} onClick={() => set(positions.filter((_, j) => j !== i))}>
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Button size="small" onClick={addCurrent} sx={{ mt: 0.5 }}>
        {t("Add current stage position")}
      </Button>
      {error && <Typography variant="caption" color="error" sx={{ ml: 1 }}>{error}</Typography>}
    </Box>
  );
};

const CaptureEditor = ({ resource, sources, onChange }) => {
  const t = useT();
  const channels = resource.channels || [];
  const set = (list) => onChange({ ...resource, channels: list });
  const update = (i, patch) => set(channels.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  return (
    <Box>
      <Table size="small">
        <TableHead>
          <TableRow>
            {["Source", "Exposure ms", "Intensity", ""].map((h) => (
              <TableCell key={h} sx={{ px: 0.5 }}>
                {t(h)}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {channels.map((c, i) => (
            <TableRow key={i}>
              <TableCell sx={{ px: 0.5 }}>
                <TextField
                  select
                  size="small"
                  value={sources.includes(c.config) ? c.config : ""}
                  sx={{ width: 120 }}
                  SelectProps={{ displayEmpty: true }}
                  onChange={(e) => update(i, { config: e.target.value })}
                >
                  <MenuItem value="">
                    <em>{c.config || t("(choose)")}</em>
                  </MenuItem>
                  {sources.map((s) => (
                    <MenuItem key={s} value={s}>
                      {s}
                    </MenuItem>
                  ))}
                </TextField>
              </TableCell>
              <NumberCell value={c.exposure} onChange={(exposure) => update(i, { exposure: exposure ?? null })} />
              <NumberCell value={c.intensity} onChange={(intensity) => update(i, { intensity: intensity ?? null })} />
              <TableCell sx={{ px: 0 }}>
                <IconButton
                  size="small"
                  aria-label={t("Remove")}
                  disabled={channels.length === 1}
                  onClick={() => set(channels.filter((_, j) => j !== i))}
                >
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Button size="small" sx={{ mt: 0.5 }} onClick={() => set([...channels, { config: sources[0] || "", exposure: 10 }])}>
        {t("Add channel")}
      </Button>
    </Box>
  );
};

const ResourceCard = ({ name, resource, sources }) => {
  const t = useT();
  const dispatch = useDispatch();
  const [open, setOpen] = useState(false);
  const [nameError, setNameError] = useState("");
  const onChange = (value) => dispatch(A.setResource({ name, resource: value }));
  const { kind, ...body } = resource;
  return (
    <Box sx={{ borderBottom: 1, borderColor: "divider", py: 1 }}>
      <Stack direction="row" spacing={1} alignItems="center">
        <CommitField
          value={name}
          error={!!nameError}
          helperText={nameError || undefined}
          InputProps={{ sx: { fontFamily: "monospace" } }}
          onCommit={(text) => {
            if (!isValidId(text)) return setNameError(t("Letters, digits and _; not starting with a digit"));
            setNameError("");
            dispatch(A.renameResource({ name, newName: text }));
          }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ minWidth: 70 }}>
          {kind}
        </Typography>
        <IconButton size="small" aria-label={t("Edit")} onClick={() => setOpen(!open)}>
          {open ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
        <IconButton size="small" aria-label={t("Remove")} onClick={() => dispatch(A.removeResource(name))}>
          <DeleteOutlineIcon fontSize="small" />
        </IconButton>
      </Stack>
      <Collapse in={open} unmountOnExit>
        <Box sx={{ mt: 1 }}>
          {kind === "positions" && <PositionsEditor resource={resource} onChange={onChange} />}
          {kind === "capture" && <CaptureEditor resource={resource} sources={sources} onChange={onChange} />}
          {kind !== "positions" && kind !== "capture" && (
            <JsonField value={body} minRows={4} onChange={(value) => onChange({ ...(value || {}), kind })} />
          )}
        </Box>
      </Collapse>
    </Box>
  );
};

const ResourcesPanel = ({ job, byId }) => {
  const t = useT();
  const dispatch = useDispatch();
  const [anchor, setAnchor] = useState(null);
  const resources = job.resources || {};
  const add = (kind) => {
    setAnchor(null);
    dispatch(A.setResource({ name: freeName(resources, kind), resource: RESOURCE_TEMPLATES[kind] }));
  };
  return (
    <Box>
      <Typography variant="body2" color="text.secondary">
        {t("Positions, channels and plans that steps refer to as @name.")}
      </Typography>
      {Object.entries(resources).map(([name, resource]) => (
        <ResourceCard key={name} name={name} resource={resource} sources={illuminationSources(byId)} />
      ))}
      <Button size="small" sx={{ mt: 1 }} onClick={(e) => setAnchor(e.currentTarget)}>
        {t("Add resource")}
      </Button>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        {Object.keys(RESOURCE_TEMPLATES).map((kind) => (
          <MenuItem key={kind} onClick={() => add(kind)}>
            {kind}
          </MenuItem>
        ))}
      </Menu>
    </Box>
  );
};

// -- job ------------------------------------------------------------------------------

const JobPanel = ({ job }) => {
  const t = useT();
  const dispatch = useDispatch();
  const storage = job.storage || {};
  const formats = storage.formats || [];
  const setField = (field) => (value) => dispatch(A.setJobField({ field, value }));
  const setStorage = (patch) => dispatch(A.setStorage(patch));
  const toggleFormat = (format) => (e) =>
    setStorage({
      formats: e.target.checked ? [...formats, format] : formats.filter((f) => f !== format),
    });
  return (
    <Stack spacing={1.5}>
      <CommitField label={t("Name")} value={job.name} onCommit={(text) => text && setField("name")(text)} />
      <CommitField
        label={t("Description")}
        value={job.description}
        multiline
        minRows={2}
        onCommit={setField("description")}
      />
      <SectionTitle>{t("Storage")}</SectionTitle>
      <FormGroup row>
        {["ome-zarr", "ome-tiff"].map((format) => (
          <FormControlLabel
            key={format}
            control={<Checkbox size="small" checked={formats.includes(format)} onChange={toggleFormat(format)} />}
            label={format === "ome-zarr" ? "OME-Zarr" : "OME-TIFF"}
          />
        ))}
      </FormGroup>
      <Stack direction="row" spacing={1}>
        <TextField
          select
          size="small"
          fullWidth
          label={t("Layout")}
          value={storage.layout || "auto"}
          onChange={(e) => setStorage({ layout: e.target.value })}
        >
          {["auto", "positions", "plate"].map((v) => (
            <MenuItem key={v} value={v}>
              {t(v)}
            </MenuItem>
          ))}
        </TextField>
        <TextField
          select
          size="small"
          fullWidth
          label={t("Stitching")}
          value={storage.stitching || "stage"}
          disabled={!storage.stitched_preview}
          onChange={(e) => setStorage({ stitching: e.target.value })}
        >
          <MenuItem value="stage">{t("stage positions")}</MenuItem>
          <MenuItem value="ashlar">Ashlar</MenuItem>
        </TextField>
      </Stack>
      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={!!storage.stitched_preview}
            onChange={(e) => setStorage({ stitched_preview: e.target.checked })}
          />
        }
        label={t("Stitched preview per position")}
      />
      <SectionTitle>{t("Variables")}</SectionTitle>
      <JsonField
        value={job.variables || {}}
        minRows={2}
        helperText={t('Starting values, e.g. {"failures": 0}')}
        onChange={(value) => setField("variables")(value || {})}
      />
      <SectionTitle>{t("Run parameters")}</SectionTitle>
      <JsonField
        value={job.parameters || {}}
        minRows={2}
        helperText={t('Asked when the job starts, e.g. {"n": {"type": "integer", "default": 3}}')}
        onChange={(value) => setField("parameters")(value || {})}
      />
    </Stack>
  );
};

const AufgabenInspector = () => {
  const t = useT();
  const { job, catalog, selectedId, validation } = useSelector(A.getAufgabenState);
  const [tab, setTab] = useState(selectedId ? "step" : "job");
  useEffect(() => {
    if (selectedId) setTab("step");
  }, [selectedId]);
  const byId = useMemo(() => catalogById(catalog), [catalog]);
  const found = selectedId ? findNode(job, selectedId) : null;
  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, height: "100%" }}>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} variant="fullWidth" sx={{ minHeight: 36 }}>
        <Tab value="step" label={t("Step")} sx={{ minHeight: 36 }} />
        <Tab value="resources" label={t("Resources")} sx={{ minHeight: 36 }} />
        <Tab value="job" label={t("Job")} sx={{ minHeight: 36 }} />
      </Tabs>
      <Box sx={{ overflowY: "auto", flex: 1, p: 1.5 }}>
        {tab === "step" &&
          (found ? (
            <StepPanel
              node={found.node}
              entry={byId[found.node.task]}
              job={job}
              byId={byId}
              validation={validation}
            />
          ) : (
            <Typography variant="body2" color="text.secondary">
              {t("Select a step in the outline.")}
            </Typography>
          ))}
        {tab === "resources" && <ResourcesPanel job={job} byId={byId} />}
        {tab === "job" && <JobPanel job={job} />}
      </Box>
    </Box>
  );
};

export default AufgabenInspector;
