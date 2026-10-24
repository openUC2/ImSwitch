import React, { useEffect, useState } from "react";
import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import apiUC2ConfigControllerGetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiUC2ConfigControllerSetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareServer";
import apiUC2ConfigControllerSetOTAFirmwareRelease from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareRelease";
import apiUC2ConfigControllerListFirmwareReleases from "../backendapi/apiUC2ConfigControllerListFirmwareReleases";

export const DEFAULT_FIRMWARE_URL = "http://host.docker.internal/firmware";

// Where firmware images come from: { source: "local" | "github", url, release }.
export const loadFirmwareSource = async () => {
  const current = await apiUC2ConfigControllerGetOTAFirmwareServer();
  return {
    source: current?.source === "github" ? "github" : "local",
    url: current?.firmware_server_url || DEFAULT_FIRMWARE_URL,
    release: current?.github_release || "latest",
    repo: current?.github_repo || "youseetoo/uc2-esp32",
  };
};

// Make the backend use *value*; resolves to the backend's answer ({ status, message, ... }).
export const applyFirmwareSource = (value) =>
  value.source === "github"
    ? apiUC2ConfigControllerSetOTAFirmwareRelease(value.release || "latest")
    : apiUC2ConfigControllerSetOTAFirmwareServer(value.url);

const releaseLabel = (r) =>
  [r.tag, r.prerelease && "pre-release", r.published_at?.slice(0, 10),
   !r.image_count && "no ImSwitch images"].filter(Boolean).join(" · ");

// Local firmware server (the firmware-image-server container) or a GitHub
// release of the firmware repository, downloaded by the ImSwitch host.
// Controlled: the parent applies the value (applyFirmwareSource).
const FirmwareSourcePicker = ({ value, onChange, disabled = false }) => {
  const [releases, setReleases] = useState(null);
  const [releaseError, setReleaseError] = useState(null);
  const github = value?.source === "github";

  useEffect(() => {
    if (!github || releases) return;
    apiUC2ConfigControllerListFirmwareReleases()
      .then((result) => {
        if (result?.status === "success") setReleases(result.releases || []);
        else setReleaseError(result?.message || "Could not list the releases.");
      })
      .catch((e) => setReleaseError(`Could not list the releases: ${e.message}`));
  }, [github, releases]);

  if (!value) return null;
  const set = (patch) => onChange({ ...value, ...patch });
  const newest = releases?.find((r) => r.image_count > 0);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      <ToggleButtonGroup size="small" exclusive value={value.source} disabled={disabled}
                         onChange={(_, source) => source && set({ source })}>
        <ToggleButton value="local">Local server</ToggleButton>
        <ToggleButton value="github">GitHub release</ToggleButton>
      </ToggleButtonGroup>
      {github ? (
        <>
          <FormControl size="small" sx={{ minWidth: 320 }} disabled={disabled}>
            <InputLabel id="fw-release-label">Release</InputLabel>
            <Select labelId="fw-release-label" label="Release" value={value.release || "latest"}
                    onChange={(e) => set({ release: e.target.value })}>
              <MenuItem value="latest">
                {`Newest release${newest ? ` (${newest.tag})` : ""}`}
              </MenuItem>
              {/* the configured tag, until the list has loaded */}
              {value.release && value.release !== "latest" &&
                !releases?.some((r) => r.tag === value.release) && (
                <MenuItem value={value.release}>{value.release}</MenuItem>
              )}
              {(releases || []).map((r) => (
                <MenuItem key={r.tag} value={r.tag} disabled={!r.image_count}>
                  {releaseLabel(r)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color={releaseError ? "error" : "text.secondary"}>
            {releaseError ||
              `Images from github.com/${value.repo || "youseetoo/uc2-esp32"}, downloaded by ` +
              "the ImSwitch host (needs internet). Each download is checked against the " +
              "release's sha256."}
          </Typography>
        </>
      ) : (
        <TextField size="small" label="Firmware server URL" value={value.url || ""}
                   disabled={disabled} sx={{ minWidth: 320 }}
                   onChange={(e) => set({ url: e.target.value })}
                   placeholder={DEFAULT_FIRMWARE_URL}
                   helperText="The firmware-image-server container serving the .bin files" />
      )}
    </Box>
  );
};

export default FirmwareSourcePicker;
