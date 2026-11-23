/**
 * 3D view of a lightsheet volume (OME-TIFF or OME-Zarr) in the browser.
 *
 * The backend sends a small uint8 copy (getVolumePreview, read from the coarse pyramid
 * level), drawn here as a maximum-intensity projection (three.js ray marching). dz and
 * the xy pixel size come from the file and can be corrected when the metadata is off;
 * the napari snippet below uses the same values.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSelector } from "react-redux";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import {
  Alert, Box, Button, IconButton, LinearProgress, MenuItem, Slider, TextField, Tooltip, Typography,
} from "@mui/material";
import { ContentCopy, Download, Refresh, ViewInAr } from "@mui/icons-material";

import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import { apiGetVolumePreview, apiListVolumes } from "../backendapi/apiLightsheetController.js";
import { dataUrl } from "./omezarr/omeZarrModel";
import VizarrViewer from "./VizarrViewer.jsx";
import ErrorBoundary from "./ErrorBoundary.js";

const VERTEX = /* glsl */ `
out vec3 vPos;
void main() {
  vPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

// Maximum along the ray through the unit box (model space), clipped to [uBoxMin, uBoxMax].
const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D uVol;
uniform vec3 uCam;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform float uLo;
uniform float uHi;
uniform int uSteps;
in vec3 vPos;
out vec4 outColor;
void main() {
  vec3 d = normalize(vPos - uCam);
  vec3 inv = 1.0 / d;
  vec3 t0 = (uBoxMin - uCam) * inv;
  vec3 t1 = (uBoxMax - uCam) * inv;
  vec3 tn = min(t0, t1);
  vec3 tf = max(t0, t1);
  float near = max(max(max(tn.x, tn.y), tn.z), 0.0);
  float far = min(min(tf.x, tf.y), tf.z);
  if (near >= far) discard;
  float dt = (far - near) / float(uSteps);
  float m = 0.0;
  for (int i = 0; i < 1024; i++) {
    if (i >= uSteps) break;
    vec3 p = uCam + d * (near + (float(i) + 0.5) * dt);
    m = max(m, texture(uVol, vec3(p.x + 0.5, 0.5 - p.y, p.z + 0.5)).r);  // row 0 on top
  }
  float v = clamp((m - uLo) / max(uHi - uLo, 1e-3), 0.0, 1.0);
  outColor = vec4(vec3(v), 1.0);
}`;

const fmt = (v, digits = 3) => Number(v.toPrecision(digits));

function napariSnippet(header, dz, px) {
  if (!header) return "";
  const p = header.absolutePath;
  if (header.kind === "zarr") {
    return [
      "import napari, zarr, dask.array as da",
      "",
      `group = zarr.open_group(r"${p}", mode="r")`,
      `levels = [da.from_zarr(group[str(i)]) for i in range(${header.levels})]  # T, C, Z, Y, X`,
      "viewer = napari.current_viewer() or napari.Viewer()",
      `viewer.add_image(levels, multiscale=True, channel_axis=1, scale=(1, ${fmt(dz)}, ${fmt(px)}, ${fmt(px)}),`,
      `                 name="${header.name}")`,
      "# in a script: napari.run()",
    ].join("\n");
  }
  return [
    "import napari, tifffile",
    "",
    `stack = tifffile.imread(r"${p}")  # ${header.fullShape.join(" x ")} (Z, Y, X), ${header.dtype}`,
    "viewer = napari.current_viewer() or napari.Viewer()",
    `viewer.add_image(stack, scale=(${fmt(dz)}, ${fmt(px)}, ${fmt(px)}), name="${header.name}")`,
    "viewer.dims.ndisplay = 3  # 3D view; drag the slider for single planes",
    "# in a script: napari.run()",
  ].join("\n");
}

const LightsheetVolumeViewer = ({ initialPath = null, height = 480 }) => {
  const { ip, apiPort } = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const [volumes, setVolumes] = useState([]);
  const [path, setPath] = useState(initialPath || "");
  const [preview, setPreview] = useState(null);
  const [dz, setDz] = useState(1);
  const [px, setPx] = useState(1);
  const [contrast, setContrast] = useState([0, 255]);
  const [zRange, setZRange] = useState([0, 0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSlices, setShowSlices] = useState(false);
  const [copied, setCopied] = useState(false);

  const boxRef = useRef(null);
  const three = useRef(null);

  const refresh = useCallback(() => {
    apiListVolumes()
      .then((list) => {
        setVolumes(list);
        setPath((p) => p || list[0]?.path || "");
      })
      .catch((e) => setError(`Could not list volumes: ${e.message}`));
  }, []);
  useEffect(refresh, [refresh, ip, apiPort]);
  useEffect(() => {
    if (initialPath) setPath(initialPath);
  }, [initialPath]);

  // path -> preview voxels
  useEffect(() => {
    if (!path) return undefined;
    let live = true;
    setLoading(true);
    setError(null);
    setShowSlices(false);
    apiGetVolumePreview(path)
      .then((p) => {
        if (!live) return;
        setPreview(p);
        setDz(fmt(p.header.zStepUm, 4));
        setPx(fmt(p.header.pixelSizeUm, 4));
        setContrast([0, 255]);
        setZRange([0, p.header.shape[0] - 1]);
      })
      .catch((e) => {
        const body = e.response?.data;
        const detail = body instanceof ArrayBuffer ? new TextDecoder().decode(body) : e.message;
        if (live) setError(`Could not load ${path}: ${detail}`);
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [path]);

  // three.js scene, once
  useEffect(() => {
    const box = boxRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    box.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    camera.position.set(0, 0, 1.9);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      side: THREE.BackSide, // also works with the camera inside the box
      uniforms: {
        uVol: { value: null },
        uCam: { value: new THREE.Vector3() },
        uBoxMin: { value: new THREE.Vector3(-0.5, -0.5, -0.5) },
        uBoxMax: { value: new THREE.Vector3(0.5, 0.5, 0.5) },
        uLo: { value: 0 },
        uHi: { value: 1 },
        uSteps: { value: 256 },
      },
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    mesh.visible = false;
    scene.add(mesh);
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0x4caf50 }),
    );
    mesh.add(outline);

    const resize = () => {
      const w = box.clientWidth || 600;
      const h = box.clientHeight || 400;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(box);
    resize();
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      material.uniforms.uCam.value.copy(mesh.worldToLocal(camera.position.clone()));
      renderer.render(scene, camera);
    };
    animate();
    three.current = { mesh, material, camera, controls };
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      material.uniforms.uVol.value?.dispose();
      renderer.dispose();
      box.removeChild(renderer.domElement);
      three.current = null;
    };
  }, []);

  // voxels -> 3D texture
  useEffect(() => {
    const t = three.current;
    if (!t || !preview) return;
    const [nz, ny, nx] = preview.header.shape;
    const tex = new THREE.Data3DTexture(preview.voxels, nx, ny, nz);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    t.material.uniforms.uVol.value?.dispose();
    t.material.uniforms.uVol.value = tex;
    t.material.uniforms.uSteps.value = Math.min(1024, Math.round(1.7 * Math.max(nx, ny, nz)));
    t.mesh.visible = true;
    t.camera.position.set(0, 0, 1.9);
    t.controls.target.set(0, 0, 0);
  }, [preview]);

  // physical proportions (user dz / px), contrast, z slab
  useEffect(() => {
    const t = three.current;
    if (!t || !preview) return;
    const [nz, ny, nx] = preview.header.shape;
    const [sz, sy, sx] = preview.header.stride;
    const ext = [nx * sx * px, ny * sy * px, nz * sz * dz]; // µm
    const m = Math.max(...ext) || 1;
    t.mesh.scale.set(ext[0] / m, ext[1] / m, ext[2] / m);
    t.material.uniforms.uLo.value = contrast[0] / 255;
    t.material.uniforms.uHi.value = contrast[1] / 255;
    t.material.uniforms.uBoxMin.value.z = zRange[0] / nz - 0.5;
    t.material.uniforms.uBoxMax.value.z = (zRange[1] + 1) / nz - 0.5;
  }, [preview, dz, px, contrast, zRange]);

  const header = preview?.header;
  const snippet = useMemo(() => napariSnippet(header, dz, px), [header, dz, px]);
  const copy = () => {
    navigator.clipboard?.writeText(snippet).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  const nz = header?.shape[0] || 1;
  const stride = header?.stride[0] || 1;

  return (
    <Box>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1, flexWrap: "wrap" }}>
        <TextField
          select
          size="small"
          label="Volume"
          value={volumes.some((v) => v.path === path) ? path : ""}
          onChange={(e) => setPath(e.target.value)}
          sx={{ minWidth: 320, flex: 1 }}
        >
          {volumes.map((v) => (
            <MenuItem key={v.path} value={v.path} dense>
              {v.folder} / {v.name}
            </MenuItem>
          ))}
        </TextField>
        <Tooltip title="Reload the list">
          <IconButton onClick={refresh}>
            <Refresh />
          </IconButton>
        </Tooltip>
        {header?.kind === "tiff" && (
          <Button size="small" startIcon={<Download />} href={dataUrl(ip, apiPort, path)} download>
            Download
          </Button>
        )}
        {header?.kind === "zarr" && (
          <Button size="small" startIcon={<ViewInAr />} onClick={() => setShowSlices((s) => !s)}>
            {showSlices ? "3D view" : "2D slices"}
          </Button>
        )}
      </Box>
      {!volumes.length && !error && (
        <Typography variant="body2" color="text.secondary">
          No lightsheet volumes in the data folder yet. Run a scan first.
        </Typography>
      )}
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 1 }}>
          {error}
        </Alert>
      )}
      {loading && <LinearProgress sx={{ height: 2 }} />}

      <Box sx={{ display: "flex", flexDirection: { xs: "column", md: "row" }, gap: 2 }}>
      <Box sx={{ flex: 1, minWidth: 0 }}>
      <Box sx={{ display: showSlices ? "none" : "block", position: "relative" }}>
        <Box ref={boxRef} sx={{ width: "100%", height, bgcolor: "#000", borderRadius: 1, overflow: "hidden" }} />
        {header && (
          <Typography
            variant="caption"
            sx={{ position: "absolute", left: 8, bottom: 6, color: "#bbb", fontFamily: "monospace" }}
          >
            {header.fullShape.join(" × ")} (z·y·x) · preview {header.shape.join(" × ")} · MIP · drag to rotate
          </Typography>
        )}
      </Box>
      {showSlices && header?.kind === "zarr" && (
        <ErrorBoundary>
          <Box sx={{ height }}>
            <VizarrViewer zarrUrl={path} embedded height="100%" width="100%" />
          </Box>
        </ErrorBoundary>
      )}

      </Box>
      {header && (
        <Box sx={{ width: { xs: "100%", md: 380 }, flexShrink: 0, maxHeight: { md: height }, overflowY: "auto", overflowX: "hidden", pt: 1 }}>
          <Box>
            <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
              <TextField
                size="small"
                type="number"
                label="dz (µm per plane)"
                value={dz}
                onChange={(e) => setDz(Math.max(1e-3, parseFloat(e.target.value) || 1e-3))}
                helperText={`file: ${fmt(header.zStepUm)} µm`}
                inputProps={{ step: 0.1, min: 0 }}
              />
              <TextField
                size="small"
                type="number"
                label="xy pixel size (µm)"
                value={px}
                onChange={(e) => setPx(Math.max(1e-4, parseFloat(e.target.value) || 1e-4))}
                helperText={`file: ${fmt(header.pixelSizeUm)} µm`}
                inputProps={{ step: 0.01, min: 0 }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              Contrast
            </Typography>
            <Slider size="small" min={0} max={255} value={contrast} onChange={(_, v) => setContrast(v)} />
            <Typography variant="caption" color="text.secondary">
              Planes {zRange[0] * stride + 1}–{Math.min((zRange[1] + 1) * stride, header.fullShape[0])} of{" "}
              {header.fullShape[0]}
            </Typography>
            <Slider size="small" min={0} max={Math.max(0, nz - 1)} value={zRange} onChange={(_, v) => setZRange(v)} />
            <Typography variant="caption" color="text.secondary" component="div">
              Contrast is stretched over the {fmt(header.range[0])}–{fmt(header.range[1])} counts of the file. If the
              volume looks squashed or stretched along z, dz is off: the continuous scan estimates it from speed × time.
            </Typography>
          </Box>
          <Box sx={{ mt: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center" }}>
              <Typography variant="subtitle2" sx={{ flex: 1 }}>
                Open in napari
              </Typography>
              <Tooltip title={copied ? "Copied" : "Copy"}>
                <IconButton size="small" onClick={copy}>
                  <ContentCopy fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <Box
              component="pre"
              sx={{
                m: 0, p: 1, fontSize: 11, bgcolor: "action.hover", borderRadius: 1, overflow: "auto",
                whiteSpace: "pre-wrap", wordBreak: "break-all",
              }}
            >
              {snippet}
            </Box>
            <Typography variant="caption" color="text.secondary">
              Paste into napari's console (or a script). The path is on the ImSwitch computer; when napari runs
              elsewhere, download the file first and change the path. scale=(dz, y, x) gives the true proportions.
            </Typography>
          </Box>
        </Box>
      )}
      </Box>
    </Box>
  );
};

export default LightsheetVolumeViewer;
