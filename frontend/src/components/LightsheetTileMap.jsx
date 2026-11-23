/**
 * Perspective overview of the scanned volumes: one box per tile, FOV wide (stage X/Y)
 * and as tall as the scan range (scan axis, drawn upwards).
 *
 * grey outline = planned, green = acquired, orange = being acquired,
 * blue = earlier scans (from their *.scan.json), red line = stage position.
 */
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Box, Button, Typography } from "@mui/material";

const COLORS = { planned: 0x9e9e9e, done: 0x4caf50, current: 0xff9800, past: 0x42a5f5, stage: 0xf44336 };

function tileBox(tile, fov, zRange, color, opacity) {
  const [fx, fy] = fov;
  const z0 = Math.min(...zRange);
  const dz = Math.max(Math.abs(zRange[1] - zRange[0]), Math.min(fx, fy) * 0.02); // keep flat scans visible
  const geometry = new THREE.BoxGeometry(fx, dz, fy);
  const group = new THREE.Group();
  group.position.set(tile.x, z0 + dz / 2, tile.y); // three: x = stage X, y = scan axis, z = stage Y
  if (opacity > 0) {
    group.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity, depthWrite: false,
    })));
  }
  group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color })));
  return group;
}

const LightsheetTileMap = ({
  tiles = [], // planned: [{x, y}]
  fov = [1000, 1000], // µm (stage X, Y) of one frame
  zRange = [0, 1000], // scan-axis positions
  currentTile = 0, // 1-based, 0 = none
  isRunning = false,
  pastScans = [], // scan records: {name, folder, tiles, fovUm, zRangeUm}
  stage = null, // {x, y, z}: z = scan-axis position
  height = 360,
}) => {
  const boxRef = useRef(null);
  const three = useRef(null);
  const [hover, setHover] = useState("");

  useEffect(() => {
    const box = boxRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    box.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121212);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const world = new THREE.Group(); // µm inside, scaled to ~1 unit for the camera
    scene.add(world);
    const grid = new THREE.GridHelper(4, 20, 0x444444, 0x2a2a2a);
    scene.add(grid);

    const resize = () => {
      renderer.setSize(box.clientWidth || 600, box.clientHeight || 360);
      camera.aspect = (box.clientWidth || 600) / (box.clientHeight || 360);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(box);
    resize();

    const raycaster = new THREE.Raycaster();
    const onMove = (e) => {
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster.intersectObjects(world.children, true).find((h) => h.object.isMesh && h.object.userData.label);
      setHover(hit ? hit.object.userData.label : "");
    };
    renderer.domElement.addEventListener("pointermove", onMove);

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    three.current = { world, camera, controls, grid, fitted: false };
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointermove", onMove);
      controls.dispose();
      renderer.dispose();
      box.removeChild(renderer.domElement);
      three.current = null;
    };
  }, []);

  const fit = () => {
    const t = three.current;
    if (!t) return;
    t.camera.position.set(1.4, 1.1, 1.6);
    t.controls.target.set(0, 0, 0);
    t.controls.update();
  };

  // Rebuild the boxes whenever the plan, the progress or the history changes.
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    const { world } = t;
    world.traverse((o) => {
      o.geometry?.dispose();
      o.material?.dispose();
    });
    world.clear();

    pastScans.forEach((scan) => {
      (scan.tiles || []).forEach((tile) => {
        if (!scan.fovUm || !scan.zRangeUm) return;
        const g = tileBox(tile, scan.fovUm, scan.zRangeUm, COLORS.past, tile.done === false ? 0.04 : 0.12);
        g.children[0].userData.label = `${scan.folder} / ${scan.name}`;
        world.add(g);
      });
    });
    tiles.forEach((tile, i) => {
      const state = !isRunning && currentTile === 0 ? "planned"
        : i + 1 < currentTile ? "done" : i + 1 === currentTile ? "current" : "planned";
      const g = tileBox(tile, fov, zRange, COLORS[state], state === "planned" ? 0 : 0.3);
      world.add(g);
    });
    if (stage) {
      const z = [Math.min(...zRange), Math.max(...zRange)];
      const pad = (z[1] - z[0]) * 0.3 + 1;
      const line = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(stage.x, z[0] - pad, stage.y), new THREE.Vector3(stage.x, z[1] + pad, stage.y),
      ]);
      world.add(new THREE.Line(line, new THREE.LineBasicMaterial({ color: COLORS.stage })));
      if (Number.isFinite(stage.z)) {
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(Math.min(...fov) * 0.04, 12, 8),
          new THREE.MeshBasicMaterial({ color: COLORS.stage }),
        );
        dot.position.set(stage.x, stage.z, stage.y);
        world.add(dot);
      }
    }

    // Frame the current plan (and the stage): its centre at the origin, ~1.5 units across.
    const xs = tiles.map((p) => p.x).concat(stage ? [stage.x] : []);
    const ys = tiles.map((p) => p.y).concat(stage ? [stage.y] : []);
    if (!xs.length) return;
    const span = Math.max(Math.max(...xs) - Math.min(...xs) + fov[0], Math.max(...ys) - Math.min(...ys) + fov[1],
      Math.abs(zRange[1] - zRange[0]), 1);
    const s = 1.5 / span;
    world.scale.setScalar(s);
    world.position.set(-(Math.max(...xs) + Math.min(...xs)) / 2 * s, -Math.min(...zRange) * s,
      -(Math.max(...ys) + Math.min(...ys)) / 2 * s);
    if (!t.fitted) {
      t.fitted = true;
      fit();
    }
  }, [tiles, fov, zRange, currentTile, isRunning, pastScans, stage]);

  const done = isRunning ? Math.max(0, currentTile - 1) : 0;
  return (
    <Box sx={{ position: "relative" }}>
      <Box ref={boxRef} sx={{ width: "100%", height, borderRadius: 1, overflow: "hidden" }} />
      <Button size="small" onClick={fit} sx={{ position: "absolute", top: 4, right: 4, minWidth: 0, textTransform: "none" }}>
        Reset view
      </Button>
      <Box sx={{ display: "flex", flexWrap: "wrap", columnGap: 1.5, mt: 0.5 }}>
        {[["planned", "planned"], ["done", "acquired"], ["current", "now"], ["past", "earlier scans"], ["stage", "stage"]]
          .map(([key, label]) => (
            <Typography key={key} variant="caption" sx={{ color: `#${COLORS[key].toString(16).padStart(6, "0")}` }}>
              ■ {label}
            </Typography>
          ))}
      </Box>
      <Typography variant="caption" sx={{ position: "absolute", top: 6, left: 8, color: "#bbb", pr: 10 }}>
        {hover || `${tiles.length} tile(s) · ${fov.map((v) => Math.round(v)).join(" × ")} µm each · ` +
          `${Math.round(Math.abs(zRange[1] - zRange[0]))} µm deep${isRunning ? ` · ${done}/${tiles.length} done` : ""}`}
      </Typography>
    </Box>
  );
};

export default LightsheetTileMap;
