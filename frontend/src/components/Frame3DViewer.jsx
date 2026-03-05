import React, { useEffect, useRef, useCallback } from "react";
import { Box } from "@mui/material";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

/**
 * Frame3DViewer – Three.js component that loads the FRAME microscope GLB model
 * and drives the stage (nodes 51-55) and objective turret (nodes 56-63) groups
 * from live stage positions coming through Redux.
 *
 * The base (nodes 1-50) is static – it is never moved by positioning data.
 *
 * @param {Object}   positions    – current microscope positions { x, y, z, a }
 * @param {Object}   axisConfig   – mapping config per logical axis (see Frame3DViewerSlice)
 * @param {Object}   visibility   – { base, stage, turret } booleans
 * @param {Object}   cameraState  – persisted camera { position, target }
 * @param {Function} onCameraChange – callback for camera persistence
 * @param {number}   width
 * @param {number}   height
 */
const Frame3DViewer = ({
  positions = { x: 0, y: 0, z: 0, a: 0 },
  axisConfig = {},
  visibility = { base: true, stage: true, turret: true },
  cameraState = null,
  onCameraChange = null,
  width = 600,
  height = 400,
}) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const groupsRef = useRef(null);
  const basePosRef = useRef({}); // original world positions of groups after loading
  const animationIdRef = useRef(null);
  const cameraTimeoutRef = useRef(null);

  // ------------------------------------------------------------------
  // Helper: apply offset / scale / invert to a raw position value
  // ------------------------------------------------------------------
  const applyMapping = useCallback(
    (rawValue, configKey) => {
      const cfg = axisConfig[configKey];
      if (!cfg) return 0;
      let v = (rawValue + cfg.offset) * cfg.scale;
      if (cfg.invert) v = -v;
      return v;
    },
    [axisConfig]
  );

  // ------------------------------------------------------------------
  // GLB helpers (same logic as frame3d.html)
  // ------------------------------------------------------------------
  const extractIndex = (name) => {
    const m = (name || "").match(/(\d+)\s*$/);
    return m ? parseInt(m[1], 10) : null;
  };

  const collectByIndexRange = (root, minIdx, maxIdx) => {
    const hits = [];
    root.traverse((obj) => {
      const idx = extractIndex(obj.name);
      if (idx == null) return;
      if (idx >= minIdx && idx <= maxIdx) hits.push(obj);
    });
    return hits;
  };

  const filterTopMost = (objects) => {
    const set = new Set(objects);
    return objects.filter((o) => {
      let p = o.parent;
      while (p) {
        if (set.has(p)) return false;
        p = p.parent;
      }
      return true;
    });
  };

  const attachAllPreserveWorld = (targetGroup, objects) => {
    const top = filterTopMost(objects);
    top.forEach((obj) => targetGroup.attach(obj));
    return top.length;
  };

  // ------------------------------------------------------------------
  // Initialise Three.js scene (runs once)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!containerRef.current) return;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100000);
    camera.position.set(250, 160, 250);
    cameraRef.current = camera;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(2, 3, 4);
    scene.add(dirLight);

    // Axes helper
    scene.add(new THREE.AxesHelper(50));

    // Assembly root + sub-groups
    const assemblyRoot = new THREE.Group();
    scene.add(assemblyRoot);

    const groups = {
      base: new THREE.Group(),
      stageXY: new THREE.Group(),
      turretY: new THREE.Group(),
    };
    Object.values(groups).forEach((g) => assemblyRoot.add(g));
    groupsRef.current = groups;

    // Load GLB
    const glbUrl = `${process.env.PUBLIC_URL}/assets/FRAME_reduced-compressed.glb`;
    const loader = new GLTFLoader();

    console.log("[Frame3D] Loading GLB from:", glbUrl);

    loader.load(
      glbUrl,
      (gltf) => {
        const model = gltf.scene;
        assemblyRoot.add(model);

        // Debug: log numeric-suffix nodes
        const numericNames = [];
        model.traverse((o) => {
          const idx = extractIndex(o.name);
          if (idx != null) numericNames.push({ name: o.name, idx, type: o.type });
        });
        numericNames.sort((a, b) => a.idx - b.idx);
        console.log("[Frame3D] numeric-suffix nodes:", numericNames);

        // Grouping: base 1..50, stage 51..55, turret 56..63
        const baseNodes = collectByIndexRange(model, 1, 50);
        const stageNodes = collectByIndexRange(model, 51, 55);
        const turretNodes = collectByIndexRange(model, 56, 63);

        attachAllPreserveWorld(groups.base, baseNodes);
        attachAllPreserveWorld(groups.stageXY, stageNodes);
        attachAllPreserveWorld(groups.turretY, turretNodes);

        console.log(
          `[Frame3D] base: ${baseNodes.length}, stage: ${stageNodes.length}, turret: ${turretNodes.length}`
        );

        // Store original positions so we can add offsets
        basePosRef.current = {
          stageXY: groups.stageXY.position.clone(),
          turretY: groups.turretY.position.clone(),
        };

        // Camera fit
        const box = new THREE.Box3().setFromObject(assemblyRoot);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        controls.target.copy(center);

        const maxDim = Math.max(size.x, size.y, size.z);
        camera.near = Math.max(0.1, maxDim / 1000);
        camera.far = maxDim * 50;

        // Restore persisted camera if available
        if (cameraState && cameraState.position) {
          camera.position.set(cameraState.position.x, cameraState.position.y, cameraState.position.z);
          if (cameraState.target) {
            controls.target.set(cameraState.target.x, cameraState.target.y, cameraState.target.z);
          }
        } else {
          camera.position.copy(center).add(new THREE.Vector3(maxDim * 1.2, maxDim * 0.6, maxDim * 1.2));
        }
        camera.updateProjectionMatrix();
        controls.update();
      },
      undefined,
      (err) => console.error("[Frame3D] Failed to load GLB:", err)
    );

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Debounced camera persistence
    const saveCameraState = () => {
      if (cameraTimeoutRef.current) clearTimeout(cameraTimeoutRef.current);
      cameraTimeoutRef.current = setTimeout(() => {
        if (onCameraChange && cameraRef.current && controlsRef.current) {
          const cp = cameraRef.current.position;
          const ct = controlsRef.current.target;
          onCameraChange({
            position: { x: cp.x, y: cp.y, z: cp.z },
            target: { x: ct.x, y: ct.y, z: ct.z },
          });
        }
      }, 500);
    };
    controls.addEventListener("change", saveCameraState);

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
      const w = containerRef.current.clientWidth || width;
      const h = containerRef.current.clientHeight || height;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      controls.removeEventListener("change", saveCameraState);
      if (cameraTimeoutRef.current) clearTimeout(cameraTimeoutRef.current);
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  // ------------------------------------------------------------------
  // Update positions when microscope positions or axis config change
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!groupsRef.current || !basePosRef.current.stageXY) return;

    const bp = basePosRef.current;
    const g = groupsRef.current;

    // Stage group: driven by stageX / stageY / stageZ mappings
    const sx = applyMapping(positions[axisConfig.stageX?.microscopeAxis] || 0, "stageX");
    const sy = applyMapping(positions[axisConfig.stageY?.microscopeAxis] || 0, "stageY");
    const sz = applyMapping(positions[axisConfig.stageZ?.microscopeAxis] || 0, "stageZ");

    g.stageXY.position.set(
      bp.stageXY.x + sx,
      bp.stageXY.y + sy,
      bp.stageXY.z + sz
    );

    // Turret group: driven by turretX mapping (single axis – usually the objective focus)
    const tx = applyMapping(positions[axisConfig.turretX?.microscopeAxis] || 0, "turretX");
    g.turretY.position.set(
      bp.turretY.x + tx,
      bp.turretY.y,
      bp.turretY.z
    );
  }, [positions, axisConfig, applyMapping]);

  // ------------------------------------------------------------------
  // Visibility toggles
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!groupsRef.current) return;
    groupsRef.current.base.visible = visibility.base;
    groupsRef.current.stageXY.visible = visibility.stage;
    groupsRef.current.turretY.visible = visibility.turret;
  }, [visibility]);

  // ------------------------------------------------------------------
  // Restore camera when Redux cameraState changes externally
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!cameraRef.current || !controlsRef.current || !cameraState) return;
    if (cameraState.position) {
      cameraRef.current.position.set(cameraState.position.x, cameraState.position.y, cameraState.position.z);
      if (cameraState.target) {
        controlsRef.current.target.set(cameraState.target.x, cameraState.target.y, cameraState.target.z);
      }
      controlsRef.current.update();
    }
  }, [cameraState]);

  return (
    <Box
      ref={containerRef}
      sx={{
        width,
        height,
        position: "relative",
        border: "1px solid #333",
        borderRadius: 1,
        overflow: "hidden",
        backgroundColor: "#1a1a1a",
      }}
    />
  );
};

export default Frame3DViewer;
