import React, { useEffect, useRef } from "react";
import { Box } from "@mui/material";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

/**
 * Lightsheet3DViewer - Three.js-based 3D visualization of lightsheet microscope assembly
 * Shows real-time position of objective, sample, and lightsheet based on stage positions
 * 
 * @param {Object} positions - Current stage positions: { x, y, z, a }
 * @param {Object} axisConfig - Per-axis configuration: { x: {offset, scale, invert}, y: {...}, z: {...}, a: {...} }
 * @param {Object} cameraState - Saved camera state: { position: {x,y,z}, target: {x,y,z}, zoom }
 * @param {Function} onCameraChange - Callback when camera position changes (for persistence)
 * @param {number} width - Viewer width in pixels
 * @param {number} height - Viewer height in pixels
 */
const Lightsheet3DViewer = ({ 
  positions = { x: 0, y: 0, z: 0, a: 0 },
  axisConfig = {
    x: { offset: 0, scale: 1, invert: false },
    y: { offset: 0, scale: 1, invert: false },
    z: { offset: 0, scale: 1, invert: false },
    a: { offset: 0, scale: 1, invert: false }
  },
  cameraState = null,
  onCameraChange = null,
  width = 600,
  height = 400
}) => {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const groupsRef = useRef(null);
  const animationIdRef = useRef(null);
  const cameraTimeoutRef = useRef(null);

  // Apply axis configuration (offset, scale, invert) to raw position
  // Offset is used for calibration - applied once to compensate for hardware zero points
  // Scale adjusts units, invert changes direction
  const applyAxisConfig = (rawValue, axisName) => {
    const config = axisConfig[axisName] || { offset: 0, scale: 1, invert: false };
    // Apply offset first (calibration), then scale, then invert direction
    let value = (rawValue + config.offset) * config.scale;
    if (config.invert) {
      value = -value;
    }
    return value;
  };

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Setup scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Setup camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100000);
    camera.position.set(200, 120, 200);
    cameraRef.current = camera;

    // Setup controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    // Add lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(2, 3, 4);
    scene.add(directionalLight);

    // Add axes helper
    scene.add(new THREE.AxesHelper(50));

    // Create assembly root and subgroups
    const assemblyRoot = new THREE.Group();
    scene.add(assemblyRoot);

    const groups = {
      lightsheet: new THREE.Group(),
      sample: new THREE.Group(),
      objective4x: new THREE.Group(),
      objective20x: new THREE.Group(),
      chamberFixed: new THREE.Group()
    };

    Object.values(groups).forEach(g => assemblyRoot.add(g));
    groupsRef.current = groups;

    // Helper function to attach objects by name
    const attachByName = (model, nodeName, targetGroup) => {
      const obj = model.getObjectByName(nodeName);
      if (!obj) {
        console.warn(`[GLB] missing node: ${nodeName}`);
        return;
      }
      targetGroup.attach(obj);
    };

    // Helper to generate name lists
    const makeNameList = (prefix, from, to) => {
      const out = [];
      for (let i = from; i <= to; i++) out.push(`${prefix}${i}`);
      return out;
    };

    // Load GLB model
    // NOTE: The original model is an STP file. You need to convert it to GLB format
    // using tools like Blender, FreeCAD, or online converters
    // Place the converted GLB file in: /public/assets/Assembly_lightsheet_objective_sample_arrangement.glb
    
    // Use process.env.PUBLIC_URL to handle different deployment paths (e.g., /imswitch)
    const glbUrl = `${process.env.PUBLIC_URL}/assets/Assembly_lightsheet_objective_sample_arrangement.glb`;
    const loader = new GLTFLoader();

    console.log('[3D Viewer] Loading GLB from:', glbUrl);

    loader.load(
      glbUrl,
      (gltf) => {
        console.log('[3D Viewer] GLB loaded successfully');
        const model = gltf.scene;
        assemblyRoot.add(model);

        console.log("[GLB] children:", model.children.map(c => c.name));

        // Map GLB parts to groups
        attachByName(model, "Körper1_9", groups.lightsheet);
        
        ["Körper2_1", "Körper1_8"].forEach(n => attachByName(model, n, groups.sample));
        
        attachByName(model, "Körper1_7", groups.objective4x);
        
        ["Körper1", ...makeNameList("Körper1_", 1, 6)].forEach(n => 
          attachByName(model, n, groups.chamberFixed)
        );
        
        makeNameList("Körper", 2, 7).forEach(n => 
          attachByName(model, n, groups.objective20x)
        );
        attachByName(model, "Körper1_6", groups.objective20x);

        // Center camera on model
        const box = new THREE.Box3().setFromObject(assemblyRoot);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        controls.target.copy(center);

        const maxDim = Math.max(size.x, size.y, size.z);
        camera.near = Math.max(0.1, maxDim / 1000);
        camera.far = maxDim * 50;
        
        // Restore saved camera state if available, otherwise use default
        if (cameraState && cameraState.position) {
          camera.position.set(
            cameraState.position.x,
            cameraState.position.y,
            cameraState.position.z
          );
          if (cameraState.target) {
            controls.target.set(
              cameraState.target.x,
              cameraState.target.y,
              cameraState.target.z
            );
          }
          console.log('[3D Viewer] Restored saved camera position');
        } else {
          // Default camera position
          camera.position.copy(center).add(
            new THREE.Vector3(maxDim * 1.2, maxDim * 0.6, maxDim * 1.2)
          );
        }
        
        camera.updateProjectionMatrix();
        controls.update();
      },
      (progress) => {
        // Log loading progress
        if (progress.lengthComputable) {
          const percentComplete = (progress.loaded / progress.total) * 100;
          console.log(`[3D Viewer] Loading: ${percentComplete.toFixed(1)}%`);
        }
      },
      (err) => {
        console.error("[3D Viewer] Failed to load GLB:", err);
        console.error("[3D Viewer] GLB URL attempted:", glbUrl);
        console.error("[3D Viewer] Make sure the file exists at: /public/assets/Assembly_lightsheet_objective_sample_arrangement.glb");
      }
    );

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Save camera state on user interaction (debounced)
    const saveCameraState = () => {
      if (cameraTimeoutRef.current) clearTimeout(cameraTimeoutRef.current);
      cameraTimeoutRef.current = setTimeout(() => {
        if (onCameraChange && cameraRef.current && controlsRef.current) {
          const camPos = cameraRef.current.position;
          const camTarget = controlsRef.current.target;
          onCameraChange({
            position: { x: camPos.x, y: camPos.y, z: camPos.z },
            target: { x: camTarget.x, y: camTarget.y, z: camTarget.z },
          });
        }
      }, 500); // Debounce 500ms
    };

    // Listen to control changes
    if (controls) {
      controls.addEventListener('change', saveCameraState);
    }

    // Handle resize
    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return;
      
      const newWidth = containerRef.current?.clientWidth || width;
      const newHeight = containerRef.current?.clientHeight || height;
      
      cameraRef.current.aspect = newWidth / newHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(newWidth, newHeight);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      if (controls && saveCameraState) {
        controls.removeEventListener('change', saveCameraState);
      }
      if (cameraTimeoutRef.current) {
        clearTimeout(cameraTimeoutRef.current);
      }
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }
    };
  }, [width, height]); // Removed cameraState and onCameraChange to prevent re-render on every position change

  // Separate useEffect for restoring camera state only when it changes in Redux
  useEffect(() => {
    if (!cameraRef.current || !controlsRef.current || !cameraState) return;
    
    // Only restore if we have a saved camera state
    if (cameraState.position) {
      cameraRef.current.position.set(
        cameraState.position.x,
        cameraState.position.y,
        cameraState.position.z
      );
      if (cameraState.target) {
        controlsRef.current.target.set(
          cameraState.target.x,
          cameraState.target.y,
          cameraState.target.z
        );
      }
      controlsRef.current.update();
      console.log('[3D Viewer] Camera state updated from Redux');
    }
  }, [cameraState]);

  // Update positions based on stage movements
  useEffect(() => {
    if (!groupsRef.current) return;

    const groups = groupsRef.current;

    // Apply axis mappings with configuration
    // Z axis → Objective 20x Y-axis movement
    const zPos = applyAxisConfig(positions.z || 0, 'z');
    groups.sample.position.y = - zPos * 0.001; // Scale factor for model units

    // A axis → Sample Y movement
    const aPos = applyAxisConfig(positions.a || 0, 'a');
    groups.objective20x.position.y = - aPos * 0.001;

    // X axis → Sample X movement
    const xPos = applyAxisConfig(positions.x || 0, 'x');
    groups.sample.position.x = xPos * 0.001;

    // Y axis → Sample Z movement
    const yPos = applyAxisConfig(positions.y || 0, 'y');
    groups.sample.position.z = yPos * 0.001;

  }, [positions, axisConfig]);

  return (
    <Box
      ref={containerRef}
      sx={{
        width: width,
        height: height,
        position: "relative",
        border: "1px solid #333",
        borderRadius: 1,
        overflow: "hidden",
        backgroundColor: "#1a1a1a"
      }}
    />
  );
};

export default Lightsheet3DViewer;
