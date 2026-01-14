import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { Box, Slider, Typography, Paper } from "@mui/material";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import { processUC2FPacket, processUC2FPacketWithMetadata, checkFeatureSupport } from "../stream/uc2f.js";

/**
 * WebGL2-based 16-bit image viewer with window/level controls and zoom/pan
 * Receives binary UC2F packets via window events and renders via WebGL2 R16UI textures
 * Falls back to Canvas2D for unsupported browsers
 * 
 * @param {function} onClick - Callback for single click: (pixelX, pixelY, imageWidth, imageHeight, displayInfo)
 * @param {function} onDoubleClick - Callback for double click: (pixelX, pixelY, imageWidth, imageHeight)
 * @param {function} onImageLoad - Callback when image dimensions change: (width, height)
 * @param {function} onHudDataUpdate - Callback for HUD data updates
 * @param {React.ReactNode} overlayContent - Optional overlay content to render on top of the canvas
 */
const LiveViewerGL = ({ onClick, onDoubleClick, onImageLoad, onHudDataUpdate, overlayContent }) => {
  const dispatch = useDispatch();
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const glRef = useRef(null);
  const programRef = useRef(null);
  const textureRef = useRef(null);
  const vaoRef = useRef(null);
  const prevDimensionsRef = useRef({ width: 0, height: 0 }); // Track dimensions to avoid redundant callbacks
  
  // View state
  const [imageSize, setImageSize] = useState({ width: 100, height: 100 });
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  const [viewTransform, setViewTransform] = useState({ scale: 1, translateX: 0, translateY: 0 });
  const [featureSupport, setFeatureSupport] = useState({ webgl2: false, intTextures: false, lz4: false });
  const [stats, setStats] = useState({ fps: 0, bps: 0, compressionRatio: 0 });
  const [isWebGL, setIsWebGL] = useState(false);
  const [currentImageData, setCurrentImageData] = useState(null); // Store current image for histogram

  // Mouse interaction state
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  
  // FPS counter
  const fpsCounterRef = useRef({
    frames: 0,
    lastTime: performance.now(),
    bytesReceived: 0,
    compressedBytes: 0
  });

  // Histogram computation counter
  const histogramCounterRef = useRef(0);

  // WebGL2 shaders
  const vertexShaderSource = `#version 300 es
    in vec2 a_position;
    in vec2 a_texCoord;
    uniform mat3 u_transform;
    out vec2 v_texCoord;
    
    void main() {
      vec3 pos = u_transform * vec3(a_position, 1.0);
      gl_Position = vec4(pos.xy, 0.0, 1.0);
      v_texCoord = a_texCoord;
    }
  `;

  const fragmentShaderSource = `#version 300 es
    precision highp float;
    precision highp int;
    
    in vec2 v_texCoord;
    uniform highp usampler2D u_texture;
    uniform float u_min;
    uniform float u_max;
    uniform float u_gamma;
    out vec4 fragColor;
    
    void main() {
      // Force use of u_texture uniform so it doesn't get optimized away
      uint rawValue = texture(u_texture, v_texCoord).r;
      float normalized = float(rawValue);
      
      // Apply window/level
      float windowed = clamp((normalized - u_min) / (u_max - u_min), 0.0, 1.0);
      
      // Apply gamma correction
      float gamma_corrected = pow(windowed, 1.0 / u_gamma);
      
      fragColor = vec4(gamma_corrected, gamma_corrected, gamma_corrected, 1.0);
    }
  `;

  // Canvas2D fallback shader (software implementation)
  const applyWindowLevelGamma = useCallback((imageData, width, height, minVal, maxVal, gamma) => {
    const data = imageData.data;
    const scale = 255.0 / Math.max(1, maxVal - minVal);
    
    for (let i = 0; i < data.length; i += 4) {
      // Get luminance (assuming grayscale)
      let value = data[i]; // Use red channel
      
      // Apply window/level
      let windowed;
      if (value <= minVal) {
        windowed = 0;
      } else if (value >= maxVal) {
        windowed = 255;
      } else {
        windowed = (value - minVal) * scale;
      }
      
      // Apply gamma
      const gamma_corrected = Math.pow(windowed / 255.0, 1.0 / gamma) * 255;
      
      data[i] = gamma_corrected;     // R
      data[i + 1] = gamma_corrected; // G  
      data[i + 2] = gamma_corrected; // B
      // Alpha channel (i + 3) remains unchanged
    }
  }, []);

  // Initialize WebGL2 context and shaders
  const initWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      console.warn('Canvas not available for WebGL initialization');
      return false;
    }


    // Set initial canvas size - will be updated when image arrives
    if (canvas.width === 0 || canvas.height === 0) {
      canvas.width = 800;
      canvas.height = 600;
    }

    const gl = canvas.getContext('webgl2');
    if (!gl) {
      console.warn('WebGL2 not supported, falling back to Canvas2D');
      return false;
    }

    console.log('WebGL2 context created successfully');
    glRef.current = gl;

    // Check for integer texture support
    if (!gl.R16UI || !gl.RED_INTEGER || !gl.UNSIGNED_SHORT) {
      console.warn('Integer textures not supported, falling back to Canvas2D');
      return false;
    }

    // Create and compile shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);

    // Check compilation
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
      console.error('Vertex shader compilation failed:', gl.getShaderInfoLog(vertexShader));
      return false;
    }
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
      console.error('Fragment shader compilation failed:', gl.getShaderInfoLog(fragmentShader));
      return false;
    }
    console.log('Shaders compiled successfully');

    // Create program
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program linking failed:', gl.getProgramInfoLog(program));
      return false;
    }
    console.log('Shader program linked successfully');

    programRef.current = program;

    // Set up geometry (fullscreen quad for triangle strip)
    const positions = new Float32Array([
      -1, -1,  0, 1,  // bottom-left
       1, -1,  1, 1,  // bottom-right  
      -1,  1,  0, 0,  // top-left
       1,  1,  1, 0,  // top-right
    ]);

    // Create and bind vertex array object FIRST
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    // THEN create buffer and set up attributes while VAO is bound
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    // Set up vertex attributes
    const positionLocation = gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');

    if (positionLocation < 0 || texCoordLocation < 0) {
      console.error('Failed to get attribute locations');
      return false;
    }

    // Configure vertex attributes
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(texCoordLocation);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 16, 8);

    // Save VAO for rendering
    vaoRef.current = vao;

    // Create texture
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    textureRef.current = texture;

    // Upload a random initial 16-bit texture so we can immediately verify the GL pipeline.
    try {
      const w = canvas.width;
      const h = canvas.height;
      // Create initial empty texture
      const size = Math.max(1, w * h);
      const emptyData = new Uint16Array(size);
      emptyData.fill(0); // Start with black

      // Ensure correct row alignment for 16-bit uploads
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

      // Set up proper GL state
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      
      // Bind texture to unit 0
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R16UI,
        w,
        h,
        0,
        gl.RED_INTEGER,
        gl.UNSIGNED_SHORT,
        emptyData
      );

      // Perform an initial draw to the canvas so the noise is visible immediately
      // Bind VAO (attributes) and program, set basic uniforms
      if (vaoRef.current) {
        gl.bindVertexArray(vaoRef.current);
      }
      else{
        console.warn('No VAO available for initial draw');
      }
      gl.useProgram(program);
      
      // Ensure GL state is correct for 2D blitting
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.BLEND);
      
      // Set texture uniform to texture unit 0
      const textureLocation = gl.getUniformLocation(program, 'u_texture');
      if (textureLocation !== null) {
        gl.uniform1i(textureLocation, 0);
        console.log('Set u_texture uniform to unit 0 - SUCCESS');
      } else {
        console.warn('u_texture uniform not found in shader - creating fallback');
        // The uniform was optimized out, use a simple fallback shader
        return false; // Let the fallback logic in the caller handle this
      }

      const minLocation = gl.getUniformLocation(program, 'u_min');
      const maxLocation = gl.getUniformLocation(program, 'u_max');
      const gammaLocation = gl.getUniformLocation(program, 'u_gamma');
      const transformLocation = gl.getUniformLocation(program, 'u_transform');

      if (minLocation >= 0) gl.uniform1f(minLocation, 0.0);
      if (maxLocation >= 0) gl.uniform1f(maxLocation, 65535.0);
      if (gammaLocation >= 0) gl.uniform1f(gammaLocation, 1.0);
      if (transformLocation >= 0) {
        const transform = new Float32Array([1,0,0, 0,1,0, 0,0,1]);
        gl.uniformMatrix3fv(transformLocation, false, transform);
      }

      gl.viewport(0, 0, w, h);
      
      // Check for errors before drawing
      let preDrawError = gl.getError();
      if (preDrawError !== gl.NO_ERROR) {
        console.error('GL error before draw:', preDrawError);
      }
      
      gl.clearColor(0,0,0,1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      

      
      // Check if VAO is properly bound
      if (!vaoRef.current) {
        console.error('No VAO bound before draw!');
        return false;
      }
      
      // Use the uniform locations already declared above
      gl.useProgram(program);
      
      // Validate the GL state before drawing
      const currentVAO = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
      if (!currentVAO) {
        console.error('No VAO bound - cannot draw');
        return false;
      }
      
      // Validate the program
      gl.validateProgram(program);
      if (!gl.getProgramParameter(program, gl.VALIDATE_STATUS)) {
        console.error('Program validation failed:', gl.getProgramInfoLog(program));
        return false;
      }
      
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); // Use TRIANGLE_STRIP with 4 vertices

      const err = gl.getError();
      if (err !== gl.NO_ERROR) {
        console.error('WebGL draw error:', err);
      }
    } catch (e) {
      console.warn('Initial random texture setup failed:', e);
    }
    console.log('WebGL initialization complete');
    return true;
  }, []);

  // Upload 16-bit data to WebGL2 texture
  const uploadTexture = useCallback((u16Data, width, height) => {
    const gl = glRef.current;
    const texture = textureRef.current;
    
    if (!gl || !texture) {
      console.warn('WebGL texture upload failed: missing GL context or texture');
      return;
    }

    // Debug: check data validity
    if (!u16Data || u16Data.length === 0) {
      console.warn('WebGL texture upload failed: empty or invalid data');
      return;
    }
    
    const expectedSize = width * height;
    if (u16Data.length !== expectedSize) {
      console.warn(`WebGL texture upload: data size mismatch. Expected ${expectedSize}, got ${u16Data.length}`);
    }
    


    // Ensure correct row alignment for 16-bit uploads
    try {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,                    // level
        gl.R16UI,            // internal format
        width,
        height,
        0,                    // border
        gl.RED_INTEGER,      // format
        gl.UNSIGNED_SHORT,   // type
        u16Data
      );

      // Check for WebGL errors
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        console.error(`WebGL texture upload error: ${error}`);
      } 
    } catch (error) {
      console.error('WebGL texture upload exception:', error);
    }
  }, []);

  // Render frame with current uniforms
  const renderFrame = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    
    if (!gl || !program || !texture) {
      console.warn('WebGL render: missing context, program, or texture');
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      console.warn('WebGL render: missing canvas');
      return;
    }



    // Canvas size should match image size for proper aspect ratio
    // The CSS styling will scale it to fit the container while maintaining aspect ratio

    // Set viewport to match canvas size
    gl.viewport(0, 0, canvas.width, canvas.height);
    
    // Clear canvas
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    // Ensure correct GL state for 2D rendering
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    
    // Bind VAO (attributes) to ensure correct vertex attributes and texcoords
    if (vaoRef.current) gl.bindVertexArray(vaoRef.current);
    gl.useProgram(program);

    // Bind texture to unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Update uniforms
    const minLocation = gl.getUniformLocation(program, 'u_min');
    const maxLocation = gl.getUniformLocation(program, 'u_max');
    const gammaLocation = gl.getUniformLocation(program, 'u_gamma');
    const transformLocation = gl.getUniformLocation(program, 'u_transform');
    const textureLocation = gl.getUniformLocation(program, 'u_texture');

    // Set texture uniform
    if (textureLocation !== null) {
      gl.uniform1i(textureLocation, 0);
    }

    // Get window/level values (no auto-windowing)
    const minVal = liveStreamState.minVal || 0;
    const maxVal = liveStreamState.maxVal || 65535;
    const gamma = liveStreamState.gamma || 1.0;

    // Set uniforms
    if (minLocation !== null) gl.uniform1f(minLocation, minVal);
    if (maxLocation !== null) gl.uniform1f(maxLocation, maxVal);
    if (gammaLocation !== null) gl.uniform1f(gammaLocation, gamma);

    // Apply view transform
    const { scale, translateX, translateY } = viewTransform;
    if (transformLocation !== null) {
      const transform = new Float32Array([
        scale, 0, translateX,
        0, scale, translateY,
        0, 0, 1
      ]);
      gl.uniformMatrix3fv(transformLocation, false, transform);
    }

    // Draw the quad (use TRIANGLE_STRIP with 4 vertices)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Check for WebGL errors
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      console.error(`WebGL render error: ${error}`);
    }
  }, [liveStreamState.minVal, liveStreamState.maxVal, liveStreamState.gamma, viewTransform.scale, viewTransform.translateX, viewTransform.translateY]);

  // Mouse event handlers for zoom and pan
  const handleMouseDown = useCallback((e) => {
    // Only handle left mouse button for panning
    if (e.button === 0) {
      setIsDragging(true);
      setLastMousePos({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    }
    console.log('Mouse down, xy:', e.clientX, e.clientY);
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const deltaX = e.clientX - lastMousePos.x;
    const deltaY = e.clientY - lastMousePos.y;
    
    // Convert pixel movement to normalized coordinates
    const rect = canvas.getBoundingClientRect();
    const normalizedDeltaX = (deltaX / rect.width) * 2.0;
    const normalizedDeltaY = -(deltaY / rect.height) * 2.0; // Invert Y
    console.log(`Mouse move delta: ${deltaX}, ${deltaY} => normalized: ${normalizedDeltaX.toFixed(3)}, ${normalizedDeltaY.toFixed(3)}`);
    setViewTransform(prev => ({
      ...prev,
      translateX: prev.translateX + normalizedDeltaX / prev.scale,
      translateY: prev.translateY + normalizedDeltaY / prev.scale
    }));
    
    setLastMousePos({ x: e.clientX, y: e.clientY });
    e.preventDefault();
  }, [isDragging, lastMousePos]);

  const handleMouseUp = useCallback((e) => {
    console.log('Mouse up');
    setIsDragging(false);
    e.preventDefault();
  }, []);

  // Canvas2D fallback rendering
  const renderCanvas2D = useCallback((u16Data, width, height) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      console.warn('Canvas2D render: missing canvas');
      return;
    }

    // console.log(`Canvas2D render: ${width}x${height}, ${u16Data.length} pixels`);

    // Set canvas size to match image dimensions for proper aspect ratio
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext('2d');
    
    // Convert 16-bit to 8-bit with window/level
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    
    const minVal = liveStreamState.minVal || 0;
    const maxVal = liveStreamState.maxVal || 65535;
    const gamma = liveStreamState.gamma || 1.0;
    
    console.log(`Canvas2D window/level: min=${minVal}, max=${maxVal}, gamma=${gamma}`);
    
    // Ensure valid range
    let actualMin = minVal;
    let actualMax = maxVal;
    if (actualMin >= actualMax) {
      actualMin = 0;
      actualMax = 65535;
    }
    
    const scale = 255.0 / (actualMax - actualMin);
    
    for (let i = 0; i < u16Data.length; i++) {
      const value = u16Data[i];
      let scaled;
      
      // Apply window/level
      if (value <= actualMin) {
        scaled = 0;
      } else if (value >= actualMax) {
        scaled = 255;
      } else {
        scaled = (value - actualMin) * scale;
      }
      
      // Apply gamma correction
      const gamma_corrected = Math.pow(scaled / 255.0, 1.0 / gamma) * 255;
      
      const idx = i * 4;
      data[idx] = gamma_corrected;     // R
      data[idx + 1] = gamma_corrected; // G
      data[idx + 2] = gamma_corrected; // B
      data[idx + 3] = 255;             // A
    }
    
    // Clear and draw
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(imageData, 0, 0);
    
  }, [liveStreamState.minVal, liveStreamState.maxVal, liveStreamState.gamma]);

  // Update FPS stats
  const updateStats = useCallback((compressedSize, uncompressedSize) => {
    const counter = fpsCounterRef.current;
    counter.frames++;
    counter.bytesReceived += uncompressedSize;
    counter.compressedBytes += compressedSize;
    
    const now = performance.now();
    const elapsed = now - counter.lastTime;
    
    if (elapsed >= 1000) { // Update every second
      const fps = Math.round((counter.frames * 1000) / elapsed);
      const bps = Math.round((counter.bytesReceived * 8) / (elapsed / 1000)); // bits per second
      const compressionRatio = counter.bytesReceived > 0 ? 
        Math.round((counter.compressedBytes / counter.bytesReceived) * 100) / 100 : 0;
      
      setStats({ fps, bps, compressionRatio });
      
      // Update Redux stats
      dispatch(liveStreamSlice.setStats?.({ fps, bps, compressionRatio }));
      
      // Reset counters
      counter.frames = 0;
      counter.bytesReceived = 0;
      counter.compressedBytes = 0;
      counter.lastTime = now;
    }
  }, [dispatch]);

  // Compute histogram from 16-bit data (client-side)
  const computeHistogram = useCallback((u16Data, width, height) => {
    // Only compute histogram every 5th frame to avoid performance issues
    histogramCounterRef.current++;
    if (histogramCounterRef.current % 5 !== 0) return;

    // Use requestIdleCallback if available, otherwise setTimeout
    const computeAsync = (callback) => {
      if (window.requestIdleCallback) {
        window.requestIdleCallback(callback);
      } else {
        setTimeout(callback, 0);
      }
    };

    computeAsync(() => {
      try {
        const binCount = 4096; // Coarse histogram for performance
        const maxValue = 65535; // 16-bit max
        const binSize = maxValue / binCount;
        
        const histogram = new Array(binCount).fill(0);
        const histogramX = new Array(binCount);
        
        // Initialize x-axis values
        for (let i = 0; i < binCount; i++) {
          histogramX[i] = Math.round(i * binSize);
        }
        
        // Count pixel values
        for (let i = 0; i < u16Data.length; i++) {
          const value = u16Data[i];
          const bin = Math.min(Math.floor(value / binSize), binCount - 1);
          histogram[bin]++;
        }
        
        // Update Redux with histogram data
        dispatch(liveStreamSlice.setHistogramData({
          x: histogramX,
          y: histogram
        }));
        
      } catch (error) {
        console.warn('Histogram computation failed:', error);
      }
    });
  }, [dispatch]);

  // Handle binary frame events
  const handleFrameEvent = useCallback((event) => {
    try {
      // Extract buffer and metadata from event
      const { buffer, metadata } = event.detail;
      
      if (!buffer) {
        console.error('LiveViewerGL: No buffer in frame event');
        return;
      }
      
      let packet;
      if (metadata && metadata.compressed_bytes) {
        // Use metadata to parse frame correctly
        packet = processUC2FPacketWithMetadata(buffer, metadata);
      } else {
        // Fallback to original parsing
        packet = processUC2FPacket(buffer);
      }
      

      setImageSize({ width: packet.width, height: packet.height });
      setCurrentImageData(packet.dataU16); // Store for histogram computation
      
      // Log pixel data range for debugging (with one-time manual adjustment for testing)
      if (packet.dataU16.length > 0) {
        // Find actual min/max values from a sample of the data for debugging only
        const sampleSize = Math.min(10000, packet.dataU16.length);
        const step = Math.max(1, Math.floor(packet.dataU16.length / sampleSize));
        
        let actualMin = packet.dataU16[0];
        let actualMax = packet.dataU16[0];
        
        for (let i = 0; i < packet.dataU16.length; i += step) {
          const value = packet.dataU16[i];
          if (value < actualMin) actualMin = value;
          if (value > actualMax) actualMax = value;
        }
                
        // No auto-windowing - manual window/level control only
      }
      
      // Notify parent of image dimensions only if they changed
      if (onImageLoad && 
          (packet.width !== prevDimensionsRef.current.width || 
           packet.height !== prevDimensionsRef.current.height)) {
        prevDimensionsRef.current = { width: packet.width, height: packet.height };
        onImageLoad(packet.width, packet.height);
      }
      
      // Update stats
      updateStats(packet.compSize, packet.dataU16.length * 2);
      
      // Compute histogram (async, throttled)
      computeHistogram(packet.dataU16, packet.width, packet.height);
      
      if (isWebGL) {
        uploadTexture(packet.dataU16, packet.width, packet.height);
        renderFrame();
        
        // Force a re-render after Redux state update to ensure new window/level values are applied
        setTimeout(() => {
          if (isWebGL && glRef.current) {
            renderFrame();
          }
        }, 10); // Small delay to allow Redux state to update
      } else {
        renderCanvas2D(packet.dataU16, packet.width, packet.height);
      }
      
    } catch (error) {
      console.error('LiveViewerGL: Frame processing error:', error);
      console.error('LiveViewerGL: Error stack:', error.stack);
    }
  }, [isWebGL, uploadTexture, renderFrame, renderCanvas2D, onImageLoad, updateStats, computeHistogram, liveStreamState.minVal, liveStreamState.maxVal, dispatch]);

  // Initialize component
  useEffect(() => {
    
    const support = checkFeatureSupport();
    setFeatureSupport(support);
    
    const webglSuccess = support.webgl2 && initWebGL();
    setIsWebGL(webglSuccess);
    
    if (!webglSuccess) {
      console.log('LiveViewerGL: Using Canvas2D fallback');
    } else {
      console.log('LiveViewerGL: Using WebGL2 rendering');
    }
  }, [initWebGL]);

  // Listen for frame events
  useEffect(() => {
    window.addEventListener('uc2:frame', handleFrameEvent);
    
    return () => {      window.removeEventListener('uc2:frame', handleFrameEvent);
      window.removeEventListener('uc2:frame', handleFrameEvent);
    };
  }, [handleFrameEvent]);

  // Send HUD data to parent component
  useEffect(() => {
    if (onHudDataUpdate) {
      const hudData = {
        stats,
        featureSupport,
        isWebGL,
        imageSize,
        viewTransform
      };
      onHudDataUpdate(hudData);
    }
  }, [stats.fps, stats.bps, featureSupport, isWebGL, imageSize.width, imageSize.height, viewTransform.scale, viewTransform.translateX, viewTransform.translateY, onHudDataUpdate]);

  // Update canvas size when image size changes - but keep minimum size  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && imageSize.width > 0 && imageSize.height > 0) {
      // Only update if size actually changed
      if (canvas.width !== imageSize.width || canvas.height !== imageSize.height) {
        canvas.width = imageSize.width;
        canvas.height = imageSize.height;
        
        // Re-render if we have WebGL context
        if (isWebGL && renderFrame) {
          renderFrame();
        }
      }
    } else {
      console.log(`ImageSize is invalid (${imageSize.width}x${imageSize.height}) - keeping canvas at current size`);
    }
  }, [imageSize.width, imageSize.height, isWebGL]); // Removed renderFrame dependency

  // Handle mouse interactions for zoom/pan
  const handleWheel = useCallback((event) => {
    event.preventDefault();
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Zoom factor
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    
    setViewTransform(prev => {
      const newScale = Math.max(0.1, Math.min(10, prev.scale * zoomFactor));
      
    // Zoom towards cursor position
    const mouseX = (x / rect.width) * 2 - 1;
    const mouseY = 1 - (y / rect.height) * 2;      const newTranslateX = prev.translateX + (mouseX - prev.translateX) * (1 - zoomFactor);
      const newTranslateY = prev.translateY + (mouseY - prev.translateY) * (1 - zoomFactor);
      
      return {
        scale: newScale,
        translateX: newTranslateX,
        translateY: newTranslateY
      };
    });
  }, []);

  const handleDoubleClick = useCallback((event) => {
    const canvas = canvasRef.current;
    if (!canvas || !onDoubleClick) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert screen coordinates to image coordinates
    const { scale, translateX, translateY } = viewTransform;
    
    // Normalize to [-1, 1] range
    const normX = (x / rect.width) * 2 - 1;
    const normY = 1 - (y / rect.height) * 2;
    
    // Apply inverse transform
    const imageX = (normX - translateX) / scale;
    const imageY = (normY - translateY) / scale;
    
    // Convert to image pixel coordinates
    const pixelX = ((imageX + 1) / 2) * imageSize.width;
    const pixelY = ((1 - imageY) / 2) * imageSize.height;
    
    onDoubleClick(pixelX, pixelY, imageSize.width, imageSize.height);
  }, [viewTransform, imageSize, onDoubleClick]);

  // Handle single click for ROI selection etc.
  const handleClick = useCallback((event) => {
    const canvas = canvasRef.current;
    if (!canvas || !onClick) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert screen coordinates to image coordinates
    const { scale, translateX, translateY } = viewTransform;
    
    // Normalize to [-1, 1] range based on displayed size
    const normX = (x / rect.width) * 2 - 1;
    const normY = 1 - (y / rect.height) * 2;
    
    // Apply inverse transform to get original image coordinates
    const imageX = (normX - translateX) / scale;
    const imageY = (normY - translateY) / scale;
    
    // Convert to image pixel coordinates
    // Note: imageSize contains actual image dimensions (width, height)
    const pixelX = ((imageX + 1) / 2) * imageSize.width;
    const pixelY = ((1 - imageY) / 2) * imageSize.height;
    
    // Provide display info for overlays
    const displayInfo = {
      displayWidth: rect.width,
      displayHeight: rect.height,
      scale: scale,
      translateX: translateX,
      translateY: translateY
    };
    
    console.log('LiveViewerGL click:', {
      screenClick: [x, y],
      normalized: [normX, normY],
      imageCoords: [imageX, imageY],
      pixelCoords: [pixelX, pixelY],
      imageSize: [imageSize.width, imageSize.height],
      displayInfo
    });
    
    onClick(pixelX, pixelY, imageSize.width, imageSize.height, displayInfo);
  }, [viewTransform, imageSize, onClick]);

  const resetView = useCallback(() => {
    setViewTransform({ scale: 1, translateX: 0, translateY: 0 });
  }, []);

  // Add global mouse event listeners for dragging
  useEffect(() => {
    const handleGlobalMouseMove = (e) => {
      if (!isDragging) return;
      
      const canvas = canvasRef.current;
      if (!canvas) return;

      const deltaX = e.clientX - lastMousePos.x;
      const deltaY = e.clientY - lastMousePos.y;
      
      // Convert pixel movement to normalized coordinates
      const rect = canvas.getBoundingClientRect();
      const normalizedDeltaX = (deltaX / rect.width) * 2.0;
      const normalizedDeltaY = -(deltaY / rect.height) * 2.0; // Invert Y
      
      setViewTransform(prev => ({
        ...prev,
        translateX: prev.translateX + normalizedDeltaX / prev.scale,
        translateY: prev.translateY + normalizedDeltaY / prev.scale
      }));
      
      setLastMousePos({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    };

    const handleGlobalMouseUp = (e) => {
      if (isDragging) {
        setIsDragging(false);
        e.preventDefault();
      }
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleGlobalMouseMove);
      document.addEventListener('mouseup', handleGlobalMouseUp);
      // Prevent context menu during drag
      document.addEventListener('contextmenu', handleGlobalMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleGlobalMouseMove);
      document.removeEventListener('mouseup', handleGlobalMouseUp);
      document.removeEventListener('contextmenu', handleGlobalMouseUp);
    };
  }, [isDragging, lastMousePos]);

  // Calculate display dimensions to maximize width while maintaining aspect ratio
  const getDisplayDimensions = useCallback(() => {
    if (!imageSize.width || !imageSize.height || !containerSize.width || !containerSize.height) {
      return { width: 800, height: 600 };
    }

    const imageAspectRatio = imageSize.width / imageSize.height;
    const containerAspectRatio = containerSize.width / containerSize.height;
    
    let displayWidth, displayHeight;
    
    if (imageAspectRatio > containerAspectRatio) {
      // Image is wider than container - fit to width
      displayWidth = containerSize.width;
      displayHeight = containerSize.width / imageAspectRatio;
    } else {
      // Image is taller than container - fit to height
      displayHeight = containerSize.height;
      displayWidth = containerSize.height * imageAspectRatio;
    }

    return { width: displayWidth, height: displayHeight };
  }, [imageSize, containerSize]);

  const displayDimensions = getDisplayDimensions();

  return (
    <Box ref={containerRef} sx={{ 
      position: 'relative', 
      width: '100%', 
      height: '100%',
      backgroundColor: 'black',
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>

      
      <canvas
        id="live-viewer-canvas"
        ref={canvasRef}
        width={imageSize.width || 800}
        height={imageSize.height || 600}
        style={{
          cursor: isDragging ? 'grabbing' : (onClick ? 'crosshair' : 'grab'),
          border: '2px solid red',
          zIndex: 1,
          display: 'block',
          width: displayDimensions.width,
          height: displayDimensions.height,
          maxWidth: '100%',
          maxHeight: '100%'
        }}
        onWheel={handleWheel}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      
      {/* Overlay content (e.g., ROI rectangle) rendered on top of the canvas */}
      {overlayContent && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: displayDimensions.width,
            height: displayDimensions.height,
            pointerEvents: 'none',
            zIndex: 2
          }}
        >
          {overlayContent}
        </Box>
      )}
      
      {/* Reset view button */}
      <Box 
        sx={{ 
          position: 'absolute', 
          bottom: 10, 
          right: 10,
          opacity: 0.8,
          '&:hover': { opacity: 1 },
          cursor: 'pointer',
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          px: 2,
          py: 1,
          borderRadius: 1,
          boxShadow: 2, 
          zIndex: 3,

        }}
        onClick={resetView}
      >
        <Typography variant="body2" fontWeight="bold">Reset View (1:1)</Typography>
      </Box>
    </Box>
  );
};

export default LiveViewerGL;