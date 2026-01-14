// src/components/VtkViewer.js
import React, { useRef, useEffect } from "react";
import vtkFullScreenRenderWindow from "vtk.js/Sources/Rendering/Misc/FullScreenRenderWindow";
import vtkHttpDataSetReader from "vtk.js/Sources/IO/Core/HttpDataSetReader";

function VtkViewer({ tifUrl }) {
  // containerRef is used as the DOM container for the VTK render window
  const containerRef = useRef(null);
  const rendererRef = useRef(null);

  useEffect(() => {
    // Proceed only if the container is available
    if (!containerRef.current) 
      return;

    // Create VTK's FullScreenRenderWindow with the containerRef
    const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
      rootContainer: containerRef.current,
      containerStyle: {
        width: "100%",
        height: "100%",
        position: "relative",
      },
      background: [0.2, 0.2, 0.2], // Example background color
    });

    // Keep reference for cleanup
    rendererRef.current = fullScreenRenderer;

    // Get renderer and render window
    const renderer = fullScreenRenderer.getRenderer();
    const renderWindow = fullScreenRenderer.getRenderWindow();

    // Setup data reader
    const reader = vtkHttpDataSetReader.newInstance({ fetchGzip: true });
    reader
      .setUrl(tifUrl)
      .then(() => reader.loadData())
      .then(() => {
        // The output is expected to be image data
        const imageData = reader.getOutputData();

        // Example: add the image as a volume or surface
        renderer.addVolume(imageData);

        // Reset the camera and render the scene
        renderer.resetCamera();
        renderWindow.render();
      })
      .catch((error) => console.error("Error loading TIF stack:", error));

    // Cleanup function to properly release resources
    return () => {
      if (rendererRef.current) {
        rendererRef.current.delete();
        rendererRef.current = null;
      }
    };
  }, [tifUrl]);

  // Provide a container div for the VTK rendering
  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "500px", border: "1px solid #ccc" }}
    />
  );
}

export default VtkViewer;
