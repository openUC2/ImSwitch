import React, { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import XYZ from "ol/source/XYZ";
import { createXYZ } from "ol/tilegrid";

const OpenLayersComponent = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  const mapRef = useRef();
  const [mapObj, setMapObj] = useState(null);

  useEffect(() => {
    if (!mapRef.current) return;

    const tileSize = [400, 300]; // this is measured in pixels
    const stageExtent = [0, 0, 512 * 25, 512 * 19]; // [minX, minY, maxX, maxY]

    const tileGrid = createXYZ({
      extent: stageExtent,
      tileSize: tileSize,
      maxResolution: 1, //(stageExtent[2] - stageExtent[0]) / tileSize,
      maxZoom: 5,
      minZoom: 5,
    });

    const urlTemplate = `${hostIP}:${hostPort}/imswitch/api/HistoScanController/get_tile?z={z}&x={x}&y={y}`;

    const tileLayer = new TileLayer({
      source: new XYZ({
        url: urlTemplate,
        tileGrid: tileGrid,
        tilePixelRatio: 1, // Scale tiles for zooming
        tileUrlFunction: (tileCoord) => {
          console.log("Resolutions:", tileGrid.getResolutions());
          const [z, x, y] = tileCoord;
          return `${hostIP}:${hostPort}/imswitch/api/...&z=${z}&x=${x}&y=${y}`;
        },
      }),
    });

    const view = new View({
      center: [256, 256], // center of 0..100000
      zoom: 5, // Start with a zoom level of around one fov
      extent: stageExtent,
      resolutions: tileGrid.getResolutions(), // Ensure consistency
    });

    const olMap = new Map({
      target: mapRef.current,
      layers: [tileLayer],
      view: view,
    });

    setMapObj(olMap);
  }, [hostIP, hostPort]);

  return <div ref={mapRef} style={{ width: "1024px", height: "1024px" }} />;
};

export default OpenLayersComponent;
