
// filepath: /Users/bene/Downloads/microscope-app/src/axon/ZarrTileView.js
// Example of handling invalid Zarr URL and disabling double-click zoom

import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import DeckGL from "@deck.gl/react";
import { OrthographicView, OrthographicController } from "@deck.gl/core";
import { MultiscaleImageLayer } from "@hms-dbmi/viv";
import * as omeZarrSlice from "../state/slices/OmeZarrTileStreamSlice.js";
import { log } from "deck.gl";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";

const ZarrTileViewController = () => {
  // reference Redux state
  const dispatch = useDispatch();
  const omeZarrState = useSelector(omeZarrSlice.getOmeZarrState);
  const [fullURL, setfullURL] = useState("");

    // Access global Redux state
    const connectionSettingsState = useSelector(
      connectionSettingsSlice.getConnectionSettingsState
    );

  // track dimensions and potential errors
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const [errorMsg, setErrorMsg] = useState("");

  // log debug info
  useEffect(() => {
    console.info("[ZARR-VIEW] new zarrUrl ➜", omeZarrState.zarrUrl);
    // e.g. https://avivator.gehlenborglab.org/?image_url=https://localhost/imswitch/data/ExperimentController/20250603_140818_FastStageScan.ome.zarr
    setfullURL(`${connectionSettingsState.ip}:${connectionSettingsState.apiPort}/imswitch/data${omeZarrState.zarrUrl}`); // Switch to https if needed
  }, [omeZarrState.zarrUrl]);

  useEffect(() => {
    console.info("[ZARR-VIEW] redux version bump ➜", omeZarrState.version);
  }, [omeZarrState.version]);

  useEffect(() => {
    console.info("[ZARR-VIEW] tilePath ➜", omeZarrState.tilePath);
  }, [omeZarrState.tilePath]);

  // Build full URL and verify accessibility
  

  useEffect(() => {
    if (!omeZarrState.zarrUrl) return;

    // try to confirm the URL is valid (simple HEAD request)
    const testUrlValidity = async () => {
      try {
        console.log("[ZARR-VIEW] testing OME-Zarr URL:", fullURL);
        const res = await fetch(`${fullURL}/.zattrs`, { method: "HEAD" });
        if (!res.ok) {
          throw new Error(`Server returned ${res.status}`);
        }
        setErrorMsg(""); // reset errors
      } catch (e) {
        console.error("[ZARR-VIEW] invalid OME-Zarr URL:", e);
        setErrorMsg(`Failed to load OME-Zarr: ${e.toString()}`);
      }
    };
    testUrlValidity();
  }, [fullURL, omeZarrState.zarrUrl]);

  // early return if no URL or if invalid
  if (!omeZarrState.zarrUrl) {
    return <p style={{ color: "gray" }}>Waiting for first tile…</p>;
  }

  if (errorMsg) {
    return <p style={{ color: "red" }}>{errorMsg}</p>;
  }

  // now render DeckGL with double-click zoom disabled
  return (
    <DeckGL
      key={omeZarrState.version}
      views={[new OrthographicView({ id: "ortho" })]}
      controller={{ type: OrthographicController, doubleClickZoom: false }}
      style={{ width: "100%", height: "100%" }}
      initialViewState={{
        target: [dims.w / 2, dims.h / 2, 0],
        zoom: 0
      }}
    >
      <MultiscaleImageLayer
        id="scan"
        loader={{ type: "zarr", url: fullURL }}
        onMetadata={(meta) => {
          // English comment: shape is typically [t, c, z, y, x] but we only need y,x
          const [y, x] = meta.data.shape.slice(-2);
          setDims({ w: x, h: y });
        }}
        pickable={false}
      />
    </DeckGL>
  );
};

export default ZarrTileViewController;
