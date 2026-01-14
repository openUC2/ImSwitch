import React from "react";
import { useDispatch, useSelector } from "react-redux";
import * as tileViewSlice from "../state/slices/TileStreamSlice.js";

const TileViewComponent = () => {
  // redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const tileStreamState = useSelector(tileViewSlice.getTileStreamState);

  return (
    <div style={{ border: "0px solid white" }}>
      {/* Conditionally render image or canvas based on whether the hardwareState.liveViewImage is available */}
      {tileStreamState?.tileViewImage ? (
        <img
          src={`data:image/jpeg;base64,${tileStreamState.tileViewImage}`}
          alt="Base64 Image"
          style={{ width: "100%", height: "auto" }}
        />
      ) : (
        <p>Loading image...</p>
      )}
    </div>
  );
};

export default TileViewComponent;
