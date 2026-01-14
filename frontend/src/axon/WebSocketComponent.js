import React from "react";
import { useDispatch, useSelector } from "react-redux";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as webSocketSlice from "../state/slices/WebSocketSlice.js";

//##################################################################################
const WebSocketComponent = () => {
  // redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const webSocketState = useSelector(webSocketSlice.getWebSocketState);

  //##################################################################################
  return (
    <div style={{ border: "0px solid white", padding: "10px" }}>
      <table style={{ textAlign: "left" }}>
        <thead>
          <tr>
            <th>Settings</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>IP Address</td>
            <td>{connectionSettingsState.ip}</td>
          </tr>
          <tr>
            <td>Port (ws)</td>
            <td>{connectionSettingsState.websocketPort}</td>
          </tr>
          <tr>
            <td>Port (api)</td>
            <td>{connectionSettingsState.apiPort}</td>
          </tr>
          <tr>
            <th>Status</th>
            <th>
              <div
                style={{
                  width: "16px",
                  height: "16px",
                  borderRadius: "100%",
                  backgroundColor: webSocketState.connected ? "green" : "red",
                }}
              />
            </th>
          </tr>
          <tr>
            <td>Connection</td>
            <td>{webSocketState.connected ? "Connected" : "Disconnected"}</td>
          </tr>
          <tr>
            <td>Signal Count</td>
            <td>{webSocketState.signalCount}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

//##################################################################################
export default WebSocketComponent;
