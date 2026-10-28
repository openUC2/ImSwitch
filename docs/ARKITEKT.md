# Arkitekt

An [Arkitekt](https://arkitekt.live) server lets notebooks, other apps and workflows call the
microscope. It also stores the images in its mikro service and shows the live state. The server can
be the public one (`https://go.arkitekt.live`) or a local deployment, e.g. on a NAS.

ImSwitch uses the `arkitekt` client (5.x, formerly `arkitekt_next`):

```bash
pip install "arkitekt[rekuest,mikro]"
```

Without the package ImSwitch starts normally. The Arkitekt panel then says what to install.

## Setup

The `"Arkitekt"` widget in `availableWidgets` loads it. The `"arkitekt"` block is optional; the panel
writes it when a setting changes:

```json
"arkitekt": {
  "enabled": true,
  "appName": "imswitch",
  "url": "http://nas.local",
  "autoConnect": true,
  "allowInsecureTransport": false,
  "useMikro": true,
  "redeemToken": ""
}
```

| Key | Meaning |
|---|---|
| `url` | The server. Set from the panel when binding. |
| `appName` | The app identifier (lower-cased, spaces become dashes). Each name is its own registration and login. |
| `autoConnect` | At startup, reconnect when the microscope was bound before. It never starts a browser login by itself. If the stored login was revoked, it stops with an error instead of asking again. |
| `allowInsecureTransport` | Allow the login over plain `http://` to a host that is not localhost, e.g. a NAS without TLS. The session token then travels unencrypted. |
| `useMikro` | Offer the image actions. They need mikro on the server. |
| `redeemToken` | Unattended login, with no browser. It is a credential: keep it out of shared setup files. |

## Binding and unbinding

In the Arkitekt panel (System category; switch it on in the App Manager first), enter the server
and click **Bind microscope**:

1. The server is found (`/.well-known/fakts`).
2. It shows a code and an approval link: the device-code login. Open the link on any computer, sign
   in, and approve the microscope. The code must match.
3. The microscope registers its actions and stays connected. The login is stored on this machine
   (arkitekt's per-user state directory), so later starts reconnect without approval.

- **Disconnect / Cancel** stops a pending login or the connection. It keeps the stored login.
- **Unbind** disconnects and removes the stored login, so binding again needs a new approval. The
  fakts protocol has no revocation endpoint: the server keeps the app's registration until someone
  removes it in the Arkitekt web interface.

The stored login is kept per app name, action-interface version (`ARKITEKT_APP_VERSION`, not the
ImSwitch release) and server URL. Updating ImSwitch therefore does not ask for approval again.

## What Arkitekt can call

Dropdowns list this setup's devices. Positions are in µm, in the ImSwitch user frame.

| Action | Notes |
|---|---|
| Get Stage Position | |
| Move Stage | Relative unless `is_absolute`; waits until it arrives. |
| Go To XY | Absolute X and Y together; Z stays. |
| Home Axis | X or Y only. Z is not homed remotely; use the frame homing in ImSwitch. |
| Move To Sample Loading Position | |
| Set Illumination | On/off and intensity, refused outside the source's range. |
| Set Camera | Exposure (ms) and gain. |
| Acquire Frame | One frame taken after the call, stored in mikro (axes c, y, x, with a pyramid and per-channel contrast). |
| Run Tile Scan | A snake-order grid around a centre (default: here). Each tile streams back as it is stored, and all tiles are registered in one stage space, so the Arkitekt viewer shows them placed by position. Optional autofocus at each tile. Afterwards it returns to the start and restores the illumination. |

- Every action that moves or acquires is refused while an experiment, a recording or a workflow
  runs in ImSwitch.
- They hold one `microscope` lock, so remote calls run one at a time.
- The live state (stage position, light on, exposure, running action) is published once a second.

The panel shows the connection, the offered actions, a log of remote calls (arguments, result,
duration, errors) and thumbnails of the images sent.

## HTTP endpoints (`ArkitektController`)

| Endpoint | |
|---|---|
| `GET getArkitektStatus` | state (`unavailable`, `disabled`, `unbound`, `connecting`, `awaiting_login`, `connected`, `error`), userCode, approveUrl, settings, actions, activity, publishedState |
| `POST bindArkitekt?url=&redeemToken=` | Starts the login in the background and returns at once. |
| `POST cancelArkitekt` | Stops a pending login or the connection. |
| `POST unbindArkitekt` | Disconnects and forgets the stored login. |
| `POST setArkitektSettings?url=&appName=&autoConnect=&allowInsecureTransport=&useMikro=` | Saved to the setup file. |
| `GET getArkitektUploads` / `POST clearArkitektActivity` | |

Socket signals: `sigArkitektStatus`, `sigArkitektActivity`, `sigArkitektUpload`.

## Code

- `imswitch/imcontrol/model/managers/ArkitektManager.py`: the connection. It runs on its own asyncio
  loop in a thread, so a pending login can be cancelled. The device-code hook feeds the panel.
- `imswitch/imcontrol/controller/controllers/ArkitektController.py`: `build_app()` declares the app
  (`arkitekt.App`, `@app.action` / `@app.model` / `@app.state`), and the controller serves the panel
  endpoints.
- `frontend/src/components/ArkitektController.jsx`, `state/slices/ArkitektSlice.js`,
  `backendapi/apiArkitektController.js`.
- Tests: `imswitch/imcontrol/_test/unit/test_arkitekt_controller.py`. With arkitekt installed, these
  also check the declarations, a tile scan and the login against a local fake server.

Untested against a real server: the approved login, the agent, and uploads to mikro.
