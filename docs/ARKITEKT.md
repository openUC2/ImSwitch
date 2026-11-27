# Arkitekt

An [Arkitekt](https://arkitekt.live) server lets notebooks, other apps and workflows call the
microscope. It also stores the images in its mikro service and shows the live state. Logins and
organizations live on a coordination server (the public `https://go.arkitekt.live`); the services
(a *hub*) can run anywhere, also on the microscope's own Pi: see
[arkitekt-hub-pi.md](arkitekt-hub-pi.md).

ImSwitch uses the `arkitekt` client 19 with mikro 15 (the family is pinned in `pyproject.toml`: the
set every stainSTORM app uses, which runs against go.arkitekt.live; a hub of the microscope's own
must speak it too):

```bash
pip install "arkitekt[rekuest,mikro]==19.0.1"
```

Without the package ImSwitch starts normally. The Arkitekt panel then says what to install, and
with an older arkitekt (e.g. 5.x in an old environment) which version it needs.

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
  "uploadExperiments": false,
  "deleteAfterUpload": false,
  "mandateKey": ""
}
```

| Key | Meaning |
|---|---|
| `url` | The server. Set from the panel when binding. A bare host (`go.arkitekt.live`) means https. |
| `appName` | The app identifier (lower-cased, spaces become dashes). Each name is its own registration and login. |
| `autoConnect` | At startup, reconnect when the microscope was bound before. It never starts a browser login by itself. If the stored login was revoked, it stops with an error instead of asking again. |
| `allowInsecureTransport` | Allow the login over plain `http://` to a host that is not localhost, e.g. a NAS without TLS. The session token then travels unencrypted. |
| `useMikro` | Offer the image actions. They need mikro on the server. |
| `uploadExperiments` | After an experiment, queue its OME-Zarr canvases for mikro. They are uploaded with their pyramid in the background, while bound and idle (no experiment, recording or workflow; CPU < 60 %; 1 GB free). The queue survives restarts. |
| `deleteAfterUpload` | Remove an uploaded canvas from the microscope once mikro holds every level of it (asked of the server first), so a scan is stored once. Off by default. Only OME-Zarr image folders are removed. |
| `mandateKey` | Unattended login, with no browser: a key a member issued on the server for one hub. It is a credential: keep it out of shared setup files. (`redeemToken` is gone with arkitekt 16.) |

## Binding and unbinding

In the Arkitekt panel (System category; switch it on in the App Manager first), enter the server
and click **Bind microscope**:

1. The server is found (`/.well-known/fakts`).
2. It shows a code and an approval link: the device-code login. Open the link on any computer, sign
   in, pick the hub whose services the microscope should use (`?hub=<id>` on the link preselects
   one), and approve. A code lives about five minutes; an unapproved one is replaced by a fresh one
   for half an hour.
3. The microscope registers its actions and stays connected. The login is stored on this machine
   (arkitekt's per-user state directory), so later starts reconnect without approval. In Docker,
   that directory must be on a volume (`/home/pi/.local/state/arkitekt`), or every new container
   asks again: on openUC2 OS, `forklift plt enable-depl-feat imswitch arkitekt` keeps it in
   `/home/pi/arkitekt/imswitch`.

- **Disconnect / Cancel** stops a pending login or the connection. It keeps the stored login.
- **Unbind** disconnects and removes the stored login, so binding again needs a new approval. While
  connected, the session is also revoked on the server.

The stored login is kept per app name, action-interface version (`ARKITEKT_APP_VERSION`, not the
ImSwitch release) and server URL. Updating ImSwitch therefore does not ask for approval again.

## This machine's own hub (openUC2 OS)

With the `arkitekt-hub` deployment of openUC2 OS, a hub runs on the microscope's own computer
(see [arkitekt-hub-pi.md](arkitekt-hub-pi.md)). The panel then starts with a card *This machine's
hub*:

- **Connect hub to go.arkitekt.live** (once): shows a code and an *Accept the hub* link. Someone
  in the organization accepts the hub there; the hub's services restart for a minute or two, and
  the card shows its name, organization, addresses and services.
- **Bind microscope** then preselects this hub on the approval page (`?hub=<id>`), whenever the hub
  is connected to the same server as the panel's url.
- The card says whether the microscope uses this hub, from the stored login (`boundHub`), and
  warns when it is bound to another one.

ImSwitch does not run konstruktor itself. It reads the hub's state from
`/home/pi/arkitekt/status/hub.json` (`IMSWITCH_ARKITEKT_HUB_STATUS`; nothing secret in it) and
starts the host's `arkitekt-hub-connect@<server>.service` and `arkitekt-hub-status.service` over
the system D-Bus. polkit allows user pi exactly those units (`20-arkitekt-hub.rules`). In Docker,
the `arkitekt` feature of the `imswitch` deployment mounts the state folder and the D-Bus socket.
Without a hub on the machine there is no state file, and the card is not shown.

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
| Acquire Frame | One frame taken after the call, stored in mikro (axes c, y, x, with a pyramid, per-channel contrast and names) and placed in the stage space. |
| Run Tile Scan | A snake-order grid around a centre (default: here), written into one OME-Zarr canvas on the microscope (tiles placed by stage position, under `<data>/ArkitektController/`). Nothing goes over the network during the scan. Afterwards the stage returns to the start, the illumination is restored, the pyramid is built and the canvas is uploaded as one image into the stage space. Optional autofocus at each tile. |

Every image lands in one persistent µm space in mikro (reference `stage`, per app name), at its
top-left corner. One image per scan, not per tile: orkestrator draws at most 64 layers and a scene
starts with 8 sources, so a scan of single tiles cannot be viewed.

- Every action that moves or acquires is refused while an experiment, a recording or a workflow
  runs in ImSwitch.
- They hold one `microscope` lock, so remote calls run one at a time.
- The live state (stage position, light on, exposure, running action) is published once a second.

The panel shows the connection, the offered actions, a log of remote calls (arguments, result,
duration, errors) and thumbnails of the images sent.

## Images that go to mikro on their own

Orkestrator only views images; nothing has to be triggered from Arkitekt. While ImSwitch is bound,
the sources below share one upload queue. It uploads while the microscope is idle, survives
restarts and network loss, and applies `deleteAfterUpload`.

| From | How |
|---|---|
| Experiments | `uploadExperiments`: the canvases of every finished run |
| Any controller | `self._commChannel.sigUploadToMikro.emit({"image": frame, "name": "…", "positionUm": {"x": …, "y": …}})` (an array is first written to disk as a canvas with a pyramid), or `{"path": "….ome.zarr"}` |
| HTTP, scripts | `POST queueArkitektUpload?path=` |
| Other programs | their own Arkitekt login and the `mikro` client, as in `scripts/arkitekt_canvas_bench.py upload` |

[newswitch](https://github.com/openUC2/newswitch) saves only to local files. It does not use mikro
yet, and it is on the older `arkitekt-next` packages.

## HTTP endpoints (`ArkitektController`)

| Endpoint | |
|---|---|
| `GET getArkitektStatus` | state (`unavailable`, `disabled`, `unbound`, `connecting`, `awaiting_login`, `connected`, `error`), userCode, approveUrl, settings, actions, activity, publishedState, boundHub `{id, name}`, localHub |
| `POST bindArkitekt?url=&mandateKey=&hub=` | Starts the login in the background and returns at once. `hub` is preselected on the approval page; by default this machine's own hub. |
| `POST cancelArkitekt` | Stops a pending login or the connection. |
| `POST unbindArkitekt` | Disconnects and forgets the stored login. |
| `POST setArkitektSettings?url=&appName=&autoConnect=&allowInsecureTransport=&useMikro=&uploadExperiments=&deleteAfterUpload=` | Saved to the setup file. |
| `POST queueArkitektUpload?path=&name=` | Queues an OME-Zarr canvas (e.g. an older experiment) for the background upload. |
| `GET getArkitektUploadQueue` | `{pending, uploaded, failed}`, with dataset ids, durations, errors and, with `deleteAfterUpload`, what became of the local copy (`localCopy`). |
| `GET getArkitektUploads` / `POST clearArkitektActivity` | |
| `GET getArkitektHub` | This machine's hub: `{present: false}`, or state (`absent`, `not_connected`, `connecting`, `awaiting_approval`, `starting`, `connected`), name, hubId, server, organizations, addresses, services, approveUrl, code, expiresAt, error |
| `POST connectArkitektHub?server=` | Connects this machine's hub (empty server: the url's). Returns at once; the approval link follows in `getArkitektHub`. |
| `POST cancelArkitektHub` / `POST refreshArkitektHub` | Stops a running connect / has the host report the hub again. |

Socket signals: `sigArkitektStatus`, `sigArkitektActivity`, `sigArkitektUpload`.

## Code

- `imswitch/imcontrol/model/managers/ArkitektManager.py`: the connection. It runs on its own asyncio
  loop in a thread, so a pending login can be cancelled. The device-code hook feeds the panel.
- `imswitch/imcontrol/controller/controllers/ArkitektController.py`: `build_app()` declares the app
  (`arkitekt.App`, `@app.action` / `@app.model` / `@app.state`, and the uploader as
  `@app.background`), and the controller serves the panel endpoints.
- `imswitch/imcontrol/model/io/mikro_upload.py`: an NGFF canvas into mikro with the levels it
  already has (nothing is downsampled again), the stage space, the persistent upload queue.
  `ExperimentController` announces finished canvases with `sigExperimentFinished`; any
  controller sends images with `sigUploadToMikro` (both in `CommunicationChannel`).
- `frontend/src/components/ArkitektController.jsx`, `state/slices/ArkitektSlice.js`,
  `backendapi/apiArkitektController.js`.
- Tests: `imswitch/imcontrol/_test/unit/test_arkitekt_controller.py`. With arkitekt installed, these
  also check the declarations, a tile scan and the login against a local fake server.

Tested against a konstruktor hub (rekuest 19, mikro 15) connected to go.arkitekt.live: the
approved login, the agent and the background upload of canvases. Not yet run on the FRAME's
hardware: Acquire Frame and Run Tile Scan.
