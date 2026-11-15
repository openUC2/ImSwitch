# Arkitekt hub on the FRAME's Pi: tracking

Branch `feat/arkitekt-hub-mikro`, started 2026-11-15 on FRAME ID10 *berry-muscle*
(`openuc2-berry-muscle-26696`: Pi 5, 8 GB, 128 GB micro SD, Hikrobot MV-CS060 3072 × 2048 RGB).
Brief: WP-423…426. Coordination server: **go.arkitekt.live** (decided). The data stays on the Pi.

## Status

| Step | State |
|---|---|
| konstruktor 0.38.1 on the Pi (aarch64 binary, checked against `SHA256SUMS`) | done |
| Hub `rekuest, mikro` at `/home/pi/arkitekt/hub`, ports 2754 (http) / 2755 (https) | running |
| Hub connected to go.arkitekt.live | done: organization 7, hub id 111 |
| ImSwitch on arkitekt 16 / mikro 14 (pins, `uv.lock`, code, 27 unit tests) | done, this branch |
| Canvas with its pyramid → mikro, placed in a µm stage space | done: datasets 1–4 |
| ImSwitch background upload in the live agent (virtual microscope) | done: dataset 3 |
| View in orkestrator on the MacBook | **blocked**: the Mac reaches none of mikro's addresses (finding 11) |
| Run Tile Scan / Acquire Frame on the FRAME's hardware | open: ImSwitch from this branch runs natively (`.venv` on arkitekt 16) |
| Hub as a pallet deployment in os-rpi | done on os-rpi branch `feat/arkitekt-hub` (local), tested on a throwaway hub; this Pi not switched |
| Delete ImSwitch's copy after the upload | done: `deleteAfterUpload`, off by default |
| WP-424 benchmark | first figures below (synthetic scans, SD card) |

## Figures (SD card; VS Code server and the ImSwitch container running beside it)

Hub, idle: **485–752 MB PSS** (rekuest 223, mikro 222, rustfs 95, Postgres 90, takt 55, Caddy 44,
Redis 23 MB at start). Images: **4.0 GB on disk** (1.1 GB is only the compressed pull). Created in
243 s including the pulls.

| Scan (RGB, 12 bit in uint16) | Raw | ImSwitch zarr | Pyramid | Upload to mikro | In RustFS | Peak RSS |
|---|---|---|---|---|---|---|
| 3 × 3 tiles, 3 × 6144 × 9216 | 0.34 GB | 258 MB | 3.7 s | 21 s | 227 MB | 744 MB |
| 10 × 10 tiles, 3 × 20480 × 30720 | 3.77 GB | 2.88 GB | **128 s** | **259 s** | 2.51 GB | 844 MB |

10 × 10: from the end of the scan to the image in mikro takes 6.5 min. CPU averaged us 32 / sy 6 /
id 24 / **wa 38 %** (the SD card is the limit). SoC at most 45.5 °C, never throttled.

## Findings

1. **Hub and coordination are separate.** A hub is 8 containers: rekuest, rekuest-takt, mikro, Postgres,
   Redis, RustFS, Caddy, and after connecting a reporter. It has no Lok and no accounts. Logins and
   organizations stay on go.arkitekt.live. Each hub is accepted once into an organization, and each app
   is approved once. The approval page is where the hub is chosen (`?hub=111` preselects it). The Pi
   needs internet for logins and token refresh, so ImSwitch queues uploads and resumes them.
2. **Client jump.** arkitekt 5.0.1 → 16.0.1 and mikro 6.2.1 → 14.0.0 (released today). Both are pinned
   as one family and must match the hub (rekuest 19, mikro 15). Redeem tokens are gone: use mandate
   keys. Logins stored by 5.x are not reused, so expect one new approval.
3. **Orkestrator** (macOS arm64 build only, no browser build) reads zarr v3 straight from the hub's
   S3 through the gateway. It picks a pyramid level per brick and refuses a large image without a
   pyramid. It draws **at most 64 layers**, and a scene starts with 8 sources. So a scan must be one
   image, not one dataset per tile: Run Tile Scan now writes one canvas (version 3.0.0 of the actions).
4. **mikro takes pyramid levels only when a dataset is created.** There is no server-side
   downscaling. ImSwitch's levels go up as they are, with nothing recomputed. mikro re-encodes them
   (zarr v3, 256² chunks, ≤ 16 MiB shards), so **every scan is stored twice** on the card
   (2.9 + 2.5 GB above).
5. **ImSwitch's pyramid step re-reads level 0 for every level** (`NgffImage.finalize`): 128 s of the
   6.5 min above. Building each level from the previous one would read ~1.33 × level 0 instead of
   4 ×. Stride decimation (= NEAREST) drops sparse bright fluorescence, so it should offer max/mean.
6. **The experiment canvas is not to scale with overlap.** Tiles are butted together (tile `ix` at
   `ix · tile_w` px), so in µm the canvas is stretched by 1/(1 − overlap): 11 % at 10 % overlap.
   The new Run Tile Scan places tiles by stage position. The image placement was also half a field
   off before (the stage position is the field's centre); fixed.
7. **RGB:** mikro shows three channels as one RGB layer only when they are labelled red/green/blue.
   The uploader renames ImSwitch's `…_rgb_r/g/b`.
8. **Device codes live 5 min** (hub acceptance 10 min). ImSwitch now shows a fresh code for 30 min.
9. **For the pallet:**
   - Only rekuest, rekuest-takt and mikro have a restart policy, and it is `on-failure`; the
     database, Redis, RustFS and the gateway have none. Nothing comes back after a reboot.
   - `cgroup_disable=memory` is on, so `docker stats` shows 0 B. Memory was measured as PSS from
     `/proc` instead. Real per-container figures need `cgroup_enable=memory` and a reboot.
   - The ImSwitch container keeps its Arkitekt login in `/home/pi/.local/state/arkitekt`, which is
     not a volume. Every new container asks for approval again.
   - Addresses are fixed when the hub connects. Reserve the DHCP lease; connect again after a change.
10. **Security:**
    - konstruktor writes `hub_credentials.json` and `hub_config.yaml` (tokens, service keys)
      world-readable. They are set to 600 here; to report upstream.
    - Two shipped setups (`FRAME_widefield_default.json` and its `example_` copy) contained a
      redeem token, now blanked.
    - ImSwitch started outside Docker opens JupyterLab without a token on `0.0.0.0:8888`.
11. **Orkestrator "Service mikro not found"** means that no address of mikro answered from the Mac
    within 5 s. The Pi lets it through:
    - the tailnet allows every port;
    - firewalld accepts Docker's DNAT;
    - the served certificate is the one the hub registered, which orkestrator pins.

    Yet no packet from the Mac reached 2754/2755. To check on the Mac: macOS Local Network
    permission for Orkestrator, and `curl -k https://100.117.194.33:2755/mikro/ht`.
12. **Forklift removes compose projects it does not know** at every apply. `forklift plt plan` on
    this Pi lists `remove hub`, so the next boot removes konstruktor's containers (the data
    stays). So the hub has to be a Forklift deployment (option B below).
13. **Offline:** `konstruktor coord create` makes a local coordination server. It runs lok,
    kontrol, Postgres, Redis, RustFS and Caddy, with accounts of its own. A hub connected to it
    works without internet. With go.arkitekt.live, logins and token refresh need internet; scans
    stay on disk and the upload queue waits.
14. **Minor:**
    - konstruktor's README says port 7080, but the CLI uses 2754/2755.
    - `--dry-run` creates the target folder.
    - `docs/setup-config.schema.json` has drifted from `SetupInfo`; only `ArkitektInfo` is
      regenerated here.

## What changed in ImSwitch (this branch)

- Pins and `uv.lock`: the arkitekt 16 family. `mikro-next` is dropped.
- `ArkitektManager`:
  - mandate key instead of redeem token
  - `allow_insecure_transport=` instead of an environment variable
  - a revoked session asks for a new login once
  - expired codes are renewed
- `ArkitektController`:
  - every image goes into one persistent stage space (`reference="stage"`), placed at its top-left
    corner
  - Run Tile Scan writes one canvas, then builds the pyramid and uploads it once
  - setting `uploadExperiments` (off by default): `sigExperimentFinished` queues the canvases, and
    an `@app.background` task uploads them while idle
  - endpoints `queueArkitektUpload` and `getArkitektUploadQueue`
- `model/io/mikro_upload.py`: the canvas upload (levels from disk, contrast from the coarsest
  level) and the persistent queue.
- `scripts/arkitekt_canvas_bench.py`: synthetic scan canvas → mikro, with timings.
- `deleteAfterUpload`: removes a canvas once mikro holds every level of it (checked on the server).
  Toggles for it and `uploadExperiments` are in the panel.
- An arkitekt older than 16 makes the panel name the version it needs. Before, the login failed
  with a TypeError, as in Bene's `.venv`.
- A bare host means https.

## Pallet (os-rpi branch `feat/arkitekt-hub`, option B)

```sh
forklift plt enable-depl arkitekt-hub
forklift plt enable-depl-feat imswitch arkitekt   # ImSwitch keeps its Arkitekt login
forklift plt stage && sudo reboot
konstruktor hub connect --in /home/pi/arkitekt/arkitekt-hub --server go.arkitekt.live   # once
```

- **Forklift owns the containers:** digest-pinned images and `restart: unless-stopped`.
- **konstruktor writes the per-device files once:** `arkitekt-hub-setup.service` runs
  `hub create --no-start` with the same pinned images, before `forklift-apply`.
- **Credentials:** the containers read them from files (`POSTGRES_*_FILE`,
  `RUSTFS_*_FILE`).
- **Before `hub connect`:** rekuest, mikro and the reporter wait for their configs instead of
  failing.
- **Shared commands:** the compose project is named like the folder, so konstruktor's commands act
  on the same containers.
- **Tested** on a throwaway hub: up in 11.5 s, Postgres healthy, both databases created, buckets
  made, gateway on HTTP and HTTPS. `forklift dev plt check` passes with everything enabled.
- **Not tested:** the boot unit itself, and `hub connect` in this layout.
- **Switching this Pi** to the branch would also remove flim-imager and uc2-flasher, which this
  Pi's pallet has but `edge` has not.
- **Moving this Pi's hub:** see the package README (`mv` to `/home/pi/arkitekt/arkitekt-hub`).

## Open questions

- **Bene:**
  - On the Mac: the Local Network permission for Orkestrator and the `curl` above, then datasets
    1–4.
  - Switch this Pi to the os-rpi branch (with flim-imager and uc2-flasher enabled again) and move
    its hub?
  - Push the two branches?
  - Is LAN and hotspot enough, or should the hub join the organization's mesh?
- **Johannes:**
  - Can mikro take an existing zarr v3 (sharded) array without re-encoding?
  - Can the 5-minute code lifetime be set?
  - How should a scene over a persistent stage space with many scans work (`nchildren` 8)?
  - konstruktor's file permissions and restart policies.

## Reproduce

```bash
konstruktor hub create /home/pi/arkitekt/hub --name frame-hub --identifier openuc2-berry-muscle-26696 \
  --services rekuest,mikro --no-plugin-engine --storage folder --mesh none \
  --host 192.168.178.78 --host 192.168.4.1 --host 100.117.194.33 --no-open --yes
konstruktor hub connect --in /home/pi/arkitekt/hub --server go.arkitekt.live --no-open  # accept in a browser
konstruktor status | check | logs | stop | up --in /home/pi/arkitekt/hub
UV_PROJECT_ENVIRONMENT=~/arkitektio/.venv-imswitch uv sync --frozen --inexact  # branch client env
python scripts/arkitekt_canvas_bench.py make 10 10        # canvas + pyramid, timed
python scripts/arkitekt_canvas_bench.py upload <canvas.ome.zarr> --hub 111
```

Arkitekt sources read for this (shallow clones in `~/arkitektio`): konstruktor, mikro, mikro-server,
arkitekt, arkitekt-runtime, arkitekt-spec, rekuest, orkestrator, kontrol, lok-server, arkitekt-gateway,
arkitekt-service, arkitekt-datalayer, ome-converter, ome-tiff-converter, init-container, the docs.
