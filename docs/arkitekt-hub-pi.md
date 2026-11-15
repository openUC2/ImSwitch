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
| View in orkestrator on the MacBook | **open** (Bene) |
| Run Tile Scan / Acquire Frame on the FRAME's hardware | open: needs an ImSwitch image from this branch |
| Hub as a pallet deployment in os-rpi | open: plan below |
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
11. **Minor:**
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

## Pallet plan (os-rpi, off by default)

- `deployments/arkitekt-hub.pkg`, two options:
  - **A, konstruktor owns the hub.** Forklift ships the pinned binary and a unit that runs
    `konstruktor up` at boot. `hub create` and `hub connect` run once by hand, with one browser
    approval. konstruktor's backup, update and rollback keep working.
  - **B, Forklift owns the compose.** Digest-pinned, `restart: unless-stopped`. konstruktor only
    writes the configs once (`hub create --no-start`): the keys are per device, so they cannot live
    in the pallet.

  A is enough for the benchmark; B needs a decision with Ethan. In both cases: a firewalld service
  for 2754/2755 and a link on the landing page.
- `imswitch.pkg`: bind `/home/pi/arkitekt/imswitch` to `/home/pi/.local/state/arkitekt` (as the
  robot apps do), and an image built from this branch.

## Open questions

- **Bene:**
  - Open datasets 1–4 in orkestrator on the MacBook, on the LAN or the hotspot: time to the first
    picture, pan/zoom, and whether the RGB layer looks right.
  - Is LAN and hotspot enough, or should the hub join the organization's mesh?
  - Delete ImSwitch's copy after the upload? That saves half the card.
  - May I enable the memory cgroup (needs a reboot)?
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
