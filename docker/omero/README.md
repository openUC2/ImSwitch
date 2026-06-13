# ImSwitch → OMERO image visualization

Spin up an [OMERO](https://www.openmicroscopy.org/omero/) server + web viewer next to
the production ImSwitch container, and auto-import every acquisition ImSwitch writes so
you can browse, zoom and inspect it in a real image database.

This stack is **separate** from the main ImSwitch deployment (`../docker-compose.yml`)
and never touches it. ImSwitch keeps owning the files; OMERO just visualizes them.

```
ImSwitch (docker)  ──writes──►  /home/pi/Datasets  ◄──reads (ro)──  OMERO server (docker)
                                       ▲                                     ▲
                                       │ polls for finished acquisitions     │ omero import
                                       └──────────────  watcher.py  ─────────┘  (docker exec)
```

## What's here

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Postgres + OMERO.server + OMERO.web (ARM64-compatible images) |
| `.env.example` | All configuration — copy to `.env` and edit |
| `watcher.py` | Host-side importer: watches the data folder, imports finished acquisitions |
| `omero-watcher.service` | systemd unit to run the watcher on boot |

## Why these images (ARM64 / Raspberry Pi 5)

The official `openmicroscopy/omero-server` and `openmicroscopy/omero-web` images are
**amd64-only** and will not start on a Pi. We use the community multi-arch fork by
[@manics](https://github.com/manics), which publishes `linux/amd64` **and** `linux/arm64`:

- `ghcr.io/manics/omero-server-docker:ubuntu`
- `ghcr.io/manics/omero-web-docker:ubuntu`

Both are pinned via `.env` so you can swap to a versioned tag (e.g.
`...:5.6.9-0-ubuntu2204`) once you settle on a version. The Postgres image
(`postgres:15`) is already multi-arch.

> The fork is a few years old but tracks upstream OMERO and works for a prototype.
> If you later want to harden this, forking the repo and bumping its base image is the
> natural next step — but it's not required to get running today.

## Quick start

```bash
cd docker/omero
cp .env.example .env
nano .env              # at minimum: set IMSWITCH_DATA_DIR and change OMERO_ROOT_PASS

# 1) Bring up the OMERO stack (first boot initializes the DB schema; give it 2–3 min)
docker compose up -d
docker compose logs -f server    # wait until you see the server is accepting connections

# 2) See what it WILL import (instant, no server needed) — great first sanity check:
python3 watcher.py --list

# 3) Start the importer (foreground, easy to watch). Pure stdlib — no venv needed.
python3 watcher.py
```

`--list` prints every importable unit it found (plus an extension histogram of your
data tree, so if it finds nothing you can see what's actually on disk) and exits. Each
poll of the running watcher also logs a one-line summary, e.g.
`scan: 247 unit(s) | 247 ready, 0 settling | 0 imported, 0 gave up`, so you can always
tell it's alive.

Then open `http://<pi-ip>:4080`, log in as **root / `<OMERO_ROOT_PASS>`**, and your
acquisitions appear under the **ImSwitch** project (one dataset per acquisition folder).

To test without running an acquisition, drop a TIFF into the data folder:

```bash
cp some-image.ome.tif "$IMSWITCH_DATA_DIR/manual-test/"
# watcher.py logs: Importing ... -> OK
```

## Run the importer on boot (systemd)

```bash
sudo cp omero-watcher.service /etc/systemd/system/
# edit User= / WorkingDirectory= / ExecStart= paths inside the unit if your checkout
# is not at /home/pi/ImSwitch
sudo systemctl daemon-reload
sudo systemctl enable --now omero-watcher.service
journalctl -u omero-watcher -f
```

## How the importer decides a file is "done"

ImSwitch writes into nested, dated sub-folders and produces both single files
(`*.ome.tif`, `*.tif`, `*.h5`) and directory stores (`*.ome.zarr`, which fill with
chunk files over time). `watcher.py` therefore:

- scans the data tree **recursively**;
- treats each `*.zarr` folder as **one** import unit (it does not import individual
  chunks);
- considers a unit ready once its data has **not changed for
  `OMERO_IMPORT_STABLE_SECONDS`** (default 20s), measured from the file/store's own
  mtime — so a freshly-written acquisition waits out the quiet period, but a
  **pre-existing backlog imports on the very first scan** instead of waiting;
- in a folder that has a "primary" output (`*.ome.tif`/`*.ome.tiff`/`*.h5` or a
  `*.zarr` store), skips the loose per-tile `*.tif` files so OMERO isn't flooded
  (`OMERO_IMPORT_SKIP_TILE_TIFFS=false` to import every tile);
- remembers what it imported in `.omero_watcher_state.json`, so restarts don't
  re-import and a crash mid-run is safe.

> First run imports your **entire existing backlog** (every old acquisition is "older
> than 20s"), which can be a lot of `omero import` calls back-to-back. That's intended
> — it's how the months of data already in `/home/pi/Datasets` get into OMERO. Delete
> `.omero_watcher_state.json` to force a re-import later.

## Transfer mode: `ln_s` vs `cp`

Default is `OMERO_IMPORT_TRANSFER=ln_s` (in-place import): OMERO stores a **symlink**
to the original ImSwitch file instead of copying it. This is ideal on a
storage-constrained Pi — no data duplication — and is why the data folder is mounted
into the server at the **same path** the host uses (the symlink must resolve inside the
container). The tradeoff: **the original files must stay where ImSwitch wrote them**; if
you delete/move them, OMERO loses the pixels. Set `OMERO_IMPORT_TRANSFER=cp` to instead
copy bytes into OMERO's managed repository (fully detached, but double the storage).

## Caveats & troubleshooting

- **OME-TIFF is the reliable path.** OME-Zarr/NGFF import is best-effort and depends on
  the Bio-Formats version inside the server image (`OMERO_IMPORT_ZARR=true` by default;
  set `false` to skip). If zarr import fails repeatedly the watcher gives up after
  `OMERO_IMPORT_MAX_RETRIES` and logs it, rather than retrying forever. For dependable
  NGFF support you'd add the `omero-cli-zarr` plugin to a forked server image — out of
  scope for this prototype.
- **First boot is slow.** OMERO initializes the Postgres schema on first run (2–3 min
  on a Pi). Imports before the server is ready are detected and retried automatically.
- **Permissions.** `ln_s` import needs the server process to read the original files.
  ImSwitch's `volume-setup` chowns the data to `1000:1000`, which usually matches the
  server's `omero` user; if imports fail with permission errors, that's the first thing
  to check.
- **The admin password is passed on the import command line** (visible in the
  container's process list). Fine for a private lab Pi; for anything shared, switch to a
  session key (`omero login` → `omero import -k <session>`).
- **This does not use ImSwitch's built-in OMERO config.** `ExperimentController` already
  stores OMERO connection settings, but currently has **no upload code** — it's a stub.
  This external watcher is intentionally decoupled: it needs zero ImSwitch changes and
  works for every output format. If you later wire up in-app uploads, point that config
  at this same server (`<pi-ip>:4064`, user `root`) and disable the watcher.

## Stopping / resetting

```bash
docker compose down            # stop, keep data + imported state
docker compose down -v         # ALSO wipe the OMERO database + managed repo (start over)
rm .omero_watcher_state.json   # make the watcher re-import everything on next run
```
