#!/usr/bin/env python3
"""
watcher.py - bridge ImSwitch acquisitions into OMERO.

Watches the ImSwitch data folder and imports every finished acquisition into the
OMERO server running in the companion docker-compose stack, so it shows up in the
OMERO.web viewer at http://<host>:<OMERO_WEB_PORT>.

Design notes (why this is not the textbook watchdog example):
  * ImSwitch writes into nested, dated sub-folders (e.g.
    <data>/ExperimentController/<timestamp>/scan.ome.tif) and produces BOTH single
    files (*.ome.tif, *.tif, *.h5) AND directory-based stores (*.ome.zarr). A
    non-recursive, file-only watcher misses almost everything.
  * .zarr is a *directory* that fills with thousands of chunk files over time, so
    "file created" is the wrong signal. We instead detect when an acquisition unit
    (a file, or a whole .zarr store) has stopped changing for STABLE_SECONDS.
  * Pure stdlib polling: no pip deps, no venv, no inotify-watch exhaustion on big
    zarr trees. Run it with the system python3.

It shells out to `docker exec <server> omero import ...`, so it must run on the
Docker host (the Pi), next to the stack. No code changes to ImSwitch required.

Usage:
    python3 watcher.py            # reads ./.env, then runs forever
    python3 watcher.py --once     # single pass (useful for testing / cron)

Stop with Ctrl-C. Already-imported units are remembered in
.omero_watcher_state.json so restarts don't re-import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / ".omero_watcher_state.json"

# Single files that are directly importable.
FILE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".h5", ".hdf5")
# Directory stores: any folder whose name ends with one of these is ONE import unit.
DIR_SUFFIXES = (".zarr",)  # covers ".ome.zarr" too
# Things that are clearly still being written / not data.
SKIP_SUFFIXES = (".tmp", ".part", ".lock", ".log", ".json", ".txt", ".csv", "~")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader so `python3 watcher.py` sees the same .env compose uses."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        # Real environment wins over the file.
        os.environ.setdefault(key, val)


class Config:
    def __init__(self) -> None:
        self.watch_dir = Path(os.environ.get("IMSWITCH_DATA_DIR", "/home/pi/Datasets"))
        self.container = os.environ.get("OMERO_SERVER_CONTAINER", "omero-server")
        self.root_pass = os.environ.get("OMERO_ROOT_PASS", "omero")
        self.transfer = os.environ.get("OMERO_IMPORT_TRANSFER", "ln_s")
        self.project = os.environ.get("OMERO_IMPORT_PROJECT", "ImSwitch")
        self.stable_seconds = float(os.environ.get("OMERO_IMPORT_STABLE_SECONDS", "20"))
        self.poll_seconds = float(os.environ.get("OMERO_IMPORT_POLL_SECONDS", "10"))
        self.import_zarr = os.environ.get("OMERO_IMPORT_ZARR", "true").lower() in ("1", "true", "yes")
        self.max_retries = int(os.environ.get("OMERO_IMPORT_MAX_RETRIES", "3"))


# --------------------------------------------------------------------------- #
# Persistent state: which units are done / permanently failed.
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            data = json.loads(STATE_FILE.read_text())
            data.setdefault("imported", [])
            data.setdefault("gave_up", [])
            return data
        except Exception as exc:  # noqa: BLE001
            log(f"WARN: could not read state file, starting fresh: {exc}")
    return {"imported": [], "gave_up": []}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


# --------------------------------------------------------------------------- #
# Discovery + stability.
# --------------------------------------------------------------------------- #
def discover_units(watch_dir: Path, done: set[str]) -> list[Path]:
    """Return import units (files and .zarr dirs), skipping anything already handled.

    .zarr directories are treated as single units and are not descended into.
    """
    units: list[Path] = []
    if not watch_dir.is_dir():
        return units

    for dirpath, dirnames, filenames in os.walk(watch_dir):
        # Treat *.zarr folders as units; prune so we don't walk their chunk files.
        keep = []
        for d in dirnames:
            if d.startswith("."):
                continue
            full = Path(dirpath) / d
            if d.lower().endswith(DIR_SUFFIXES):
                if str(full) not in done:
                    units.append(full)
                # Either way, don't descend into a zarr store.
            else:
                keep.append(d)
        dirnames[:] = keep

        for f in filenames:
            if f.startswith(".") or f.lower().endswith(SKIP_SUFFIXES):
                continue
            if not f.lower().endswith(FILE_EXTENSIONS):
                continue
            full = Path(dirpath) / f
            if str(full) not in done:
                units.append(full)
    return units


def signature(unit: Path) -> tuple:
    """A cheap fingerprint that changes while data is still being written.

    File  -> (size, mtime). Directory (zarr) -> (file_count, total_size, max_mtime).
    Returns () if the unit vanished mid-scan.
    """
    try:
        if unit.is_dir():
            count = 0
            total = 0
            newest = 0.0
            for dp, _dn, fn in os.walk(unit):
                for name in fn:
                    try:
                        st = os.stat(os.path.join(dp, name))
                    except OSError:
                        continue
                    count += 1
                    total += st.st_size
                    newest = max(newest, st.st_mtime)
            return (count, total, int(newest))
        st = unit.stat()
        return (st.st_size, int(st.st_mtime))
    except OSError:
        return ()


# --------------------------------------------------------------------------- #
# Import.
# --------------------------------------------------------------------------- #
def import_target(cfg: Config, unit: Path) -> str:
    """Build an OMERO import target so images are grouped instead of orphaned.

    Dataset name = the unit's folder path relative to the data root (slashes ->
    '__', because '/' separates Project from Dataset in the target syntax).
    """
    try:
        rel = unit.parent.relative_to(cfg.watch_dir)
        ds = str(rel).replace(os.sep, "__") if str(rel) not in (".", "") else "unsorted"
    except ValueError:
        ds = "unsorted"
    return f"Project:name:{cfg.project}/Dataset:name:{ds}"


def run_import(cfg: Config, unit: Path, target: str | None) -> tuple[bool, str]:
    cmd = [
        "docker", "exec", cfg.container,
        "omero", "import",
        "-s", "localhost", "-p", "4064",
        "-u", "root", "-w", cfg.root_pass,
        f"--transfer={cfg.transfer}",
    ]
    if target:
        cmd += ["-T", target]
    cmd.append(str(unit))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        return False, "`docker` CLI not found on host"
    except subprocess.TimeoutExpired:
        return False, "import timed out after 3600s"
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "unknown error").strip()


def import_unit(cfg: Config, unit: Path) -> bool:
    if unit.is_dir() and not cfg.import_zarr:
        log(f"SKIP zarr (OMERO_IMPORT_ZARR=false): {unit}")
        return False

    log(f"Importing: {unit}")
    target = import_target(cfg, unit)
    ok, err = run_import(cfg, unit, target)
    if not ok and target:
        # Target syntax can vary across server versions; fall back to an orphaned
        # import so the data is at least visible, then surface the original error.
        log(f"  targeted import failed ({err.splitlines()[-1] if err else 'n/a'}); retrying as orphaned")
        ok, err = run_import(cfg, unit, None)
    if ok:
        log(f"  OK -> OMERO ({'Dataset ' + target.split(':')[-1] if target else 'orphaned'})")
        return True
    log(f"  FAILED: {err}")
    return False


# --------------------------------------------------------------------------- #
# Main loop.
# --------------------------------------------------------------------------- #
def server_ready(cfg: Config) -> bool:
    """True if the omero-server container is up (import would otherwise just error)."""
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cfg.container],
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"
    except Exception:  # noqa: BLE001
        return False


def run_once(cfg: Config, state: dict, pending: dict) -> None:
    done = set(state["imported"]) | set(state["gave_up"])
    failures: dict[str, int] = state.setdefault("failures", {})
    now = time.time()

    for unit in discover_units(cfg.watch_dir, done):
        key = str(unit)
        sig = signature(unit)
        if not sig:
            continue
        prev_sig, since = pending.get(key, (None, now))
        if sig != prev_sig:
            pending[key] = (sig, now)  # changed -> (re)start the quiet timer
            continue
        if now - since < cfg.stable_seconds:
            continue  # stable, but not long enough yet

        # Stable long enough -> import.
        if not server_ready(cfg):
            log(f"omero-server '{cfg.container}' not running yet; will retry: {unit.name}")
            continue
        if import_unit(cfg, unit):
            state["imported"].append(key)
            failures.pop(key, None)
            pending.pop(key, None)
            save_state(state)
        else:
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= cfg.max_retries:
                log(f"giving up on {unit} after {cfg.max_retries} attempts")
                state["gave_up"].append(key)
                pending.pop(key, None)
            save_state(state)


def main() -> int:
    load_dotenv(HERE / ".env")
    cfg = Config()
    state = load_state()
    pending: dict = {}

    log("ImSwitch -> OMERO watcher starting")
    log(f"  watch dir : {cfg.watch_dir}")
    log(f"  container : {cfg.container}   transfer={cfg.transfer}   import_zarr={cfg.import_zarr}")
    log(f"  stable={cfg.stable_seconds:.0f}s  poll={cfg.poll_seconds:.0f}s")
    log(f"  already imported: {len(state['imported'])}, gave up: {len(state['gave_up'])}")
    if not cfg.watch_dir.is_dir():
        log(f"  NOTE: watch dir does not exist yet — will pick it up once it appears")

    once = "--once" in sys.argv[1:]
    try:
        while True:
            try:
                run_once(cfg, state, pending)
            except Exception as exc:  # noqa: BLE001 — never let one bad scan kill the loop
                log(f"ERROR during scan (continuing): {exc}")
            if once:
                break
            time.sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        log("stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
