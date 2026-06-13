#!/usr/bin/env python3
"""
watcher.py - bridge ImSwitch acquisitions into OMERO.

Watches the ImSwitch data folder and imports every finished acquisition into the
OMERO server running in the companion docker-compose stack, so it shows up in the
OMERO.web viewer at http://<host>:<OMERO_WEB_PORT>.

Design notes (why this is not the textbook watchdog example):
  * ImSwitch writes into nested, dated sub-folders (e.g.
    <data>/ExperimentController/<timestamp>/scan.ome.tif) and produces BOTH single
    files (*.ome.tif/*.ome.tiff, *.tif, *.h5) AND directory stores (*.ome.zarr). A
    non-recursive, file-only watcher misses almost everything.
  * .zarr is a *directory* that fills with chunk files over time, so "file created"
    is the wrong signal. A unit (a file, or a whole .zarr store) is considered ready
    once nothing inside it has changed for STABLE_SECONDS -- measured from the data's
    own mtime, so a pre-existing backlog imports immediately.
  * Pure stdlib polling: no pip deps, no venv, no inotify-watch exhaustion on big
    zarr trees. Run it with the system python3.

It shells out to `docker exec <server> omero import ...`, so it must run on the
Docker host (the Pi), next to the stack. No code changes to ImSwitch required.

Usage:
    python3 watcher.py            # read ./.env, then run forever
    python3 watcher.py --once     # single pass (testing / cron)
    python3 watcher.py --list     # show what WOULD be imported, plus an extension
                                  # histogram of the data tree, then exit (no import)

Stop with Ctrl-C. Already-imported units are remembered in
.omero_watcher_state.json so restarts don't re-import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / ".omero_watcher_state.json"

# Single files that are directly importable.
FILE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".h5", ".hdf5")
# A "primary" output marks a finished acquisition; when present we skip the loose
# per-tile *.tif files written alongside it (otherwise OMERO floods with tiles).
PRIMARY_FILE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".h5", ".hdf5")
# Directory stores: any folder whose name ends with one of these is ONE import unit.
DIR_SUFFIXES = (".zarr",)  # covers ".ome.zarr" too
# Things that are clearly not importable image data.
SKIP_SUFFIXES = (".tmp", ".part", ".lock", ".log", ".json", ".txt", ".csv", ".zip",
                 ".png", ".jpg", ".jpeg", ".yaml", ".yml", ".xml", ".npy", "~")


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
        os.environ.setdefault(key, val)  # real environment wins over the file


class Config:
    def __init__(self) -> None:
        self.watch_dir = Path(os.environ.get("IMSWITCH_DATA_DIR", "/home/pi/Datasets"))
        self.container = os.environ.get("OMERO_SERVER_CONTAINER", "omero-server")
        self.root_pass = os.environ.get("OMERO_ROOT_PASS", "omero")
        self.transfer = os.environ.get("OMERO_IMPORT_TRANSFER", "ln_s")
        self.project = os.environ.get("OMERO_IMPORT_PROJECT", "ImSwitch")
        self.stable_seconds = float(os.environ.get("OMERO_IMPORT_STABLE_SECONDS", "20"))
        self.poll_seconds = float(os.environ.get("OMERO_IMPORT_POLL_SECONDS", "10"))
        self.import_zarr = _truthy(os.environ.get("OMERO_IMPORT_ZARR", "true"))
        self.skip_tile_tiffs = _truthy(os.environ.get("OMERO_IMPORT_SKIP_TILE_TIFFS", "true"))
        self.max_retries = int(os.environ.get("OMERO_IMPORT_MAX_RETRIES", "3"))


def _truthy(val: str) -> bool:
    return str(val).lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Persistent state: which units are done / permanently failed.
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            data = json.loads(STATE_FILE.read_text())
            data.setdefault("imported", [])
            data.setdefault("gave_up", [])
            data.setdefault("failures", {})
            return data
        except Exception as exc:  # noqa: BLE001
            log(f"WARN: could not read state file, starting fresh: {exc}")
    return {"imported": [], "gave_up": [], "failures": {}}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


# --------------------------------------------------------------------------- #
# Discovery + stability.
# --------------------------------------------------------------------------- #
def discover_units(watch_dir: Path, done: set[str], skip_tile_tiffs: bool = True) -> list[Path]:
    """Return import units (files and .zarr dirs), skipping anything already handled.

    .zarr directories are treated as single units and are not descended into. When a
    folder contains a "primary" output (*.ome.tif/*.ome.tiff/*.h5 or a *.zarr store),
    the loose per-tile *.tif files in that same folder are skipped (configurable).
    """
    units: list[Path] = []
    if not watch_dir.is_dir():
        return units

    for dirpath, dirnames, filenames in os.walk(watch_dir):
        has_primary = False

        # Treat *.zarr folders as units; prune so we don't walk their chunk files.
        keep = []
        for d in dirnames:
            if d.startswith("."):
                continue
            full = Path(dirpath) / d
            if d.lower().endswith(DIR_SUFFIXES):
                has_primary = True
                if str(full) not in done:
                    units.append(full)
                # don't descend into a zarr store
            else:
                keep.append(d)
        dirnames[:] = keep

        # Collect matching files, noting whether a primary exists in this folder.
        matched: list[Path] = []
        for f in filenames:
            fl = f.lower()
            if f.startswith(".") or fl.endswith(SKIP_SUFFIXES):
                continue
            if not fl.endswith(FILE_EXTENSIONS):
                continue
            if fl.endswith(PRIMARY_FILE_EXTENSIONS):
                has_primary = True
            matched.append(Path(dirpath) / f)

        for full in matched:
            fl = full.name.lower()
            is_plain_tile = (fl.endswith((".tif", ".tiff"))
                             and not fl.endswith((".ome.tif", ".ome.tiff")))
            if skip_tile_tiffs and has_primary and is_plain_tile:
                continue
            if str(full) not in done:
                units.append(full)
    return units


def newest_mtime(unit: Path) -> float:
    """Most recent mtime under a unit (file, or every file in a .zarr dir).

    Used to tell "still being written" from "finished a while ago". 0.0 if it vanished.
    """
    try:
        if unit.is_dir():
            newest = 0.0
            for dp, _dn, fn in os.walk(unit):
                for name in fn:
                    try:
                        m = os.stat(os.path.join(dp, name)).st_mtime
                    except OSError:
                        continue
                    if m > newest:
                        newest = m
            try:  # empty store: fall back to the dir's own mtime
                newest = max(newest, unit.stat().st_mtime)
            except OSError:
                pass
            return newest
        return unit.stat().st_mtime
    except OSError:
        return 0.0


def rel(watch_dir: Path, unit: Path) -> str:
    try:
        return str(unit.relative_to(watch_dir))
    except ValueError:
        return str(unit)


# --------------------------------------------------------------------------- #
# Import.
# --------------------------------------------------------------------------- #
def import_target(cfg: Config, unit: Path) -> str:
    """Group images under a Project/Dataset instead of leaving them orphaned.

    Dataset name = the unit's folder path relative to the data root (slashes -> '__',
    because '/' separates Project from Dataset in the target syntax).
    """
    try:
        rel_parent = unit.parent.relative_to(cfg.watch_dir)
        ds = str(rel_parent).replace(os.sep, "__") if str(rel_parent) not in (".", "") else "unsorted"
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
        log(f"SKIP zarr (OMERO_IMPORT_ZARR=false): {rel(cfg.watch_dir, unit)}")
        return False

    log(f"Importing: {rel(cfg.watch_dir, unit)}")
    target = import_target(cfg, unit)
    ok, err = run_import(cfg, unit, target)
    if not ok and target:
        # Target syntax can vary across server versions; fall back to an orphaned
        # import so data is at least visible, then surface the original error.
        last = err.splitlines()[-1] if err else "n/a"
        log(f"  targeted import failed ({last}); retrying as orphaned")
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


def run_once(cfg: Config, state: dict, seen: dict, announce_new: bool) -> None:
    done = set(state["imported"]) | set(state["gave_up"])
    failures: dict = state["failures"]
    now = time.time()

    units = discover_units(cfg.watch_dir, done, cfg.skip_tile_tiffs)
    ready, settling = [], 0
    for unit in units:
        key = str(unit)
        if key not in seen:
            seen[key] = now
            if announce_new:
                log(f"  detected new: {rel(cfg.watch_dir, unit)}")
        m = newest_mtime(unit)
        if m and (now - m) >= cfg.stable_seconds:
            ready.append(unit)
        else:
            settling += 1

    log(f"scan: {len(units)} unit(s) | {len(ready)} ready, {settling} settling | "
        f"{len(state['imported'])} imported, {len(state['gave_up'])} gave up")

    for unit in ready:
        if not server_ready(cfg):
            log(f"omero-server '{cfg.container}' not running yet; will retry next scan")
            return  # no point trying the rest until the server is up
        key = str(unit)
        if import_unit(cfg, unit):
            state["imported"].append(key)
            failures.pop(key, None)
        else:
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= cfg.max_retries:
                log(f"giving up on {rel(cfg.watch_dir, unit)} after {cfg.max_retries} attempts")
                state["gave_up"].append(key)
        save_state(state)


def cmd_list(cfg: Config) -> int:
    """Diagnostic: show importable units + an extension histogram, then exit."""
    if not cfg.watch_dir.is_dir():
        log(f"watch dir does NOT exist: {cfg.watch_dir}")
        return 1

    units = discover_units(cfg.watch_dir, set(), cfg.skip_tile_tiffs)
    now = time.time()
    print(f"\nWatch dir : {cfg.watch_dir}")
    print(f"Importable units found: {len(units)}\n")
    for u in units[:80]:
        kind = "zarr" if u.is_dir() else "file"
        age = now - newest_mtime(u)
        status = "READY" if age >= cfg.stable_seconds else f"settling ({age:.0f}s < {cfg.stable_seconds:.0f}s)"
        print(f"  [{kind}] {rel(cfg.watch_dir, u)}   {status}")
    if len(units) > 80:
        print(f"  ... and {len(units) - 80} more")

    # Histogram of every file extension in the tree (zarr stores not descended into),
    # so if 0 units are found you can see what IS on disk.
    ext: Counter = Counter()
    for dp, dn, fn in os.walk(cfg.watch_dir):
        dn[:] = [d for d in dn if not d.lower().endswith(DIR_SUFFIXES)]
        for f in fn:
            suf = "".join(Path(f).suffixes[-2:]).lower() or "(no extension)"
            ext[suf] += 1
    print("\nFile-extension histogram (top 25):")
    for suf, count in ext.most_common(25):
        mark = "  <- imported" if suf.endswith(FILE_EXTENSIONS) else ""
        print(f"  {count:7d}  {suf}{mark}")
    print("\n(Tip: '.zarr' directories are counted as units above, not in this file histogram.)\n")
    return 0


def main() -> int:
    load_dotenv(HERE / ".env")
    cfg = Config()
    args = sys.argv[1:]

    if "--list" in args or "--diagnose" in args:
        return cmd_list(cfg)

    state = load_state()
    seen: dict = {}

    log("ImSwitch -> OMERO watcher starting")
    log(f"  watch dir : {cfg.watch_dir}")
    log(f"  container : {cfg.container}   transfer={cfg.transfer}   import_zarr={cfg.import_zarr}")
    log(f"  stable={cfg.stable_seconds:.0f}s  poll={cfg.poll_seconds:.0f}s  skip_tile_tiffs={cfg.skip_tile_tiffs}")
    log(f"  already imported: {len(state['imported'])}, gave up: {len(state['gave_up'])}")
    if not cfg.watch_dir.is_dir():
        log("  NOTE: watch dir does not exist yet -- will pick it up once it appears")

    once = "--once" in args
    first = True
    try:
        while True:
            try:
                run_once(cfg, state, seen, announce_new=not first)
            except Exception as exc:  # noqa: BLE001 — never let one bad scan kill the loop
                log(f"ERROR during scan (continuing): {exc}")
            first = False
            if once:
                break
            time.sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        log("stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
