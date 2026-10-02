"""The firmware server: image listing, version.json, verified downloads.

The frame build's firmware-image-server (Caddy file_server browse) serves the
images and version.json at <url>/: GET <url>/ with Accept: application/json
lists the files; version.json = {version, commit, commit_time, run_url,
files: {name: {size, sha256}}}.
"""
import datetime
import hashlib
import logging
import shutil
from pathlib import Path

import requests

from .images import CAN_NODES, legacy_images, recommend_image

# Master images from before the CANopen naming, tried after the CANopen one.
_LEGACY_MASTER_IMAGES = (
    "esp32_UC2_3_CAN_HAT_Master.bin",
    "esp32_UC2_CAN_HAT_Master.bin",
    "UC2_CAN_HAT_Master.bin",
    "CAN_HAT_Master.bin",
    "UC2_3_CAN_HAT_Master_v2.bin",
    "esp32_UC2_3_CAN_HAT_Master_v2.bin",
)


class FirmwareServer:
    def __init__(self, url: str, cache_dir: Path, logger=None):
        self.url = url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger or logging.getLogger(__name__)

    @property
    def _base(self) -> str:
        return (self.url or "").rstrip("/")

    # ── listing / manifest ──────────────────────────────────────────────────

    def listing(self, timeout=10) -> list:
        """Raw file listing [{name, size, mod_time, ...}]; raises on failure."""
        response = requests.get(f"{self._base}/", headers={"Accept": "application/json"},
                                timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or not all(isinstance(i, dict) and "name" in i for i in data):
            raise ValueError("Invalid response format from firmware server")
        return data

    def image_names(self) -> list:
        return [i["name"] for i in self.listing() if i["name"].endswith(".bin")]

    def set_url(self, url: str) -> dict:
        """Switch to *url* after checking it answers with a listing."""
        previous, self.url = self.url, url
        try:
            listing = self.listing(timeout=1)
        except (requests.exceptions.RequestException, ValueError) as e:
            self.url = previous
            return {"status": "error", "server_url": url,
                    "message": f"Failed to connect to firmware server: {e}"}
        self._log.info(f"Firmware server set: {url} ({len(listing)} files)")
        return {"status": "success", "message": f"Firmware server set: {url}", "server_url": url,
                "firmware_files": listing, "count": len(listing)}

    def manifest(self) -> dict:
        """version.json, or {} when the server has none or is unreachable."""
        if not self._base:
            return {}
        try:
            response = requests.get(f"{self._base}/version.json", timeout=5)
            response.raise_for_status()
            manifest = response.json()
            return manifest if isinstance(manifest, dict) else {}
        except (requests.exceptions.RequestException, ValueError) as e:
            self._log.debug(f"No firmware manifest at {self._base}/version.json: {e}")
            return {}

    @staticmethod
    def file_version(manifest: dict, filename: str):
        """Release string of *filename*; None if version.json does not list it."""
        if filename and filename in manifest.get("files", {}):
            return manifest.get("version") or None
        return None

    def files(self) -> dict:
        """Every .bin on the server with its version.json version and sha256."""
        if not self._base:
            return {"status": "error",
                    "message": "Firmware server not set. Use setOTAFirmwareServer first."}
        try:
            listing = self.listing()
        except (requests.exceptions.RequestException, ValueError) as e:
            self._log.error(f"Error fetching firmware files: {e}")
            return {"status": "error", "message": f"Failed to fetch firmware list: {e}",
                    "server_url": self.url}
        manifest = self.manifest()
        entries = manifest.get("files", {})
        files = [{"filename": i["name"], "size": i.get("size", 0),
                  "mod_time": i.get("mod_time", ""),
                  "url": f"{self._base}/{i['name']}",
                  "version": self.file_version(manifest, i["name"]),
                  "sha256": entries.get(i["name"], {}).get("sha256")}
                 for i in listing if i["name"].endswith(".bin")]
        return {"status": "success", "firmware_server": self.url,
                "server_version": manifest.get("version"),
                "server_commit_time": manifest.get("commit_time"), "files": files}

    def images_by_can_id(self, can_ids=None) -> dict:
        """The fixed-role images present on the server, keyed by CAN id."""
        flat = self.files()
        if flat.get("status") != "success":
            return flat
        by_name = {f["filename"]: f for f in flat["files"]}
        firmware = {can_id: {**{k: by_name[image][k] for k in ("url", "size", "mod_time",
                                                                "version", "sha256")},
                             "filename": image, "can_id": can_id}
                    for can_id, image in legacy_images().items()
                    if image in by_name and (can_ids is None or can_id in can_ids)}
        return {"status": "success", "firmware_server": self.url,
                "server_version": flat.get("server_version"),
                "firmware_count": len(firmware), "firmware": firmware}

    # ── which image ─────────────────────────────────────────────────────────

    def can_image(self, can_id: int):
        """Image for a CAN node that does not report its own: a custom
        id_<can_id>_*.bin wins over the fixed-role image. None if neither is
        on the server."""
        names = self.image_names()
        custom = [n for n in names if n.startswith(f"id_{can_id}_")]
        if custom:
            return custom[0]
        image = legacy_images().get(can_id)
        return image if image in names else None

    def recommend(self, identity: dict) -> dict:
        """The server's image for the board *identity* describes
        (images.recommend_image), with that file's listing entry:
        {status, firmware_server, server_version, recommended: {filename,
        merged, source, reason, candidates, file}}."""
        flat = self.files()
        if flat.get("status") != "success":
            return flat
        by_name = {f["filename"]: f for f in flat["files"]}
        recommended = recommend_image(identity, by_name)
        recommended["file"] = by_name.get(recommended["filename"])
        return {"status": "success", "firmware_server": self.url,
                "server_version": flat.get("server_version"), "recommended": recommended}

    def master_image(self):
        """The USB master's image when none was named: the CANopen master,
        older HAT names, id_1_*.bin, then anything with 'hat' and 'master'."""
        names = self.image_names()
        for name in (CAN_NODES[1][2], *_LEGACY_MASTER_IMAGES):
            if name in names:
                return name
        by_id = [n for n in names if n.startswith("id_1_")]
        fuzzy = [n for n in names if "hat" in n.lower() and "master" in n.lower()]
        return (by_id or fuzzy or [None])[0]

    # ── download ────────────────────────────────────────────────────────────

    def verify(self, path: Path, filename: str, manifest: dict = None) -> bool:
        """Size + sha256 against version.json. True when they match or the
        server does not list the file (nothing to check against)."""
        manifest = self.manifest() if manifest is None else manifest
        entry = manifest.get("files", {}).get(filename)
        if not entry:
            self._log.warning(f"{filename} is not in version.json — download not verified")
            return True
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        return len(data) == entry.get("size", len(data)) and digest == entry.get("sha256")

    def download(self, filename: str, manifest: dict = None):
        """Local path of *filename*, verified against version.json; None on
        failure (a mismatching download is deleted). A cached copy whose
        sha256 matches version.json is reused without downloading."""
        if not self._base:
            self._log.error("Firmware server URL is not set.")
            return None
        filename = filename.lstrip("./")
        manifest = self.manifest() if manifest is None else manifest
        local = self.cache_dir / filename
        listed = filename in manifest.get("files", {})
        if local.exists() and listed and self.verify(local, filename, manifest):
            self._log.info(f"Using cached {filename} (matches version.json)")
            return local
        url = f"{self._base}/{filename}"
        try:
            self._log.info(f"Downloading firmware: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
        except Exception as e:
            self._log.error(f"Failed to download {url}: {e}")
            local.unlink(missing_ok=True)
            return None
        if not self.verify(local, filename, manifest):
            self._log.error(f"Download of {filename} does not match version.json "
                            f"(size/sha256) — deleted")
            local.unlink(missing_ok=True)
            return None
        return local

    # ── cache ───────────────────────────────────────────────────────────────

    def clear_cache(self) -> dict:
        try:
            cached = list(self.cache_dir.glob("*.bin"))
            for f in cached:
                f.unlink()
            self._log.info(f"Cleared {len(cached)} files from firmware cache")
            return {"status": "success", "message": f"Cleared {len(cached)} cached firmware files",
                    "cache_directory": str(self.cache_dir)}
        except Exception as e:
            return {"status": "error", "message": f"Failed to clear cache: {e}"}

    def cache_status(self) -> dict:
        try:
            cached = list(self.cache_dir.glob("*.bin"))
            total = sum(f.stat().st_size for f in cached)
            return {"status": "success", "cache_directory": str(self.cache_dir), "exists": True,
                    "cached_files": [{"filename": f.name, "size": f.stat().st_size,
                                      "modified": datetime.datetime.fromtimestamp(
                                          f.stat().st_mtime).isoformat()} for f in cached],
                    "file_count": len(cached), "total_size": total,
                    "total_size_mb": round(total / (1024 * 1024), 2)}
        except Exception as e:
            return {"status": "error", "message": f"Failed to get cache status: {e}"}
