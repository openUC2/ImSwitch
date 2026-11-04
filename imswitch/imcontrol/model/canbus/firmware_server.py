"""The firmware server: image listing, version.json, verified downloads.

Images come from one of two sources:

- a local firmware server: the frame build's firmware-image-server (Caddy
  file_server browse) serves the images and version.json at <url>/: GET <url>/
  with Accept: application/json lists the files; version.json = {version,
  commit, commit_time, run_url, files: {name: {size, sha256}}}.
- a GitHub release of the firmware repository: the release's assets are the
  listing, and the same version.json is built from the API (version = the tag,
  which tagged builds bake in as UC2_FW_VERSION; sha256 = the asset digest).
"""
import datetime
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote

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


GITHUB_API = "https://api.github.com"
DEFAULT_GITHUB_REPO = "youseetoo/uc2-esp32"
# GitHub allows 60 API requests per hour and IP without a token, and one update
# check reads the release several times: answers are kept this long.
_GITHUB_TTL_S = 300


def _release_images(release: dict) -> dict:
    """{name: asset} of the images in a GitHub release. The web-flasher images
    (<id>.bin next to <id>-manifest.json) are left out: they are merged
    images without the _merged suffix, so they would be flashed at the app
    offset. The same builds are in the release as esp32_<env>[_merged].bin."""
    assets = release.get("assets") or []
    names = {a["name"] for a in assets}
    return {a["name"]: a for a in assets
            if a["name"].endswith(".bin") and f"{a['name'][:-4]}-manifest.json" not in names}


def _asset_sha256(asset: dict):
    digest = asset.get("digest") or ""
    return digest[len("sha256:"):] if digest.startswith("sha256:") else None


class FirmwareServer:
    """:param url: the local firmware server
    :param github_repo: the firmware repository whose releases can be used
    :param release: None = the local server; "latest" or a tag = that GitHub
                    release of *github_repo*"""

    def __init__(self, url: str, cache_dir: Path, logger=None,
                 github_repo: str = DEFAULT_GITHUB_REPO, release: str = None):
        self.url = url
        self.github_repo = github_repo or DEFAULT_GITHUB_REPO
        self.release = release or None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger or logging.getLogger(__name__)
        self._github_cache = {}  # API path -> (monotonic time, JSON)

    @property
    def _base(self) -> str:
        return (self.url or "").rstrip("/")

    @property
    def location(self) -> str:
        """Where the images come from: the local URL, or the release page."""
        if not self.release:
            return self.url
        try:
            return self._github_release()["html_url"]
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return f"https://github.com/{self.github_repo}/releases ({self.release})"

    def source(self) -> dict:
        """The configured source, without asking any server."""
        return {"source": "github" if self.release else "local",
                "firmware_server_url": self.url, "github_repo": self.github_repo,
                "github_release": self.release}

    def file_url(self, filename: str) -> str:
        if not self.release:
            return f"{self._base}/{filename}"
        tag = self._github_release()["tag_name"]
        return f"https://github.com/{self.github_repo}/releases/download/{tag}/{filename}"

    # ── GitHub releases ─────────────────────────────────────────────────────

    def _github(self, path: str, timeout=10):
        """GET <GitHub API>/<path> as JSON, cached for _GITHUB_TTL_S. Uses
        $GITHUB_TOKEN when set (higher rate limit). Raises on failure."""
        hit = self._github_cache.get(path)
        if hit and time.monotonic() - hit[0] < _GITHUB_TTL_S:
            return hit[1]
        headers = {"Accept": "application/vnd.github+json"}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        response = requests.get(f"{GITHUB_API}/{path}", headers=headers, timeout=timeout)
        if response.status_code in (403, 429) and \
                response.headers.get("X-RateLimit-Remaining") == "0":
            raise requests.exceptions.HTTPError(
                "GitHub API rate limit reached (60 requests per hour without GITHUB_TOKEN); "
                "try again later")
        response.raise_for_status()
        data = response.json()
        self._github_cache[path] = (time.monotonic(), data)
        return data

    def github_releases(self, repo: str = None, limit: int = 20) -> list:
        """Published (non-draft) releases of *repo*, newest first; raises on failure."""
        data = self._github(f"repos/{repo or self.github_repo}/releases?per_page={int(limit)}")
        if not isinstance(data, list):
            raise ValueError("Invalid release list from GitHub")
        return [r for r in data if isinstance(r, dict) and not r.get("draft")]

    def _github_release(self) -> dict:
        """The selected release. "latest" is the newest release with images,
        pre-releases included: GitHub's own "latest" skips pre-releases, and
        the CANopen images have only been published in pre-releases so far."""
        if self.release == "latest":
            found = next((r for r in self.github_releases() if _release_images(r)), None)
            if found is None:
                raise ValueError(f"{self.github_repo} has no release with firmware images")
            return found
        tag = quote(self.release, safe="")
        data = self._github(f"repos/{self.github_repo}/releases/tags/{tag}")
        if not isinstance(data, dict) or "tag_name" not in data:
            raise ValueError(f"Invalid release {self.release} from GitHub")
        return data

    def releases(self, repo: str = None, limit: int = 20) -> dict:
        """The releases to choose from: {status, repo, releases: [{tag, name,
        prerelease, published_at, url, image_count}]}; image_count 0 = only
        web-flasher images, nothing ImSwitch can flash."""
        repo = repo or self.github_repo
        try:
            found = self.github_releases(repo, limit)
        except (requests.exceptions.RequestException, ValueError) as e:
            return {"status": "error", "repo": repo,
                    "message": f"Could not list the releases of {repo}: {e}"}
        return {"status": "success", "repo": repo, "releases": [
            {"tag": r.get("tag_name"), "name": r.get("name"), "prerelease": r.get("prerelease"),
             "published_at": r.get("published_at"), "url": r.get("html_url"),
             "image_count": len(_release_images(r))} for r in found]}

    # ── listing / manifest ──────────────────────────────────────────────────

    def listing(self, timeout=10) -> list:
        """Raw file listing [{name, size, mod_time, ...}]; raises on failure."""
        if self.release:
            return [{"name": name, "size": a.get("size", 0), "mod_time": a.get("updated_at", "")}
                    for name, a in _release_images(self._github_release()).items()]
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
        """Switch to the local server *url* after checking it answers with a listing."""
        previous = self.url, self.release
        self.url, self.release = url, None
        try:
            listing = self.listing(timeout=1)
        except (requests.exceptions.RequestException, ValueError) as e:
            self.url, self.release = previous
            return {"status": "error", "server_url": url,
                    "message": f"Failed to connect to firmware server: {e}"}
        self._log.info(f"Firmware server set: {url} ({len(listing)} files)")
        return {"status": "success", "message": f"Firmware server set: {url}", "server_url": url,
                "firmware_files": listing, "count": len(listing)}

    def set_release(self, release: str = "latest", repo: str = None) -> dict:
        """Switch to GitHub release *release* ("latest" or a tag) of *repo*
        after checking it exists and has images ImSwitch can flash."""
        previous = self.release, self.github_repo
        self.release, self.github_repo = release or "latest", repo or self.github_repo
        try:
            tag = self._github_release()["tag_name"]
            listing = self.listing()
            if not listing:
                raise ValueError("it has no esp32_*.bin images, only web-flasher images "
                                 "(flash those with the UC2 Web Flasher)")
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            failed = f"{self.github_repo} {self.release}"
            self.release, self.github_repo = previous
            return {"status": "error", "release": release,
                    "message": f"Cannot use GitHub release {failed}: {e}"}
        self._log.info(f"Firmware source set: GitHub {self.github_repo} {tag} "
                       f"({len(listing)} images)")
        return {"status": "success", "message": f"Firmware from GitHub release {tag}",
                "server_url": self.location, "release": self.release, "release_tag": tag,
                "firmware_files": listing, "count": len(listing)}

    def manifest(self) -> dict:
        """version.json, or {} when the server has none or is unreachable."""
        if self.release:
            try:
                return self._release_manifest()
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                self._log.warning(f"GitHub release {self.github_repo} {self.release} "
                                  f"not readable: {e}")
                return {}
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

    def _release_manifest(self) -> dict:
        """version.json of the selected GitHub release. Assets uploaded before
        GitHub published digests have no sha256: only their size is checked."""
        release = self._github_release()
        return {"version": release["tag_name"], "commit": release.get("target_commitish"),
                "commit_time": release.get("published_at"), "run_url": release.get("html_url"),
                "files": {name: {"size": a.get("size"), "sha256": _asset_sha256(a)}
                          for name, a in _release_images(release).items()}}

    @staticmethod
    def file_version(manifest: dict, filename: str):
        """Release string of *filename*; None if version.json does not list it."""
        if filename and filename in manifest.get("files", {}):
            return manifest.get("version") or None
        return None

    def files(self) -> dict:
        """Every .bin on the server with its version.json version and sha256."""
        if not self.release and not self._base:
            return {"status": "error",
                    "message": "Firmware server not set. Use setOTAFirmwareServer first."}
        try:
            listing = self.listing()
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            self._log.error(f"Error fetching firmware files: {e}")
            return {"status": "error", "message": f"Failed to fetch firmware list: {e}",
                    "server_url": self.location}
        manifest = self.manifest()
        entries = manifest.get("files", {})
        files = [{"filename": i["name"], "size": i.get("size", 0),
                  "mod_time": i.get("mod_time", ""),
                  "url": self.file_url(i["name"]),
                  "version": self.file_version(manifest, i["name"]),
                  "sha256": entries.get(i["name"], {}).get("sha256")}
                 for i in listing if i["name"].endswith(".bin")]
        return {"status": "success", "firmware_server": self.location,
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
        return {"status": "success", "firmware_server": self.location,
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
        return {"status": "success", "firmware_server": self.location,
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
        server does not list the file (nothing to check against); only the
        size when the entry has no sha256."""
        manifest = self.manifest() if manifest is None else manifest
        entry = manifest.get("files", {}).get(filename)
        if not entry:
            self._log.warning(f"{filename} is not in version.json — download not verified")
            return True
        data = path.read_bytes()
        if len(data) != (entry.get("size") or len(data)):
            return False
        if not entry.get("sha256"):
            self._log.warning(f"{filename} has no sha256 on the server — only its size was checked")
            return True
        return hashlib.sha256(data).hexdigest() == entry["sha256"]

    def download(self, filename: str, manifest: dict = None):
        """Local path of *filename*, verified against version.json; None on
        failure (a mismatching download is deleted). A cached copy whose
        sha256 matches version.json is reused without downloading."""
        if not self.release and not self._base:
            self._log.error("Firmware server URL is not set.")
            return None
        filename = filename.lstrip("./")
        manifest = self.manifest() if manifest is None else manifest
        local = self.cache_dir / filename
        has_sha = bool(manifest.get("files", {}).get(filename, {}).get("sha256"))
        if local.exists() and has_sha and self.verify(local, filename, manifest):
            self._log.info(f"Using cached {filename} (matches version.json)")
            return local
        url = filename
        try:
            url = self.file_url(filename)
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
