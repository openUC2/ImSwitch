"""ArkitektManager: binds this microscope to an Arkitekt server.

It owns the connection only. What the microscope offers is declared by
ArkitektController, which hands in an app factory. Binding uses the device-code
login. bind() starts it on a background thread, and the code and its approval
link appear in status(). Someone approves it in a browser and picks the hub
(the services, e.g. one on this machine), then the agent starts providing. A
mandate key, issued for one hub on the server, logs in without a browser.
unbind() stops the agent and forgets the stored login on this machine.

The connection runs in its own thread, on its own asyncio loop. Sync actions
still reach it through koil (rekuest runs them via koil.run_threaded), and a
pending login or a running agent can be cancelled from any thread.

The arkitekt package (pip install "arkitekt[rekuest,mikro]") is imported
lazily. Without it, ImSwitch starts normally and status() names what is
missing.
"""
import asyncio
import datetime
import hashlib
import importlib.util
import json
import os
import re
import threading
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.SetupInfo import ArkitektInfo

REQUIRED_PACKAGES = ("arkitekt", "rekuest")
# The client this code is written for (pinned with its family in pyproject.toml).
# An older one fails at the login instead: its device-code hook takes other arguments.
ARKITEKT_VERSION = "19.0.1"
DEFAULT_URL = "https://go.arkitekt.live"
# A device code lives about five minutes; a manual bind asks for a fresh one this
# many times (about half an hour) before it gives up.
LOGIN_CODE_RENEWALS = 6
# Retry an automatic reconnect (stored login, no prompt) after this long, e.g.
# when the server on the NAS is not up yet at boot.
RECONNECT_DELAY_S = 30.0
# arkitekt.constants: the platformdirs identity fakts caches sessions under
ARKITEKT_DIRS = ("arkitekt", "arkitekt.live")

UNAVAILABLE = "unavailable"        # the arkitekt package is not installed
DISABLED = "disabled"              # "enabled": false in the setup
UNBOUND = "unbound"                # not connected
CONNECTING = "connecting"          # discovery, stored login, registering the agent
AWAITING_LOGIN = "awaiting_login"  # device code shown, waiting for approval in a browser
CONNECTED = "connected"            # the agent provides the microscope's actions
ERROR = "error"

AppFactory = Callable[[str, str], Any]
"""(identifier, version) -> arkitekt.App"""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _outdated_arkitekt() -> Optional[str]:
    """The installed arkitekt's version when its major is older than ARKITEKT_VERSION's."""
    try:
        from importlib.metadata import version
        installed = version("arkitekt")
    except Exception:  # noqa: BLE001 - no metadata: let the import decide
        return None
    have, need = (re.match(r"\d+", v) for v in (installed, ARKITEKT_VERSION))
    return installed if have and need and int(have.group()) < int(need.group()) else None


def _slug(name: str) -> str:
    """An app identifier from the configured name ("FRAME Fork" -> "frame-fork")."""
    return re.sub(r"[^a-z0-9._-]+", "-", (name or "").strip().lower()).strip("-") or "imswitch"


def _is_insecure(url: str) -> bool:
    """Plain http to a host that is not this machine: fakts needs an opt-in."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1")


class StoredLoginInvalid(RuntimeError):
    """An automatic reconnect hit the device-code login: the stored session
    expired or was revoked. Asking again unattended would only pile up codes."""


def _caused_by(e: BaseException, kind: str) -> bool:
    """Whether *e*, or what it was raised from, is a *kind* (a class name)."""
    seen = set()
    while e is not None and id(e) not in seen:
        if type(e).__name__ == kind:
            return True
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return False


def _is_reauth(e: BaseException) -> bool:
    """fakts' NeedsReauthenticationError: the stored session or the mandate key
    was revoked or refused, and no code is asked for by itself."""
    return _caused_by(e, "NeedsReauthenticationError")


def cache_file(identifier: str, version: str, url: str) -> str:
    """Where fakts keeps an app's session (a rotating refresh token): one file
    per app identifier, version and server, as arkitekt.app.sessions.session_path
    names it (the url without a trailing slash). Written out so that status()
    need not import arkitekt."""
    from platformdirs import user_state_dir
    key = hashlib.sha256(url.strip().rstrip("/").encode()).hexdigest()[:6]
    return os.path.join(user_state_dir(*ARKITEKT_DIRS), "cache",
                        f"{identifier}-{version}-{key}_fakts_cache.json")


def _friendly_error(e: BaseException, url: str) -> str:
    """One sentence for the panel; the full exception goes to the log."""
    kind = type(e).__name__
    text = str(e) or kind
    if kind == "InsecureTransportError":
        return (f"{url} is plain http. Enable 'Allow insecure (http) login' for a server "
                f"without TLS, or use https.")
    if kind in ("DeviceCodeExpiredError", "DeviceCodeTimeoutError"):
        return "The login code expired before it was approved. Bind again for a new code."
    if isinstance(e, StoredLoginInvalid):
        return text
    if _is_reauth(e):
        return ("The stored login or the mandate key was revoked or has expired on the server. "
                "Bind again to approve this microscope.")
    if kind == "UserDeniedError":
        return "The login was declined in the browser."
    if "discover" in text.lower() or kind in ("ClientConnectorError", "DiscoveryError"):
        return f"Could not reach an Arkitekt server at {url}: {text}"
    return f"{kind}: {text}"


class ArkitektManager:
    """The connection of this microscope to an Arkitekt server."""

    def __init__(self, setupInfo: Optional[ArkitektInfo]):
        self.__logger = initLogger(self)
        self.info = setupInfo if setupInfo is not None else ArkitektInfo()
        self._missing = [p for p in REQUIRED_PACKAGES if importlib.util.find_spec(p) is None]
        self._outdated = None if self._missing else _outdated_arkitekt()
        self._factory: Optional[AppFactory] = None
        self._version = "0.0.1"
        self._listeners: List[Callable[[dict], None]] = []
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_requested = False
        self._forget = False
        self._automatic = False
        self._hub_hint: Optional[str] = None
        self._retry: Optional[threading.Timer] = None
        self._services: List[str] = []
        self._session: Dict[str, Any] = {}
        initial = UNAVAILABLE if not self.is_available() else \
            DISABLED if not self.info.enabled else UNBOUND
        self._set(state=initial, message=self._missing_message())
        if not self.is_available():
            self.__logger.warning(self._missing_message())

    # ── for the controller ──────────────────────────────────────────────────

    def set_app_factory(self, factory: AppFactory, version: str) -> None:
        """What to provide: factory(identifier, version) -> arkitekt.App.
        *version* is the action interface's version. The stored login is kept
        per identifier, version and server, so it must not follow ImSwitch's
        release number."""
        self._factory, self._version = factory, version

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """Called with status() on every change, from any thread."""
        self._listeners.append(callback)

    def is_available(self) -> bool:
        return not self._missing and not self._outdated

    @property
    def identifier(self) -> str:
        return _slug(self.info.appName)

    @property
    def url(self) -> str:
        """The server a bind goes to: the setup's url, $FAKTS_URL, the public one.
        A bare host ("go.arkitekt.live") means https: fakts would otherwise try
        protocols in turn, and the stored login is kept per url string."""
        url = (self.info.url or os.getenv("FAKTS_URL") or DEFAULT_URL).strip().rstrip("/")
        return url if "://" in url else f"https://{url}"

    def status(self) -> dict:
        with self._lock:
            session = dict(self._session)
        url = self.url
        return {**session,
                "available": self.is_available(), "missingPackages": list(self._missing),
                "enabled": bool(self.info.enabled), "url": url,
                "appName": self.identifier, "appVersion": self._version,
                "autoConnect": bool(self.info.autoConnect),
                "allowInsecureTransport": bool(self.info.allowInsecureTransport),
                "useMikro": bool(self.info.useMikro),
                "uploadExperiments": bool(getattr(self.info, "uploadExperiments", False)),
                "deleteAfterUpload": bool(getattr(self.info, "deleteAfterUpload", False)),
                "hasMandateKey": bool(getattr(self.info, "mandateKey", "")),
                "hasStoredLogin": self.has_stored_login(),
                "boundHub": self.bound_hub(),
                "insecureUrl": _is_insecure(url),
                "services": list(self._services)}

    def has_stored_login(self) -> bool:
        path = self._cache_file()
        return bool(path) and os.path.exists(path)

    def bound_hub(self) -> Optional[dict]:
        """The hub chosen at the approval, from the stored login: {id, name}
        (fakts' self.hub and deployment_name; nothing else is read)."""
        path = self._cache_file()
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                own = ((json.load(f).get("fakts") or {}).get("self")) or {}
        except (OSError, ValueError, AttributeError):
            return None
        hub = own.get("hub")
        return {"id": str(hub), "name": own.get("deployment_name")} if hub else None

    # ── binding ─────────────────────────────────────────────────────────────

    def bind(self, url: Optional[str] = None, mandate_key: Optional[str] = None,
             automatic: bool = False, hub: Optional[str] = None) -> dict:
        """Connect in the background: discovery, login, then provide.

        Without a stored login or mandate key this is the device-code login:
        status() turns to "awaiting_login" with userCode and approveUrl for
        someone to approve in a browser. *hub* (a hub id on the server, e.g.
        this machine's own hub) is preselected on the approval page. Returns
        status() at once. *automatic* (startup) retries a failed connect every
        RECONNECT_DELAY_S instead of stopping at the error."""
        refusal = self._refusal()
        if refusal:
            return {**self.status(), "status": "error", "message": refusal}
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {**self.status(), "status": "error",
                        "message": "Already connecting or connected. Cancel or unbind first."}
            if url:
                self.info.url = url.strip()
            self._cancel_retry()
            self._stop_requested = self._forget = False
            self._automatic = automatic
            self._hub_hint = str(hub) if hub and str(hub).isalnum() else None
            self._session = {}
            self._set(state=CONNECTING, message=f"Connecting to {self.url}…",
                      connectingSince=_now())
            key = mandate_key or getattr(self.info, "mandateKey", "") or None
            self._thread = threading.Thread(
                target=self._run, args=(self.url, key, automatic),
                name="ArkitektConnection", daemon=True)
            self._thread.start()
        return {**self.status(), "status": "started"}

    def auto_connect(self) -> None:
        """At startup: reconnect when this microscope was bound before.
        Never starts a browser login by itself."""
        if self._refusal() or not self.info.autoConnect:
            return
        if self.has_stored_login() or getattr(self.info, "mandateKey", ""):
            self.__logger.info(f"Reconnecting to Arkitekt at {self.url} (stored login)")
            self.bind(automatic=True)

    def cancel(self) -> dict:
        """Stop a pending login or disconnect. The stored login is kept."""
        self._stop(forget=False)
        return {**self.status(), "status": "success"}

    def unbind(self) -> dict:
        """Disconnect and forget the stored login on this machine, so the next
        bind asks for approval again. While connected, the session is also
        revoked on the server (fakts 6); otherwise only the local copy goes."""
        self._stop(forget=True)
        removed = self._delete_stored_login()
        state = self._session.get("state") if self._refusal() else UNBOUND
        self._set(state=state, message="Unbound. The stored login on this microscope was "
                                       "removed." if removed else "Unbound.")
        return {**self.status(), "status": "success"}

    def shutdown(self) -> None:
        self._stop(forget=False, timeout=5.0)

    # ── internals ───────────────────────────────────────────────────────────

    def _refusal(self) -> Optional[str]:
        if not self.is_available():
            return self._missing_message()
        if not self.info.enabled:
            return "Arkitekt is disabled in the setup file (arkitekt.enabled = false)."
        if self._factory is None:
            return "The Arkitekt controller is not loaded."
        return None

    def _missing_message(self) -> Optional[str]:
        if self._outdated:
            return (f"arkitekt {self._outdated} is installed, but ImSwitch needs arkitekt "
                    f"{ARKITEKT_VERSION} and its family. Update the environment (uv sync), or: "
                    f"pip install \"arkitekt[rekuest,mikro]=={ARKITEKT_VERSION}\"")
        if not self._missing:
            return None
        return (f"Python package(s) {', '.join(self._missing)} not installed. Install with: "
                f"pip install \"arkitekt[rekuest,mikro]\"")

    def _stop(self, forget: bool, timeout: float = 10.0) -> None:
        with self._lock:
            self._cancel_retry()
            thread, loop, task = self._thread, self._loop, self._task
            self._stop_requested, self._forget = True, forget
        if thread is None or not thread.is_alive():
            if self._session.get("state") in (ERROR, CONNECTING, AWAITING_LOGIN):
                self._set(state=UNBOUND, message=None)
            return
        if loop is not None and task is not None:
            loop.call_soon_threadsafe(task.cancel)
        thread.join(timeout)
        if thread.is_alive():
            self.__logger.warning("Arkitekt connection did not stop in time; left as daemon")

    def _cancel_retry(self) -> None:
        if self._retry is not None:
            self._retry.cancel()
            self._retry = None

    def _run(self, url: str, mandate_key: Optional[str], automatic: bool) -> None:
        try:
            asyncio.run(self._amain(url, mandate_key))
        except asyncio.CancelledError:
            pass
        except BaseException as e:  # noqa: BLE001 - surfaced in the panel
            self.__logger.error(f"Arkitekt connection to {url} failed: {e!r}")
            self._set(state=ERROR, message=_friendly_error(e, url), userCode=None,
                      approveUrl=None)
            retry = not isinstance(e, StoredLoginInvalid) and not _is_reauth(e) and \
                type(e).__name__ not in ("InsecureTransportError", "UserDeniedError")
            if automatic and retry and not self._stop_requested:
                with self._lock:
                    self._retry = threading.Timer(RECONNECT_DELAY_S, self.bind,
                                                  kwargs={"automatic": True})
                    self._retry.daemon = True
                    self._retry.start()
                self._set(message=f"{self._session.get('message')} Retrying in "
                                  f"{RECONNECT_DELAY_S:.0f} s.")
            return
        finally:
            with self._lock:
                self._loop = self._task = None
        if self._stop_requested:
            self._set(state=UNBOUND, message="Disconnected.", boundSince=None,
                      userCode=None, approveUrl=None)

    async def _amain(self, url: str, mandate_key: Optional[str]) -> None:
        with self._lock:
            self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        app = self._factory(self.identifier, self._version)
        self._services = [r.key for r in app.requirements]  # e.g. mikro, s3: what the app needs
        reauth, renewals = False, 0
        while True:
            try:
                await self._provide(app, url, mandate_key, reauth=reauth)
                break
            except Exception as e:  # noqa: BLE001
                # Only a person binding by hand gets a new code: at startup a
                # prompt is StoredLoginInvalid's case instead.
                if self._automatic or mandate_key or self._stop_requested:
                    raise
                if _is_reauth(e) and not reauth:
                    self.__logger.info("Arkitekt: the stored login was revoked; asking anew")
                    reauth = True
                elif _caused_by(e, "DeviceCodeExpiredError") and renewals < LOGIN_CODE_RENEWALS:
                    renewals += 1
                    self.__logger.info("Arkitekt: the login code expired; asking for a fresh one")
                else:
                    raise
        # arun() only returns when the agent stopped on its own
        if not self._stop_requested:
            self._set(state=ERROR, message="The Arkitekt agent stopped.", boundSince=None)

    async def _provide(self, app, url: str, mandate_key: Optional[str], reauth: bool) -> None:
        from arkitekt import connect

        login: Dict[str, Any] = {"mandate_key": mandate_key} if mandate_key else {}
        if reauth:
            login["reauth"] = True
        if self.info.allowInsecureTransport and _is_insecure(url):
            login["allow_insecure_transport"] = True
        runtime = connect(app, provide=True, url=url, headless=True,
                          device_code_hook=self._on_device_code, force=True, **login)
        async with runtime:  # discovery, login, service clients, agent registration
            manifest = runtime.snapshot.manifest if runtime.snapshot else None
            self._set(state=CONNECTED, message=None, userCode=None, approveUrl=None,
                      boundSince=_now(), deviceId=getattr(manifest, "device_id", None))
            self.__logger.info(f"Arkitekt: providing '{self.identifier}' at {url}")
            try:
                await runtime.arun()
            finally:
                if self._forget and runtime.fakts is not None:
                    try:
                        await runtime.fakts.alogout()
                    except Exception as e:  # noqa: BLE001 - the file is removed anyway
                        self.__logger.warning(f"Arkitekt logout failed: {e}")

    async def _on_device_code(self, challenge: Any) -> None:
        """fakts' device-code hook, called with a DeviceCodeChallenge: show the
        code instead of opening a browser on the microscope's own computer.
        Codes live about five minutes on go.arkitekt.live."""
        if self._automatic:
            raise StoredLoginInvalid(
                "The stored login is no longer valid (expired or revoked on the server). "
                "Bind again to approve this microscope.")
        endpoint, code = challenge.endpoint, challenge.user_code
        approve = getattr(challenge, "verification_uri_complete", None)
        if not approve:
            configure = getattr(endpoint, "configure", None)
            approve = configure.replace("{code}", code) if configure else endpoint.base_url
        if urlparse(approve or "").scheme not in ("http", "https"):
            approve = None  # the panel renders it as a link: never javascript: and the like
        elif self._hub_hint:  # preselected on the approval page
            approve += ("&" if "?" in approve else "?") + f"hub={self._hub_hint}"
        self.__logger.info(f"Arkitekt login: approve code {code} at {approve}")
        self._set(state=AWAITING_LOGIN, userCode=code, approveUrl=approve,
                  serverName=getattr(endpoint, "name", None), loginStartedAt=_now(),
                  message="Approve this microscope in a browser to finish binding.")

    def _cache_file(self) -> Optional[str]:
        if self._missing:
            return None
        return cache_file(self.identifier, self._version, self.url)

    def _delete_stored_login(self) -> bool:
        path = self._cache_file()
        if path and os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError as e:
                self.__logger.error(f"Could not remove the stored Arkitekt login {path}: {e}")
        return False

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self._session.update(changes)
            self._session["updatedAt"] = _now()
        status = self.status()
        for callback in list(self._listeners):
            try:
                callback(status)
            except Exception as e:  # noqa: BLE001 - a listener must not break the connection
                self.__logger.warning(f"Arkitekt status listener failed: {e}")
