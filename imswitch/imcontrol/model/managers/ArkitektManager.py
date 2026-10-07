"""ArkitektManager: binds this microscope to an Arkitekt server.

It owns the connection only. What the microscope offers is declared by
ArkitektController, which hands in an app factory. Binding uses the device-code
login. bind() starts it on a background thread, and the code and its approval
link appear in status(). Someone approves it in a browser, then the agent
starts providing. unbind() stops the agent and forgets the stored login on this
machine.

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
import os
import re
import threading
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.SetupInfo import ArkitektInfo

REQUIRED_PACKAGES = ("arkitekt", "rekuest")
DEFAULT_URL = "https://go.arkitekt.live"
# fakts refuses to send credentials over plain http to anything but localhost
# unless this is set (a NAS without TLS).
INSECURE_TRANSPORT_ENV = "FAKTS_ALLOW_INSECURE_TRANSPORT"
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


def cache_file(identifier: str, version: str, url: str) -> str:
    """Where fakts keeps an app's session (a rotating refresh token): one file
    per app identifier, version and server, as arkitekt.app.fakts._cache_path
    names it. Written out so that status() need not import arkitekt."""
    from platformdirs import user_state_dir
    key = hashlib.sha256(url.encode()).hexdigest()[:6]
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
        self._retry: Optional[threading.Timer] = None
        self._services: List[str] = []
        self._session: Dict[str, Any] = {}
        initial = UNAVAILABLE if self._missing else DISABLED if not self.info.enabled else UNBOUND
        self._set(state=initial, message=self._missing_message())
        if self._missing:
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
        return not self._missing

    @property
    def identifier(self) -> str:
        return _slug(self.info.appName)

    @property
    def url(self) -> str:
        """The server a bind goes to: the setup's url, $FAKTS_URL, the public one."""
        return (self.info.url or os.getenv("FAKTS_URL") or DEFAULT_URL).strip()

    def status(self) -> dict:
        with self._lock:
            session = dict(self._session)
        url = self.url
        return {**session,
                "available": not self._missing, "missingPackages": list(self._missing),
                "enabled": bool(self.info.enabled), "url": url,
                "appName": self.identifier, "appVersion": self._version,
                "autoConnect": bool(self.info.autoConnect),
                "allowInsecureTransport": bool(self.info.allowInsecureTransport),
                "useMikro": bool(self.info.useMikro),
                "hasRedeemToken": bool(self.info.redeemToken),
                "hasStoredLogin": self.has_stored_login(),
                "insecureUrl": _is_insecure(url),
                "services": list(self._services)}

    def has_stored_login(self) -> bool:
        path = self._cache_file()
        return bool(path) and os.path.exists(path)

    # ── binding ─────────────────────────────────────────────────────────────

    def bind(self, url: Optional[str] = None, redeem_token: Optional[str] = None,
             automatic: bool = False) -> dict:
        """Connect in the background: discovery, login, then provide.

        Without a stored login or redeem token this is the device-code login:
        status() turns to "awaiting_login" with userCode and approveUrl for
        someone to approve in a browser. Returns status() at once. *automatic*
        (startup) retries a failed connect every RECONNECT_DELAY_S instead of
        stopping at the error."""
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
            if self.info.allowInsecureTransport and _is_insecure(self.url):
                os.environ[INSECURE_TRANSPORT_ENV] = "1"
            self._session = {}
            self._set(state=CONNECTING, message=f"Connecting to {self.url}…",
                      connectingSince=_now())
            self._thread = threading.Thread(
                target=self._run, args=(self.url, redeem_token or self.info.redeemToken or None,
                                        automatic),
                name="ArkitektConnection", daemon=True)
            self._thread.start()
        return {**self.status(), "status": "started"}

    def auto_connect(self) -> None:
        """At startup: reconnect when this microscope was bound before.
        Never starts a browser login by itself."""
        if self._refusal() or not self.info.autoConnect:
            return
        if self.has_stored_login() or self.info.redeemToken:
            self.__logger.info(f"Reconnecting to Arkitekt at {self.url} (stored login)")
            self.bind(automatic=True)

    def cancel(self) -> dict:
        """Stop a pending login or disconnect. The stored login is kept."""
        self._stop(forget=False)
        return {**self.status(), "status": "success"}

    def unbind(self) -> dict:
        """Disconnect and forget the stored login on this machine, so the next
        bind asks for approval again. fakts has no revocation endpoint: the
        app's client stays registered on the server until removed there."""
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
        if self._missing:
            return self._missing_message()
        if not self.info.enabled:
            return "Arkitekt is disabled in the setup file (arkitekt.enabled = false)."
        if self._factory is None:
            return "The Arkitekt controller is not loaded."
        return None

    def _missing_message(self) -> Optional[str]:
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

    def _run(self, url: str, redeem_token: Optional[str], automatic: bool) -> None:
        try:
            asyncio.run(self._amain(url, redeem_token))
        except asyncio.CancelledError:
            pass
        except BaseException as e:  # noqa: BLE001 - surfaced in the panel
            self.__logger.error(f"Arkitekt connection to {url} failed: {e!r}")
            self._set(state=ERROR, message=_friendly_error(e, url), userCode=None,
                      approveUrl=None)
            retry = not isinstance(e, StoredLoginInvalid) and type(e).__name__ not in (
                "InsecureTransportError", "UserDeniedError")
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

    async def _amain(self, url: str, redeem_token: Optional[str]) -> None:
        from arkitekt import connect

        with self._lock:
            self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        app = self._factory(self.identifier, self._version)
        self._services = list(app.services)
        runtime = connect(app, provide=True, url=url, redeem_token=redeem_token,
                          headless=True, device_code_hook=self._on_device_code, force=True)
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
        # arun() only returns when the agent stopped on its own
        if not self._stop_requested:
            self._set(state=ERROR, message="The Arkitekt agent stopped.", boundSince=None)

    async def _on_device_code(self, *args: Any) -> None:
        """fakts' device-code hook: show the code instead of opening a browser
        on the microscope's own computer.

        fakts 5.4+ (arkitekt 6) hands over one DeviceCodeChallenge; earlier
        releases called ``hook(endpoint, code)``. Both are accepted, because
        pyproject allows ``arkitekt>=5.0.1`` and a fresh install gets the new one.
        """
        if self._automatic:
            raise StoredLoginInvalid(
                "The stored login is no longer valid (expired or revoked on the server). "
                "Bind again to approve this microscope.")
        if len(args) == 1:
            challenge = args[0]
            endpoint, code = challenge.endpoint, challenge.user_code
            approve = getattr(challenge, "verification_uri_complete", None)
        else:
            endpoint, code = args
            approve = None
        if not approve:
            configure = getattr(endpoint, "configure", None)
            approve = configure.replace("{code}", code) if configure else endpoint.base_url
        if urlparse(approve or "").scheme not in ("http", "https"):
            approve = None  # the panel renders it as a link: never javascript: and the like
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
