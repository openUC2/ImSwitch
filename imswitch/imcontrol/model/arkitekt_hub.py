"""This machine's own Arkitekt hub, as the Arkitekt panel shows and connects it.

On openUC2 OS the arkitekt-hub deployment runs a hub (rekuest and mikro) on the
microscope's own computer. Connecting it to a coordination server (e.g.
go.arkitekt.live) needs one approval in a browser. The host's
arkitekt-hub-connect does it and writes the hub's state to STATUS_FILE: its
name, organization and addresses, and the approval link while one is pending;
nothing secret.

ImSwitch starts the host's units over the system D-Bus. polkit lets user pi
(ImSwitch, also inside its container) start and stop only
arkitekt-hub-connect@<server>.service and start arkitekt-hub-status.service
(20-arkitekt-hub.rules). The container gets the state folder and the D-Bus
socket from the imswitch deployment's "arkitekt" feature. Elsewhere there is no
state file, and the panel shows no hub.
"""
import datetime
import json
import os
import re
from typing import Any, Callable, Optional
from urllib.parse import urlparse

STATUS_FILE = os.environ.get("IMSWITCH_ARKITEKT_HUB_STATUS", "/home/pi/arkitekt/status/hub.json")
STATUS_UNIT = "arkitekt-hub-status.service"
CONNECT_UNIT = "arkitekt-hub-connect@{}.service"
# What a connect reports while it runs; the unit stops it after 30 minutes.
IN_PROGRESS = ("connecting", "awaiting_approval", "starting")
STALE_AFTER_S = 35 * 60
# A host name and an optional port: also what polkit accepts as the unit's instance.
SERVER = re.compile(r"^[A-Za-z0-9.-]+(:[0-9]{1,5})?$")
# The state file's keys the panel gets (the file holds nothing else, but stay explicit).
FIELDS = ("state", "updatedAt", "identifier", "name", "hubId", "server", "connected",
          "organizations", "addresses", "services", "approveUrl", "code", "expiresAt",
          "message", "error")

SystemdCall = Callable[[str, str, tuple], Any]


def server_of(url: str) -> Optional[str]:
    """The coordination server's host[:port] in *url* ("https://go.arkitekt.live" ->
    "go.arkitekt.live"), or None when it is not a plain host name."""
    url = (url or "").strip()
    host = urlparse(url if "://" in url else f"https://{url}").netloc
    return host if SERVER.match(host or "") else None


def _systemd(method: str, signature: str, body: tuple) -> Any:
    """Call systemd's manager on the system bus (jeepney: pure Python, no dbus library)."""
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection
    from jeepney.wrappers import unwrap_msg
    address = DBusAddress("/org/freedesktop/systemd1", bus_name="org.freedesktop.systemd1",
                          interface="org.freedesktop.systemd1.Manager")
    with open_dbus_connection(bus="SYSTEM") as connection:
        return unwrap_msg(connection.send_and_get_reply(
            new_method_call(address, method, signature, body), timeout=5))


def _explain(e: BaseException, unit: str) -> str:
    """One sentence for the panel."""
    name = getattr(e, "name", "") or ""
    if isinstance(e, (FileNotFoundError, ConnectionRefusedError)):
        return ("ImSwitch cannot reach this machine's D-Bus. On openUC2 OS, enable the imswitch "
                "deployment's feature: forklift plt enable-depl-feat imswitch arkitekt")
    if name.endswith(("NoSuchUnit", "LoadFailed")) or "not found" in str(e).lower():
        return ("This machine has no Arkitekt hub service. On openUC2 OS: forklift plt "
                "enable-depl arkitekt-hub, stage the pallet and reboot.")
    if "AccessDenied" in name or "InteractiveAuthorization" in name:
        return (f"Not allowed to start {unit}: the arkitekt-hub deployment's polkit rule is "
                f"missing (reboot after enabling the deployment).")
    return f"{unit}: {type(e).__name__}: {e}"


def _seconds_since(iso: Optional[str]) -> float:
    try:
        moment = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - moment).total_seconds()


class LocalHub:
    """Reads the hub's state and starts the host's units."""

    def __init__(self, status_file: Optional[str] = None, systemd: Optional[SystemdCall] = None):
        self._file = status_file or STATUS_FILE
        self._systemd = systemd or _systemd

    def status(self) -> dict:
        """{present: False} without a hub on this machine; otherwise its state:
        absent (deployment enabled, hub not created yet) | not_connected |
        connecting | awaiting_approval (approveUrl, code, expiresAt) | starting |
        connected (name, hubId, organizations, addresses, services), with the
        last attempt's error."""
        try:
            with open(self._file) as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"present": False}
        except (OSError, ValueError) as e:
            return {"present": True, "state": "unknown", "error": f"Unreadable {self._file}: {e}"}
        if not isinstance(data, dict):
            return {"present": True, "state": "unknown", "error": f"Unexpected {self._file}"}
        status = {key: data.get(key) for key in FIELDS}
        if status["state"] in IN_PROGRESS and _seconds_since(status["updatedAt"]) > STALE_AFTER_S:
            # The connect ended without writing its result (e.g. the machine restarted).
            status.update(state="connected" if status.get("connected") else "not_connected",
                          approveUrl=None, code=None,
                          error="The last connection attempt ended without a result.")
        return {"present": True, **status}

    def connect(self, server: str) -> dict:
        """Start connecting the hub to *server* (host[:port]); returns at once."""
        if not SERVER.match(server or ""):
            return {"status": "error", "message": f"Not a server name: {server!r}"}
        return self._start(CONNECT_UNIT.format(server))

    def cancel(self) -> dict:
        """Stop a running connect; the hub stays as it was."""
        server = self.status().get("server")
        if not server or not SERVER.match(server):
            return {"status": "error", "message": "No connection attempt to stop."}
        unit = CONNECT_UNIT.format(server)
        try:
            self._systemd("StopUnit", "ss", (unit, "replace"))
        except Exception as e:  # noqa: BLE001 - shown in the panel
            return {"status": "error", "message": _explain(e, unit)}
        return {"status": "success"}

    def refresh(self) -> dict:
        """Have the host write the hub's state again (a few seconds)."""
        return self._start(STATUS_UNIT)

    def _start(self, unit: str) -> dict:
        try:
            self._systemd("StartUnit", "ss", (unit, "replace"))
        except Exception as e:  # noqa: BLE001 - shown in the panel
            return {"status": "error", "message": _explain(e, unit)}
        return {"status": "started"}
