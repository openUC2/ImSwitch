"""
imswitch.plugin_sdk
===================

Stable public API for ImSwitch **v2 plugins**.

This is the *only* ImSwitch module a plugin should import. Everything else
(``imswitch.imcontrol``, ``imswitch.imcommon``, the ``MasterController``,
the setup-file parser, etc.) is host-private and may change between
releases.

The SDK is versioned independently of the host (``sdk_min`` in
``plugin.toml``). The host guarantees backwards compatibility within a
major version: a plugin built against 1.x will keep loading on every 1.y
release of ImSwitch.

Public surface
--------------
- :class:`PluginController`     — base class for backend controllers.
- :class:`PluginContext`        — runtime bag passed to plugins.
- :func:`APIExport`             — decorator: turn a method into an HTTP
                                  endpoint mounted under
                                  ``/plugin/<name>/api/<path>``.
- :class:`Event`                — declarative server-pushed event over the
                                  per-plugin Socket.IO namespace.
- :class:`PluginManifest`       — pydantic schema for ``plugin.toml``.
- :class:`PluginRegistration`   — return type of ``register(ctx)``.
- :func:`load_manifest`         — helper that parses ``plugin.toml`` into
                                  :class:`PluginManifest`.

See ``imswitch-plugin-goniometer`` for the reference implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

try:                                  # python 3.11+
    import tomllib
except ImportError:                   # python 3.10
    import tomli as tomllib           # type: ignore

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator


__all__ = [
    "PluginController",
    "PluginContext",
    "PluginManifest",
    "PluginRegistration",
    "HardwareRequirement",
    "UIManifest",
    "Permissions",
    "APIExport",
    "Event",
    "load_manifest",
    "SDK_VERSION",
]

SDK_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
#  Manifest schema (plugin.toml)
# ─────────────────────────────────────────────────────────────────────────────
class HardwareRequirement(BaseModel):
    kind:     str
    role:     str
    optional: bool = False

    @field_validator("kind")
    @classmethod
    def _kind(cls, v):
        if v not in {"detector", "positioner", "laser", "recording", "custom"}:
            raise ValueError(f"unknown hardware kind: {v}")
        return v


class UIManifest(BaseModel):
    dist_dir:   str = "ui/dist"
    scope:      str
    exposed:    str = "./Widget"
    menu_label: str
    menu_icon:  str  = "ExtensionIcon"
    menu_group: str  = "Plugins"
    order:      int  = 100


class Permissions(BaseModel):
    camera_read:     bool = False
    camera_settings: bool = False
    file_write:      bool = False
    network_egress:  bool = False


class PluginManifest(BaseModel):
    """Pydantic mirror of the ``[plugin]`` table in ``plugin.toml``."""
    name:               str
    display_name:       str
    description:        str = ""
    version:            str
    author:             str = ""
    license:            str = "Unspecified"
    homepage:           str = ""
    imswitch_min:       str = "2.0.0"
    sdk_min:            str = "1.0.0"
    required_hardware:  List[HardwareRequirement] = Field(default_factory=list)
    permissions:        Permissions = Field(default_factory=Permissions)
    ui:                 UIManifest


def load_manifest(toml_path) -> PluginManifest:
    """Parse a ``plugin.toml`` file into a validated :class:`PluginManifest`.

    ``toml_path`` can be anything ``open()`` accepts in binary mode — a
    :class:`pathlib.Path`, a string, or an ``importlib.resources`` traversable.
    """
    with open(str(toml_path), "rb") as f:
        raw = tomllib.load(f)
    block = raw.get("plugin", {})
    return PluginManifest(**block)


# ─────────────────────────────────────────────────────────────────────────────
#  Decorator: @APIExport
# ─────────────────────────────────────────────────────────────────────────────
def APIExport(method: str = "GET", path: Optional[str] = None):
    """Mark a controller method as an HTTP endpoint.

    The :class:`~imswitch.plugin_manager.PluginManager` scans every
    controller for methods carrying this marker and builds an FastAPI
    ``APIRouter``. The URL is
    ``/plugin/<plugin-name>/api/<path-or-fn-name>``.

    ``method`` is one of ``"GET"`` or ``"POST"``. ``path`` defaults to
    ``"/<function-name>"``.
    """
    def decorator(fn: Callable) -> Callable:
        fn.__api_export__ = {                                # noqa: SLF001
            "method": method.upper(),
            "path":   path or f"/{fn.__name__}",
        }
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
#  Declarative event channel
# ─────────────────────────────────────────────────────────────────────────────
class Event:
    """An event the plugin emits to the frontend via Socket.IO.

    Declare it as a class attribute on your controller::

        class MyController(PluginController):
            sig_measurement = Event("measurement", schema={"value": "float"})

            def do_something(self):
                self.sig_measurement.emit({"value": 42.0})

    The PluginManager binds the descriptor to the per-plugin Socket.IO
    namespace ``/plugin/<name>`` when the controller is instantiated.
    """
    def __init__(self, name: str, schema: Optional[Dict[str, str]] = None):
        self.name = name
        self.schema = schema or {}
        self._bound_emit: Optional[Callable[[dict], None]] = None

    def emit(self, payload: dict) -> None:
        if self._bound_emit is None:
            # Never raise — silently drop. Plugins emit during normal
            # operation and a missing socket bus should not blow up an
            # otherwise healthy controller. The PluginManager logs at bind
            # time if the socket app is unavailable.
            return
        self._bound_emit(payload)

    def _bind(self, emit_fn: Callable[[dict], None]) -> None:
        self._bound_emit = emit_fn


# ─────────────────────────────────────────────────────────────────────────────
#  Hardware access — typed handles
# ─────────────────────────────────────────────────────────────────────────────
class _DetectorHandle:
    """Typed accessor for a detector that the host has assigned to a role."""
    def __init__(self, master, device_name: str):
        self._master = master
        self.name = device_name

    def _device(self):
        return self._master.detectorsManager[self.name]

    def get_latest_frame(self):
        dev = self._device()
        # Most detectors expose either getLatestFrame() or a property called
        # `latestFrame`. Probe in order.
        for attr in ("getLatestFrame", "latestFrame"):
            obj = getattr(dev, attr, None)
            if obj is None:
                continue
            return obj() if callable(obj) else obj
        raise AttributeError(
            f"detector {self.name!r} does not expose a frame accessor")

    def get(self, key: str):
        det = self._device()
        for attr in (f"get{key.title()}", key, f"{key}_value"):
            v = getattr(det, attr, None)
            if v is None:
                continue
            return v() if callable(v) else v
        return None

    def set(self, key: str, value):
        det = self._device()
        for method in (f"set{key.title()}", "setParameter"):
            fn = getattr(det, method, None)
            if fn is None:
                continue
            if method == "setParameter":
                fn(key, value)
            else:
                fn(value)
            return


class _PositionerHandle:
    def __init__(self, master, device_name: str):
        self._master = master
        self.name = device_name

    def _device(self):
        return self._master.positionersManager[self.name]

    def move(self, axis: str, value: float, is_absolute: bool = False):
        self._device().move(value, axis, is_absolute)

    def get_position(self, axis: Optional[str] = None):
        pos = self._device().position
        return pos[axis] if axis else pos


class _LaserHandle:
    def __init__(self, master, device_name: str):
        self._master = master
        self.name = device_name

    def _device(self):
        return self._master.lasersManager[self.name]

    def set_enabled(self, on: bool):
        self._device().setEnabled(bool(on))

    def set_value(self, value: float):
        self._device().setValue(float(value))


class _HardwareAPI:
    """Single entry point a plugin uses to reach hardware.

    Roles are resolved from the manifest, so the controller never sees raw
    device names. The PluginManager populates ``bindings`` before handing
    the context to the controller.
    """
    def __init__(self, master, bindings: Dict[str, str]):
        self._master = master
        self._bindings = bindings   # role-key (kind:role) -> device-name

    def _resolve(self, kind: str, role: str) -> str:
        key = f"{kind}:{role}"
        if key not in self._bindings:
            raise KeyError(
                f"plugin requested {kind}:{role!r} but no binding was "
                f"declared in plugin.toml or resolved at load time")
        return self._bindings[key]

    def detector(self, role: str) -> _DetectorHandle:
        return _DetectorHandle(self._master, self._resolve("detector", role))

    def positioner(self, role: str) -> _PositionerHandle:
        return _PositionerHandle(self._master, self._resolve("positioner", role))

    def laser(self, role: str) -> _LaserHandle:
        return _LaserHandle(self._master, self._resolve("laser", role))


# ─────────────────────────────────────────────────────────────────────────────
#  PluginContext — what every plugin receives
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PluginContext:
    """Runtime context injected into every plugin.

    Plugins read from ``ctx.hardware``, ``ctx.logger``, and ``ctx.manifest``
    and otherwise know nothing about the host.
    """
    master:        Any
    setup_info:    Any
    socket_app:    Any = None
    source:        str = ""

    # Filled in by the PluginManager after manifest validation.
    manifest:      Optional[PluginManifest] = None
    hardware:      Optional[_HardwareAPI]   = None
    logger:        logging.Logger           = field(
        default_factory=lambda: logging.getLogger("imswitch.plugin"))

    # ── used internally by the PluginManager ──────────────────────────────
    def bind_hardware(self, manifest: PluginManifest,
                      bindings: Dict[str, str]) -> "PluginContext":
        """Return a new context with the manifest + hardware bindings applied.

        The PluginManager calls this once per plugin, after it has resolved
        every entry in ``manifest.required_hardware`` against the active
        setup file. Plugins never call this themselves.
        """
        return PluginContext(
            master      = self.master,
            setup_info  = self.setup_info,
            socket_app  = self.socket_app,
            source      = self.source,
            manifest    = manifest,
            hardware    = _HardwareAPI(self.master, bindings),
            logger      = logging.getLogger(f"imswitch.plugin.{manifest.name}"),
        )

    def build_router(self, controller: "PluginController") -> APIRouter:
        """Build a FastAPI :class:`APIRouter` from a controller's methods.

        Called by the PluginManager after the controller is instantiated.
        Also binds any declarative :class:`Event` objects to the plugin's
        per-plugin Socket.IO namespace.
        """
        router = APIRouter()
        # Collect @APIExport-decorated methods
        for attr in dir(controller):
            fn = getattr(controller, attr, None)
            meta = getattr(fn, "__api_export__", None)
            if meta is None:
                continue
            router.add_api_route(
                meta["path"], fn,
                methods=[meta["method"]],
                name=f"{controller.__class__.__name__}.{attr}",
            )

        # Bind Event() descriptors to the per-plugin Socket.IO namespace
        if self.manifest is not None:
            namespace = f"/plugin/{self.manifest.name}"
            sio = self._get_socketio()
            for attr in dir(controller.__class__):
                descriptor = getattr(controller.__class__, attr, None)
                if not isinstance(descriptor, Event):
                    continue
                evt_name = descriptor.name

                def _make_emit(name=evt_name, ns=namespace, sio=sio):
                    def emit(payload):
                        if sio is None:
                            return
                        try:
                            sio.emit(name, payload, namespace=ns)
                        except Exception:
                            # Never let a Socket.IO failure break the
                            # controller — log via the descriptor's owner.
                            self.logger.debug(
                                "socket.io emit failed for %s", name,
                                exc_info=True)
                    return emit
                descriptor._bind(_make_emit())
        return router

    def _get_socketio(self):
        """Pull the underlying python-socketio Server out of the ASGI app.

        ImSwitch's noqt framework wraps the python-socketio ``AsyncServer``
        in an ASGI app. Different versions stash the server in different
        attributes; probe a few common ones.
        """
        app = self.socket_app
        if app is None:
            return None
        for attr in ("_sio", "engineio_server", "_engineio_server"):
            obj = getattr(app, attr, None)
            if obj is not None and hasattr(obj, "emit"):
                return obj
        # Some wrappers expose .sio
        return getattr(app, "sio", None)


# ─────────────────────────────────────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────────────────────────────────────
class PluginController:
    """Base class for every plugin controller.

    Subclasses receive a :class:`PluginContext` and use it (and only it) to
    interact with the host. There is no QT, no ``_master``, no
    ``setupInfo`` — those live behind the SDK.
    """
    def __init__(self, ctx: PluginContext):
        self.ctx = ctx
        self.log = ctx.logger

    # Lifecycle hook the manager calls on shutdown. Override if needed.
    def on_shutdown(self) -> None:    # noqa: D401  (deliberately simple)
        """Hook called by the PluginManager when ImSwitch is stopping."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  PluginRegistration — the return type of register()
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PluginRegistration:
    """What a plugin's ``register(ctx)`` returns to the host."""
    manifest:           PluginManifest
    controller_factory: Type[PluginController]
    ui_dir:             Optional[str] = None
