"""
imswitch.plugin_manager
=======================

Replacement for the old ``createAPI()`` / ``_uiapi`` machinery in
``ImSwitchServer.py``. Loads plugins from two sources, validates them
against the manifest schema, instantiates one controller per plugin, and
mounts both backend routes and frontend bundles onto the FastAPI app.

Design goals
------------
1. **One source of truth** — the plugin manifest. The host never has to
   read the controller's source to learn about the plugin.
2. **Decoupling** — plugins import only :mod:`imswitch.plugin_sdk`;
   everything else is private to ImSwitch.
3. **Two install paths** — Python entry points (``imswitch.plugins``) for
   ``pip install``-style deployment and a filesystem drop-in directory
   (Docker bind mount or user customisation) keyed off
   ``$IMSWITCH_PLUGIN_DIR``.
4. **Predictable error model** — a malformed plugin is reported via
   ``/api/plugins`` with a ``status`` of ``"error"`` and a human-readable
   message; it never crashes the host.

Back-compat (Phase 1 of the migration plan)
-------------------------------------------
While both the v1 ``@UIExport`` mechanism and this manager coexist, the
manager **also appends** a v1-shaped record to ``_ui_manifests`` for every
loaded plugin, so the existing React shell (which already reads
``/api/plugins`` and looks for ``remote``/``scope``/``exposed``) keeps
working without a frontend change.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from imswitch.imcommon.model import initLogger
from imswitch.plugin_sdk import (
    PluginContext,
    PluginManifest,
    PluginRegistration,
)


# Entry-point group all pip-installed ImSwitch plugins must use.
ENTRY_POINT_GROUP = "imswitch.plugins"

# Drop-in directory scanned at startup. In the official Docker image this is
# /opt/imswitch/plugins and the user bind-mounts plugin packages into it.
DROPIN_ENV_VAR    = "IMSWITCH_PLUGIN_DIR"
DEFAULT_DROPIN    = "/opt/imswitch/plugins"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoadedPlugin:
    """Bookkeeping struct for one plugin that has been successfully loaded."""
    manifest:   PluginManifest
    controller: Any
    router:     APIRouter
    ui_dir:     Optional[Path]
    mount:      str           # e.g. "/plugin/goniometer"
    source:     str           # "entry_point:goniometer" or "dropin:/opt/..."


# ─────────────────────────────────────────────────────────────────────────────
class PluginManager:
    """The single object that loads, holds, and exposes v2 plugins.

    Lifecycle
    ---------
    The host calls :meth:`discover` once, after the hardware managers have
    been instantiated but before ``uvicorn`` starts serving requests. The
    manager keeps references to all controllers so they are not
    garbage-collected.

    :meth:`attach_to_app` wires routers and static mounts onto the main
    FastAPI app.
    """

    def __init__(self, master, setup_info, socket_app=None,
                 legacy_manifest_sink: Optional[list] = None):
        self._log         = initLogger(self)
        self._master      = master
        self._setup_info  = setup_info
        self._socket_app  = socket_app
        # Optional list to receive v1-shaped manifest records for back-compat
        # with the existing /api/plugins endpoint.
        self._legacy_sink = legacy_manifest_sink
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._errors:  List[Dict[str, str]]    = []

    # ── public API ─────────────────────────────────────────────────────────
    def discover(self) -> None:
        """Find, validate, and instantiate every available plugin."""
        for source, register_fn in self._iter_register_fns():
            try:
                reg = register_fn(self._make_context_for(source))
                self._activate(source, reg)
            except Exception as e:
                self._record_error(source, e)

    def attach_to_app(self, app: FastAPI) -> None:
        """Mount every loaded plugin onto the given FastAPI application.

        Mount layout per plugin::

            /plugin/<name>/api/*    → controller's APIRouter
            /plugin/<name>/ui/*     → built React bundle (static files)
        """
        for plugin in self._plugins.values():
            try:
                app.include_router(plugin.router, prefix=f"{plugin.mount}/api")
            except Exception:
                self._log.exception(
                    "failed to mount router for plugin %s", plugin.manifest.name)
                continue
            if plugin.ui_dir and plugin.ui_dir.is_dir():
                try:
                    app.mount(
                        f"{plugin.mount}/ui",
                        StaticFiles(directory=str(plugin.ui_dir), html=False),
                        name=f"plugin_{plugin.manifest.name}_ui",
                    )
                except Exception:
                    self._log.exception(
                        "failed to mount UI for plugin %s",
                        plugin.manifest.name)
            self._log.info("mounted plugin %r at %s (ui=%s)",
                           plugin.manifest.name, plugin.mount,
                           plugin.ui_dir is not None)
            # Back-compat: also publish a v1-shaped record so the existing
            # /api/plugins endpoint keeps serving us.
            if self._legacy_sink is not None:
                self._legacy_sink.append(self._legacy_record(plugin))

    def manifest_list(self) -> List[Dict[str, Any]]:
        """v2 manifest payload (richer than the v1 list)."""
        out = []
        for p in self._plugins.values():
            ui = p.manifest.ui
            out.append({
                "name":         p.manifest.name,
                "display_name": p.manifest.display_name,
                "version":      p.manifest.version,
                "status":       "ok",
                "menu": {
                    "label": ui.menu_label,
                    "icon":  ui.menu_icon,
                    "group": ui.menu_group,
                    "order": ui.order,
                },
                "remote_entry": f"{p.mount}/ui/remoteEntry.js" if p.ui_dir else None,
                "scope":        ui.scope,
                "exposed":      ui.exposed,
                "api_base":     f"{p.mount}/api",
                "socket_ns":    f"{p.mount}",
            })
        return out

    def errors(self) -> List[Dict[str, str]]:
        return list(self._errors)

    def shutdown(self) -> None:
        for p in self._plugins.values():
            try:
                if hasattr(p.controller, "on_shutdown"):
                    p.controller.on_shutdown()
            except Exception:
                self._log.exception(
                    "shutdown failed for %s", p.manifest.name)

    # ── discovery: entry-points + drop-in ──────────────────────────────────
    def _iter_register_fns(self):
        # (1) pip-installed plugins (the production path)
        try:
            eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:                                # py<3.10 fallback
            eps = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])
        for ep in eps:
            try:
                fn = ep.load()
            except Exception as e:
                self._record_error(f"entry_point:{ep.name}", e)
                continue
            yield f"entry_point:{ep.name}", fn

        # (2) drop-in directory (development + user customisation)
        root = Path(os.environ.get(DROPIN_ENV_VAR, DEFAULT_DROPIN))
        if not root.is_dir():
            return
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "src").is_dir() \
                    and not (child / "__init__.py").is_file() \
                    and not any(child.glob("*/__init__.py")):
                continue
            try:
                fn = self._load_dropin(child)
            except Exception as e:
                self._record_error(f"dropin:{child}", e)
                continue
            yield f"dropin:{child}", fn

    def _load_dropin(self, pkg_dir: Path) -> Callable:
        """Import a plugin from a filesystem directory and return ``register``."""
        # Pick the directory that contains the actual Python package.
        src_dir = pkg_dir / "src" if (pkg_dir / "src").is_dir() else pkg_dir
        pkg_name = None
        for c in src_dir.iterdir():
            if c.is_dir() and (c / "__init__.py").is_file():
                pkg_name = c.name
                break
        if pkg_name is None:
            raise FileNotFoundError(f"no python package in {pkg_dir}")

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        mod = importlib.import_module(pkg_name)
        if not hasattr(mod, "register"):
            raise AttributeError(f"{pkg_name} has no register() function")
        return mod.register

    # ── activation ─────────────────────────────────────────────────────────
    def _make_context_for(self, source: str) -> PluginContext:
        return PluginContext(
            master      = self._master,
            setup_info  = self._setup_info,
            socket_app  = self._socket_app,
            source      = source,
        )

    def _activate(self, source: str, reg: PluginRegistration) -> None:
        manifest = reg.manifest
        name = manifest.name

        if name in self._plugins:
            raise ValueError(
                f"plugin {name!r} already loaded from "
                f"{self._plugins[name].source}")

        # Resolve required hardware roles against the active setup.
        bindings, unmet = self._resolve_hardware(manifest)
        if unmet:
            raise RuntimeError(
                f"plugin {name!r} cannot load — required hardware not "
                f"available: " + ", ".join(unmet))

        # Build the controller's context, then instantiate it.
        ctx = self._make_context_for(source).bind_hardware(manifest, bindings)
        controller = reg.controller_factory(ctx)
        router     = ctx.build_router(controller)

        ui_dir = Path(reg.ui_dir) if reg.ui_dir else None
        self._plugins[name] = LoadedPlugin(
            manifest   = manifest,
            controller = controller,
            router     = router,
            ui_dir     = ui_dir,
            mount      = f"/plugin/{name}",
            source     = source,
        )
        self._log.info("loaded plugin %r v%s from %s",
                       name, manifest.version, source)

    def _resolve_hardware(self, manifest: PluginManifest):
        """Resolve every ``required_hardware`` entry to a concrete device name.

        Strategy (priority order):

        1. ``setup_info.plugin_bindings["<kind>:<role>"] = "<device>"`` —
           explicit binding declared in the user's setup file.
        2. ``setup_info.<plugin_name>.<role>`` — legacy v1 alias, e.g.
           ``setupInfo.goniometer.camera = "MyCamera"``. Kept for
           back-compat with v1 setups.
        3. First available device of the right kind. Convenient during
           development; future work can gate this behind a host flag.
        """
        bindings: Dict[str, str] = {}
        unmet:    List[str]       = []
        explicit = getattr(self._setup_info, "plugin_bindings", None) or {}
        plugin_alias = getattr(self._setup_info, manifest.name, None)

        for req in manifest.required_hardware:
            key    = f"{req.kind}:{req.role}"
            device = explicit.get(key)
            if device is None and plugin_alias is not None:
                device = getattr(plugin_alias, req.role, None)
            if device is None:
                device = self._first_available(req.kind)
            if device is None:
                if not req.optional:
                    unmet.append(key)
                continue
            bindings[key] = device
        return bindings, unmet

    def _first_available(self, kind: str) -> Optional[str]:
        mgr_attr = {
            "detector":   "detectorsManager",
            "positioner": "positionersManager",
            "laser":      "lasersManager",
        }.get(kind)
        if mgr_attr is None:
            return None
        mgr = getattr(self._master, mgr_attr, None)
        if mgr is None or not hasattr(mgr, "getAllDeviceNames"):
            return None
        names = mgr.getAllDeviceNames()
        return names[0] if names else None

    # ── errors / back-compat ───────────────────────────────────────────────
    def _record_error(self, source: str, exc: Exception) -> None:
        self._log.error("plugin %s failed to load:\n%s",
                        source, traceback.format_exc())
        self._errors.append({
            "source":  source,
            "error":   f"{type(exc).__name__}: {exc}",
        })

    @staticmethod
    def _legacy_record(plugin: LoadedPlugin) -> Dict[str, Any]:
        """Build a manifest entry that *both* the v1 and v2 React shells
        understand.

        Schema overview:

        * ``name``, ``icon``, ``scope``, ``exposed`` — v1 fields the
          existing shell already reads.
        * ``remote_entry`` — absolute URL path (rooted under the FastAPI
          app's ``root_path``) the v2 shell uses with Module Federation.
        * ``api_base`` / ``socket_ns`` — v2 fields for the new loader.

        The existing v1 frontend prepends
        ``${hostIP}:${apiPort}/imswitch/api`` to its ``remote`` field. We
        expose ``remote_entry`` as an absolute path (``/imswitch/plugin/...``)
        instead, and let the updated shell (see
        ``frontend/src/App.jsx``) prefer it over the legacy fields when
        present.
        """
        ui = plugin.manifest.ui
        has_ui = plugin.ui_dir is not None
        remote_entry = f"{plugin.mount}/ui/remoteEntry.js" if has_ui else None
        return {
            "name":         plugin.manifest.name,
            "display_name": plugin.manifest.display_name,
            "version":      plugin.manifest.version,
            "status":       "ok",
            # menu metadata
            "icon":         ui.menu_icon,
            "label":        ui.menu_label,
            "group":        ui.menu_group,
            "order":        ui.order,
            "menu": {
                "label": ui.menu_label,
                "icon":  ui.menu_icon,
                "group": ui.menu_group,
                "order": ui.order,
            },
            # module federation
            "scope":        ui.scope,
            "exposed":      ui.exposed if ui.exposed.startswith("./") else f"./{ui.exposed}",
            "remote_entry": remote_entry,
            # API + socket
            "api_base":     f"{plugin.mount}/api",
            "socket_ns":    f"{plugin.mount}",
            # legacy keys (kept so older shells don't crash)
            "path":         str(plugin.ui_dir) if has_ui else "",
            "url":          f"{plugin.mount}/ui/index.html" if has_ui else None,
        }
