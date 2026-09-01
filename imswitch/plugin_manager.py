"""
imswitch.plugin_manager
=======================

Discovers plugins, validates them against the manifest schema, instantiates
one controller per plugin, and mounts both backend routes and frontend
bundles onto the FastAPI app.

1. **One source of truth** — the plugin manifest. The host never has to
   read the controller's source to learn about the plugin.
2. **Decoupling** — plugins import only :mod:`imswitch.plugin_sdk`;
   everything else is private to ImSwitch.
3. **One install path** — a filesystem directory. Plugins are directories
   that get bind-mounted (Docker) or copied (native) into the plugin
   directory, and imported from there. They are never ``pip install``-ed,
   which is what guarantees a plugin cannot drag a second copy of a
   library the host already has loaded into the process.
4. **Predictable error model** — a malformed plugin is reported via
   ``/api/plugins`` with a ``status`` of ``"error"`` and a human-readable
   message; it never crashes the host.

The plugin directory is chosen, in order: the path passed to
:func:`set_plugin_dir` (from ``main(plugin_dir=...)``), then
``$IMSWITCH_PLUGIN_DIR``, then :data:`DEFAULT_DROPIN`.
"""
from __future__ import annotations

import importlib
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


# Directory scanned at startup. In the official Docker image this is
# /opt/imswitch/plugins and the operator bind-mounts plugin directories into it.
DROPIN_ENV_VAR = "IMSWITCH_PLUGIN_DIR"
DEFAULT_DROPIN = "/opt/imswitch/plugins"

# Set by set_plugin_dir(), i.e. by main(plugin_dir=...). Takes precedence over
# the environment variable, so a developer running natively can point ImSwitch
# at a directory in one place rather than juggling shell configuration.
_PLUGIN_DIR_OVERRIDE: Optional[str] = None


def set_plugin_dir(path: Optional[str]) -> None:
    """Set the plugin directory explicitly. ``None`` clears the override."""
    global _PLUGIN_DIR_OVERRIDE
    _PLUGIN_DIR_OVERRIDE = str(path) if path else None


def dropin_root() -> Path:
    """Directory scanned for plugins.

    Precedence: ``main(plugin_dir=...)`` → ``$IMSWITCH_PLUGIN_DIR`` →
    :data:`DEFAULT_DROPIN`.

    Resolved on every call, never cached at import time, so a container whose
    entrypoint exports the variable after Python has already imported this
    module still works — as do tests that monkeypatch :data:`DEFAULT_DROPIN`.
    """
    return Path(
        _PLUGIN_DIR_OVERRIDE
        or os.environ.get(DROPIN_ENV_VAR)
        or DEFAULT_DROPIN
    )


def _plugin_root_for(register_fn: Callable) -> Optional[Path]:
    """Filesystem directory of the package that provided ``register_fn``.

    Stamped onto the function by :meth:`PluginManager._load_dropin`, and used
    for the ``ui_dir`` fallback in :meth:`PluginManager._activate`.
    """
    stamped = getattr(register_fn, "__imswitch_plugin_root__", None)
    return Path(stamped) if stamped else None


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoadedPlugin:
    """Bookkeeping struct for one plugin that has been successfully loaded."""
    manifest:   PluginManifest
    controller: Any
    router:     APIRouter
    ui_dir:     Optional[Path]
    mount:      str           # e.g. "/plugin/goniometer"
    source:     str           # "dropin:/opt/imswitch/plugins/goniometer"


@dataclass
class DisabledPlugin:
    """A plugin whose manifest parsed fine but which ``availableWidgets`` gated.

    Kept — and reported — deliberately. A plugin that simply does not appear is
    the worst debugging experience we can hand an operator, so a gated plugin
    stays visible with an explicit reason instead of vanishing.
    """
    manifest: PluginManifest
    source:   str
    reason:   str


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

    def __init__(self, master, setup_info, socket_app=None):
        self._log        = initLogger(self)
        self._master     = master
        self._setup_info = setup_info
        self._socket_app = socket_app
        self._plugins:  Dict[str, LoadedPlugin]   = {}
        self._disabled: Dict[str, DisabledPlugin] = {}
        self._errors:   List[Dict[str, str]]      = []
        # Filled in by attach_to_app() from the host app's root_path. See the
        # note there for why the browser-facing URLs need it.
        self._url_prefix: str = ""

    # ── public API ─────────────────────────────────────────────────────────
    def discover(self) -> None:
        """Find, validate, and instantiate every available plugin."""
        for source, register_fn in self._iter_register_fns():
            try:
                reg = register_fn(self._make_context_for(source))
                self._activate(source, reg,
                               plugin_root=_plugin_root_for(register_fn))
            except Exception as e:
                self._record_error(source, e)
        self._log_summary()

    def _log_summary(self) -> None:
        """One INFO line per plugin. This is what an operator reads when a bind
        mount is wrong, so it names the source path explicitly."""
        self._log.info("plugin discovery finished — scanning %s", dropin_root())
        for p in self._plugins.values():
            self._log.info("  plugin %-24s v%-10s status=loaded    source=%s",
                           p.manifest.name, p.manifest.version, p.source)
        for d in self._disabled.values():
            self._log.info("  plugin %-24s v%-10s status=disabled  source=%s (%s)",
                           d.manifest.name, d.manifest.version, d.source, d.reason)
        for e in self._errors:
            self._log.info("  plugin %-24s %-11s status=error     source=%s (%s)",
                           "?", "", e["source"], e["error"])
        if not (self._plugins or self._disabled or self._errors):
            self._log.info("  no plugins found")

    def attach_to_app(self, app: FastAPI) -> None:
        """Mount every loaded plugin onto the given FastAPI application.

        Mount layout per plugin::

            /plugin/<name>/api/*    → controller's APIRouter
            /plugin/<name>/ui/*     → built React bundle (static files)

        Note on ``root_path``: the host app is created with
        ``root_path="/imswitch"`` and uvicorn serves it directly, with no proxy
        stripping that prefix — so clients send it. Starlette resolves the two
        route kinds differently under those conditions: an ``APIRouter`` route
        matches both ``/plugin/x/api/y`` and ``/imswitch/plugin/x/api/y``, but a
        ``Mount`` (which is what ``StaticFiles`` is) matches *only* the prefixed
        spelling. So the browser-facing URLs published in :meth:`manifest_list`
        must carry the prefix, exactly like the host's own React bundle at
        ``/imswitch/ui/index.html``. We capture it here rather than hard-coding
        it so a differently-mounted host still produces working URLs.
        """
        self._url_prefix = (getattr(app, "root_path", "") or "").rstrip("/")
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

    def manifest_list(self) -> List[Dict[str, Any]]:
        """v2 manifest payload consumed by the frontend.

        Includes plugins gated off by ``availableWidgets`` with
        ``status: "disabled"`` so the UI can grey them out. Their URL fields are
        ``None`` because nothing is mounted for them — a disabled plugin must
        never hand the browser a link that 404s.
        """
        out = [self._manifest_entry(p.manifest, "loaded", p.source, mount=p.mount,
                                    has_ui=p.ui_dir is not None)
               for p in self._plugins.values()]
        out += [self._manifest_entry(d.manifest, "disabled", d.source,
                                     reason=d.reason)
                for d in self._disabled.values()]
        return out

    def _manifest_entry(self, manifest: PluginManifest, status: str, source: str,
                        mount: Optional[str] = None, has_ui: bool = False,
                        reason: str = "") -> Dict[str, Any]:
        ui = manifest.ui
        # URL fields carry the host's root_path; see attach_to_app(). socket_ns
        # deliberately does not — it is a Socket.IO namespace, not a URL path.
        url = f"{self._url_prefix}{mount}" if mount else None
        return {
            "name":         manifest.name,
            "display_name": manifest.display_name,
            "version":      manifest.version,
            "status":       status,
            "source":       source,
            "reason":       reason,
            "menu": {
                "label": ui.menu_label,
                "icon":  ui.menu_icon,
                "group": ui.menu_group,
                "order": ui.order,
            },
            "remote_entry": f"{url}/ui/remoteEntry.js" if (url and has_ui) else None,
            "scope":        ui.scope,
            "exposed":      ui.exposed,
            "api_base":     f"{url}/api" if url else None,
            "socket_ns":    mount,
        }

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

    # ── discovery: the drop-in directory, and nothing else ─────────────────
    def _iter_register_fns(self):
        """Yield ``(source, register_fn)`` for every plugin in the drop-in dir.

        There is exactly one discovery mechanism. Plugins are directories that
        get mounted or copied into :func:`dropin_root`; they are never
        ``pip install``-ed, and the host does not consult entry points.
        """
        root = dropin_root()
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
        """Import a plugin from a filesystem directory and return ``register``.

        Two layouts are supported, both of which put the Python package one
        level *below* the plugin's own directory::

            <plugin-dir>/<package>/__init__.py
            <plugin-dir>/src/<package>/__init__.py

        The package cannot be the plugin directory itself: the plugin directory
        is named by the operator (often with dashes, which are not importable),
        whereas the package name has to be a valid Python identifier and unique
        across every plugin loaded into the process.
        """
        # Pick the directory that contains the actual Python package.
        src_dir = pkg_dir / "src" if (pkg_dir / "src").is_dir() else pkg_dir
        pkg_name = None
        for c in src_dir.iterdir():
            if c.is_dir() and (c / "__init__.py").is_file():
                pkg_name = c.name
                break
        if pkg_name is None:
            # The most common setup mistake, and worth a message that names the
            # fix: the scanner accepts a bare __init__.py at this level, but a
            # package has to sit one directory deeper to be importable.
            hint = ""
            if (pkg_dir / "__init__.py").is_file():
                hint = (f" — {pkg_dir.name}/__init__.py exists, so this looks "
                        f"like the package was placed directly in the plugin "
                        f"directory. Move it one level down, into "
                        f"{pkg_dir.name}/<package_name>/, where <package_name> "
                        f"is a valid Python identifier.")
            raise FileNotFoundError(
                f"no python package in {pkg_dir}: expected "
                f"{pkg_dir.name}/<package>/__init__.py or "
                f"{pkg_dir.name}/src/<package>/__init__.py{hint}")

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        mod = importlib.import_module(pkg_name)
        if not hasattr(mod, "register"):
            raise AttributeError(f"{pkg_name} has no register() function")

        register_fn = mod.register
        # Stamp the resolved package directory onto the register function.
        # A path-imported package cannot always resolve its own data files the
        # way an installed one can (importlib.resources behaves differently),
        # so the host remembers where the package actually lives and uses it as
        # the ui_dir fallback in _activate(). Plugin authors do not have to care
        # which install mode they are in.
        try:
            register_fn.__imswitch_plugin_root__ = str(src_dir / pkg_name)
        except (AttributeError, TypeError):
            pass                     # e.g. a builtin or a slotted callable
        return register_fn

    # ── activation ─────────────────────────────────────────────────────────
    def _make_context_for(self, source: str) -> PluginContext:
        return PluginContext(
            master      = self._master,
            setup_info  = self._setup_info,
            socket_app  = self._socket_app,
            source      = source,
        )

    def _activate(self, source: str, reg: PluginRegistration,
                  plugin_root: Optional[Path] = None) -> None:
        manifest = reg.manifest
        name = manifest.name

        # ADR-003: names are a flat namespace, first one wins. The caller
        # (discover) catches this and records it, so a collision costs the
        # second plugin only — it never aborts the discovery loop.
        if name in self._plugins:
            raise ValueError(
                f"plugin {name!r} already loaded from "
                f"{self._plugins[name].source}")
        if name in self._disabled:
            raise ValueError(
                f"plugin {name!r} already registered (and disabled) from "
                f"{self._disabled[name].source}")

        # Gate on the setup file BEFORE touching hardware or building the
        # controller: an inactive plugin must not claim a camera, and must not
        # run any of its own constructor code.
        enabled, reason = self._is_enabled(name)
        if not enabled:
            self._disabled[name] = DisabledPlugin(
                manifest=manifest, source=source, reason=reason)
            self._log.info("plugin %r disabled: %s", name, reason)
            return

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

        ui_dir = self._resolve_ui_dir(reg, manifest, plugin_root)
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

    def _is_enabled(self, name: str):
        """Is this plugin activated by the setup file's ``availableWidgets``?

        Returns ``(enabled, reason)``; ``reason`` is a human-readable string
        surfaced in the manifest list so an operator can see *why* a plugin is
        greyed out rather than guessing.

        One plugin directory can be bind-mounted across a fleet of microscopes
        while each instrument's setup file decides which plugins are live —
        that is what this gate is for.
        """
        available = getattr(self._setup_info, "availableWidgets", None)

        if available is True:
            return True, ""
        if isinstance(available, (list, tuple, set)):
            if name in available:
                return True, ""
            return False, (f"{name!r} is not listed in availableWidgets; "
                           f"add it to the setup file to enable this plugin")
        if available is False:
            return False, "availableWidgets is false — all widgets are disabled"
        return False, ("the active setup exposes no availableWidgets list, so "
                       "no plugin can be enabled")

    @staticmethod
    def _resolve_ui_dir(reg: PluginRegistration, manifest: PluginManifest,
                        plugin_root: Optional[Path]) -> Optional[Path]:
        """Where the built frontend bundle actually lives on disk.

        ``register()`` may return a ``ui_dir`` that does not resolve — most
        often because ``importlib.resources`` behaves differently for a
        path-imported package than for an installed one. Fall back to
        ``<plugin_root>/<manifest.ui.dist_dir>`` so bind-mounted and
        pip-installed plugins behave identically.
        """
        if reg.ui_dir:
            candidate = Path(reg.ui_dir)
            if candidate.is_dir():
                return candidate
        if plugin_root is not None:
            fallback = plugin_root / manifest.ui.dist_dir
            if fallback.is_dir():
                return fallback
        return None

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

    # ── errors ─────────────────────────────────────────────────────────────
    def _record_error(self, source: str, exc: Exception) -> None:
        self._log.error("plugin %s failed to load:\n%s",
                        source, traceback.format_exc())
        self._errors.append({
            "source":  source,
            "error":   f"{type(exc).__name__}: {exc}",
        })

