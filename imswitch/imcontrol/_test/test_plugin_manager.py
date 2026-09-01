"""Tests for v2 plugin discovery: drop-in loading and availableWidgets gating.

Builds real plugin packages on disk under ``tmp_path`` and points
``$IMSWITCH_PLUGIN_DIR`` at them, so this exercises the same code path a
bind-mounted plugin takes in the Docker image — import via ``sys.path``, no
``pip install``.

ImSwitch imports are lazy on purpose; see the note in ``test_plugin_api.py``.
"""
import importlib
import importlib.metadata
import sys
import textwrap

import pytest

pytest.importorskip("fastapi")


PLUGIN_TOML = """\
[plugin]
name         = "{name}"
display_name = "Demo Plugin"
version      = "0.1.0"

[plugin.ui]
dist_dir   = "ui/dist"
scope      = "demo_plugin"
menu_label = "Demo"
"""

PLUGIN_INIT = '''\
from pathlib import Path

from imswitch.plugin_sdk import (
    APIExport, PluginController, PluginRegistration, load_manifest,
)

# Written only if the host actually constructs the controller. A gated-off
# plugin must never get this far.
MARKER = Path(__file__).with_name("controller_constructed.marker")


class DemoController(PluginController):
    def __init__(self, ctx):
        super().__init__(ctx)
        MARKER.write_text("constructed")

    @APIExport("GET")
    def status(self):
        return {{"ok": True}}


def register(ctx):
    manifest = load_manifest(Path(__file__).with_name("plugin.toml"))
    return PluginRegistration(
        manifest=manifest,
        controller_factory=DemoController,
        ui_dir={ui_dir!r},
    )
'''


# ── helpers ─────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_imports():
    """Drop-in loading mutates sys.path and sys.modules; undo it per test."""
    path_before = list(sys.path)
    modules_before = set(sys.modules)
    yield
    sys.path[:] = path_before
    for name in set(sys.modules) - modules_before:
        del sys.modules[name]


def make_dropin(root, pkg_name, plugin_name="demo", ui_dir="__PACKAGE__",
                with_bundle=True):
    """Write a minimal but real drop-in plugin package under ``root``.

    ``ui_dir`` is what ``register()`` returns: ``"__PACKAGE__"`` resolves to the
    package's own ``ui/dist`` (the well-behaved case), ``None`` forces the host
    fallback, or pass a literal path to simulate a stale/wrong value.
    """
    pkg = root / f"{pkg_name}_dir" / pkg_name
    pkg.mkdir(parents=True)
    (pkg / "plugin.toml").write_text(PLUGIN_TOML.format(name=plugin_name))

    bundle = pkg / "ui" / "dist"
    if with_bundle:
        bundle.mkdir(parents=True)
        (bundle / "remoteEntry.js").write_text("// stub bundle")

    resolved = str(bundle) if ui_dir == "__PACKAGE__" else ui_dir
    (pkg / "__init__.py").write_text(
        PLUGIN_INIT.format(ui_dir=resolved))
    return pkg


class FakeSetupInfo:
    def __init__(self, availableWidgets):
        self.availableWidgets = availableWidgets


def build_manager(root, monkeypatch, availableWidgets):
    from imswitch.plugin_manager import PluginManager

    monkeypatch.setenv("IMSWITCH_PLUGIN_DIR", str(root))
    return PluginManager(master=None,
                         setup_info=FakeSetupInfo(availableWidgets))


def entry_for(manager, name):
    return next(e for e in manager.manifest_list() if e["name"] == name)


# ── availableWidgets gating ─────────────────────────────────────────────────
def test_loads_when_listed_in_available_widgets(tmp_path, monkeypatch):
    pkg = make_dropin(tmp_path, "demo_listed")
    manager = build_manager(tmp_path, monkeypatch, ["Settings", "demo"])
    manager.discover()

    assert manager.errors() == []
    assert list(manager._plugins) == ["demo"]
    assert (pkg / "controller_constructed.marker").is_file()

    entry = entry_for(manager, "demo")
    assert entry["status"] == "loaded"
    assert entry["api_base"] == "/plugin/demo/api"
    assert entry["remote_entry"] == "/plugin/demo/ui/remoteEntry.js"


def test_available_widgets_true_enables_everything(tmp_path, monkeypatch):
    make_dropin(tmp_path, "demo_all")
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert entry_for(manager, "demo")["status"] == "loaded"


def test_disabled_when_not_listed_and_controller_never_built(tmp_path, monkeypatch):
    pkg = make_dropin(tmp_path, "demo_gated")
    manager = build_manager(tmp_path, monkeypatch, ["Settings"])
    manager.discover()

    # The gate must fire before the controller is constructed — an inactive
    # plugin may not claim hardware or run its own __init__.
    assert not (pkg / "controller_constructed.marker").exists()
    assert manager._plugins == {}
    assert manager.errors() == []

    entry = entry_for(manager, "demo")
    assert entry["status"] == "disabled"
    assert "availableWidgets" in entry["reason"]
    # Nothing is mounted for a disabled plugin, so it must not advertise URLs.
    assert entry["remote_entry"] is None
    assert entry["api_base"] is None
    # ...but it stays visible, with enough metadata for the UI to grey it out.
    assert entry["display_name"] == "Demo Plugin"
    assert entry["menu"]["label"] == "Demo"


def test_disabled_plugin_is_not_mounted(tmp_path, monkeypatch):
    from fastapi import FastAPI

    make_dropin(tmp_path, "demo_unmounted")
    manager = build_manager(tmp_path, monkeypatch, [])
    manager.discover()

    app = FastAPI()
    manager.attach_to_app(app)
    assert not [r for r in app.routes if "/plugin/demo" in getattr(r, "path", "")]


def test_advertised_urls_actually_resolve_on_the_host_app(tmp_path, monkeypatch):
    """The URLs in the manifest must be fetchable as published.

    Starlette resolves the two route kinds differently once the host app has a
    root_path: an APIRouter route matches with or without the prefix, but a
    StaticFiles Mount matches *only* with it. Publishing an unprefixed
    remote_entry therefore 404s the plugin's bundle in production even though
    the mount exists. Assert against the real app, not against string shapes.
    """
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    make_dropin(tmp_path, "demo_urls")
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    app = FastAPI(root_path="/imswitch")
    app.include_router(APIRouter(prefix="/api"))
    manager.attach_to_app(app)
    client = TestClient(app)

    entry = entry_for(manager, "demo")
    assert entry["remote_entry"] == "/imswitch/plugin/demo/ui/remoteEntry.js"
    assert entry["api_base"] == "/imswitch/plugin/demo/api"
    # The Socket.IO namespace is not a URL and must stay unprefixed.
    assert entry["socket_ns"] == "/plugin/demo"

    bundle = client.get(entry["remote_entry"])
    assert bundle.status_code == 200
    assert "stub bundle" in bundle.text
    assert client.get(f"{entry['api_base']}/status").status_code == 200


def test_urls_are_unprefixed_when_the_host_has_no_root_path(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    make_dropin(tmp_path, "demo_noroot")
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    app = FastAPI()
    manager.attach_to_app(app)

    entry = entry_for(manager, "demo")
    assert entry["remote_entry"] == "/plugin/demo/ui/remoteEntry.js"
    assert TestClient(app).get(entry["remote_entry"]).status_code == 200


def test_available_widgets_false_disables_everything(tmp_path, monkeypatch):
    make_dropin(tmp_path, "demo_off")
    manager = build_manager(tmp_path, monkeypatch, False)
    manager.discover()

    assert manager._plugins == {}
    assert entry_for(manager, "demo")["status"] == "disabled"


# ── ui_dir resolution ───────────────────────────────────────────────────────
def test_ui_dir_falls_back_to_package_root_when_register_returns_none(
        tmp_path, monkeypatch):
    pkg = make_dropin(tmp_path, "demo_noui", ui_dir=None)
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert manager._plugins["demo"].ui_dir == pkg / "ui" / "dist"
    assert entry_for(manager, "demo")["remote_entry"] == \
        "/plugin/demo/ui/remoteEntry.js"


def test_ui_dir_falls_back_when_register_returns_a_stale_path(
        tmp_path, monkeypatch):
    pkg = make_dropin(tmp_path, "demo_stale",
                      ui_dir=str(tmp_path / "does" / "not" / "exist"))
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert manager._plugins["demo"].ui_dir == pkg / "ui" / "dist"


def test_plugin_without_a_bundle_reports_no_remote_entry(tmp_path, monkeypatch):
    make_dropin(tmp_path, "demo_headless", ui_dir=None, with_bundle=False)
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert manager._plugins["demo"].ui_dir is None
    assert entry_for(manager, "demo")["remote_entry"] is None


# ── duplicate names (ADR-003) ───────────────────────────────────────────────
def test_duplicate_name_is_an_error_and_does_not_block_the_first(
        tmp_path, monkeypatch):
    make_dropin(tmp_path, "aaa_first")
    make_dropin(tmp_path, "zzz_second")
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    # First one wins; the collision costs only the second plugin.
    assert list(manager._plugins) == ["demo"]
    assert "aaa_first" in manager._plugins["demo"].source

    assert len(manager.errors()) == 1
    assert "already loaded" in manager.errors()[0]["error"]
    assert "zzz_second" in manager.errors()[0]["source"]


def test_package_directly_under_the_root_gets_an_actionable_error(
        tmp_path, monkeypatch):
    """The layout people try first: package placed straight in the plugin dir.

    The scanner accepts it (there is an __init__.py) but the loader cannot
    import it, because the plugin directory is named by whoever deploys it and
    the package name has to be an identifier. That is defensible, but the error
    has to say so — this is a setup mistake, not a bug in the plugin.
    """
    pkg = tmp_path / "imswitch_plugin_flat"
    pkg.mkdir()
    (pkg / "plugin.toml").write_text(PLUGIN_TOML.format(name="flat"))
    (pkg / "__init__.py").write_text(PLUGIN_INIT.format(ui_dir=None))

    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert manager._plugins == {}
    assert len(manager.errors()) == 1
    message = manager.errors()[0]["error"]
    # Names the two layouts that do work, and the fix for this one.
    assert "<package>/__init__.py" in message
    assert "src/<package>/__init__.py" in message
    assert "Move it one level down" in message


def test_a_broken_plugin_does_not_stop_a_healthy_one(tmp_path, monkeypatch):
    broken = tmp_path / "broken_dir" / "broken_pkg"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text("# no register() here\n")
    make_dropin(tmp_path, "healthy_pkg", plugin_name="healthy")

    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert list(manager._plugins) == ["healthy"]
    assert len(manager.errors()) == 1
    assert "register()" in manager.errors()[0]["error"]


# ── drop-in root resolution ─────────────────────────────────────────────────
def test_default_dropin_is_read_at_call_time(tmp_path, monkeypatch):
    """DEFAULT_DROPIN must be resolved per call, not captured at import."""
    plugin_manager = importlib.import_module("imswitch.plugin_manager")

    make_dropin(tmp_path, "demo_default")
    monkeypatch.delenv("IMSWITCH_PLUGIN_DIR", raising=False)
    monkeypatch.setattr(plugin_manager, "DEFAULT_DROPIN", str(tmp_path))

    assert plugin_manager.dropin_root() == tmp_path

    manager = plugin_manager.PluginManager(
        master=None, setup_info=FakeSetupInfo(True))
    manager.discover()
    assert list(manager._plugins) == ["demo"]


def test_env_var_wins_over_default(tmp_path, monkeypatch):
    plugin_manager = importlib.import_module("imswitch.plugin_manager")

    monkeypatch.setattr(plugin_manager, "DEFAULT_DROPIN", str(tmp_path / "nope"))
    monkeypatch.setenv("IMSWITCH_PLUGIN_DIR", str(tmp_path))
    assert plugin_manager.dropin_root() == tmp_path


def test_explicit_plugin_dir_wins_over_env_and_default(tmp_path, monkeypatch):
    """main(plugin_dir=...) must beat both, so a developer can point ImSwitch
    at a directory in one place instead of juggling shell configuration."""
    plugin_manager = importlib.import_module("imswitch.plugin_manager")

    monkeypatch.setattr(plugin_manager, "DEFAULT_DROPIN", str(tmp_path / "default"))
    monkeypatch.setenv("IMSWITCH_PLUGIN_DIR", str(tmp_path / "from-env"))
    try:
        plugin_manager.set_plugin_dir(str(tmp_path / "explicit"))
        assert plugin_manager.dropin_root() == tmp_path / "explicit"

        # Clearing falls back to the environment variable...
        plugin_manager.set_plugin_dir(None)
        assert plugin_manager.dropin_root() == tmp_path / "from-env"

        # ...and then to the default.
        monkeypatch.delenv("IMSWITCH_PLUGIN_DIR")
        assert plugin_manager.dropin_root() == tmp_path / "default"
    finally:
        plugin_manager.set_plugin_dir(None)


def test_explicit_plugin_dir_actually_loads_plugins(tmp_path, monkeypatch):
    plugin_manager = importlib.import_module("imswitch.plugin_manager")

    make_dropin(tmp_path, "demo_explicit")
    monkeypatch.delenv("IMSWITCH_PLUGIN_DIR", raising=False)
    monkeypatch.setattr(plugin_manager, "DEFAULT_DROPIN", str(tmp_path / "wrong"))
    try:
        plugin_manager.set_plugin_dir(str(tmp_path))
        manager = plugin_manager.PluginManager(
            master=None, setup_info=FakeSetupInfo(True))
        manager.discover()
        assert list(manager._plugins) == ["demo"]
    finally:
        plugin_manager.set_plugin_dir(None)


def test_pip_installed_plugins_are_not_discovered(tmp_path, monkeypatch):
    """Entry points are not a deployment mechanism. Discovery is directory-only,
    so a package declaring `imswitch.plugins` must be ignored entirely."""
    plugin_manager = importlib.import_module("imswitch.plugin_manager")

    assert not hasattr(plugin_manager, "ENTRY_POINT_GROUP")

    called = []

    def _fail_if_called(*args, **kwargs):
        called.append(args)
        raise AssertionError("entry_points() must not be consulted")

    monkeypatch.setattr(importlib.metadata, "entry_points", _fail_if_called)

    make_dropin(tmp_path, "demo_dironly")
    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert list(manager._plugins) == ["demo"]
    assert called == []


def test_missing_dropin_dir_is_not_an_error(tmp_path, monkeypatch):
    manager = build_manager(tmp_path / "absent", monkeypatch, True)
    manager.discover()

    assert manager._plugins == {}
    assert manager.errors() == []


# ── hardware requirements ───────────────────────────────────────────────────
def test_unmet_required_hardware_is_reported_as_an_error(tmp_path, monkeypatch):
    pkg = make_dropin(tmp_path, "demo_hw")
    (pkg / "plugin.toml").write_text(
        PLUGIN_TOML.format(name="demo") + textwrap.dedent("""
            [[plugin.required_hardware]]
            kind     = "detector"
            role     = "camera"
            optional = false
        """))

    manager = build_manager(tmp_path, monkeypatch, True)
    manager.discover()

    assert manager._plugins == {}
    assert len(manager.errors()) == 1
    assert "detector:camera" in manager.errors()[0]["error"]
    # Reported, not raised — a plugin that cannot get its hardware must not
    # take the server down.
    assert not (pkg / "controller_constructed.marker").exists()
