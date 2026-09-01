"""End-to-end test for the v2 plugin chain: bind-mounted directory → served bundle.

This is the definition of "the plugin system works". It exercises every link the
manual deployment path uses — drop-in discovery, manifest parsing, the
availableWidgets gate, hardware role resolution, router mounting, static UI
mounting, and the HTTP surface the React shell consumes — in one process.

It exists because that chain crosses two languages, a container boundary and
four files, and will otherwise regress silently the first time someone reorders
startup in ImSwitchServer.run() or touches the federation config.

Hermetic on purpose: no Docker, no hardware, no network, no uvicorn.

ImSwitch imports are lazy; importing ImSwitchServer at collection time mounts
Socket.IO onto its module-global app and breaks the real-server fixture in
_test/api/. See the note in test_plugin_api.py.
"""
import importlib
import sys
import textwrap

import pytest

pytest.importorskip("fastapi")


MARKER = "__IMSWITCH_E2E_BUNDLE_MARKER__"

GOOD_MANIFEST = """\
[plugin]
name         = "e2e"
display_name = "E2E Plugin"
version      = "1.2.3"

[[plugin.required_hardware]]
kind     = "detector"
role     = "camera"
optional = false

[plugin.ui]
dist_dir   = "ui/dist"
scope      = "e2e_plugin"
exposed    = "./Widget"
menu_label = "E2E"
menu_icon  = "Science"
menu_group = "apps"
order      = 7
"""

PLUGIN_INIT = '''\
from pathlib import Path

from imswitch.plugin_sdk import (
    APIExport, Event, PluginController, PluginRegistration, load_manifest,
)

CONSTRUCTED = Path(__file__).with_name("constructed.marker")


class E2EController(PluginController):
    sig_done = Event("done", schema={"value": "float"})

    def __init__(self, ctx):
        super().__init__(ctx)
        CONSTRUCTED.write_text("yes")
        self._camera = ctx.hardware.detector("camera")

    @APIExport()
    def status(self):
        return {"ok": True, "camera": self._camera.name}

    @APIExport(method="POST")
    def compute(self, value: float = 1.0):
        return {"value": value * 2}


def register(ctx):
    return PluginRegistration(
        manifest=load_manifest(Path(__file__).with_name("plugin.toml")),
        controller_factory=E2EController,
        # None on purpose: exercises the host's fallback to
        # <package>/<manifest.ui.dist_dir>, which is what makes a bind-mounted
        # plugin resolve its own bundle the same way an installed one does.
        ui_dir=None,
    )
'''


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_imports():
    """Drop-in loading mutates sys.path and sys.modules; undo it per test."""
    path_before = list(sys.path)
    modules_before = set(sys.modules)
    yield
    sys.path[:] = path_before
    for name in set(sys.modules) - modules_before:
        del sys.modules[name]


class FakeDetectorsManager:
    def getAllDeviceNames(self):
        return ["WidefieldCamera"]


class FakeMaster:
    """Stub host. Enough for role resolution, nothing more."""
    detectorsManager = FakeDetectorsManager()


class FakeSetupInfo:
    def __init__(self, availableWidgets):
        self.availableWidgets = availableWidgets


def write_plugin(root, dirname, pkg_name, manifest=GOOD_MANIFEST, bundle=True):
    """Write a plugin into the drop-in root, laid out like `make dist` output."""
    pkg = root / dirname / pkg_name
    pkg.mkdir(parents=True)
    (pkg / "plugin.toml").write_text(manifest)
    (pkg / "__init__.py").write_text(PLUGIN_INIT)
    if bundle:
        dist = pkg / "ui" / "dist"
        dist.mkdir(parents=True)
        (dist / "remoteEntry.js").write_text(
            f"// federated bundle\nwindow.e2e_plugin = {{}}; // {MARKER}\n"
        )
    return pkg


@pytest.fixture
def plugin_root(tmp_path):
    root = tmp_path / "plugins"
    root.mkdir()
    return root


def build_host(plugin_root, monkeypatch, availableWidgets):
    """Discover plugins and mount them on a FastAPI app shaped like the real one.

    Returns (manager, TestClient). Mirrors ImSwitchServer.run() ordering:
    register routes on api_router → include_router → discover → attach_to_app
    → publish the manager.
    """
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    from imswitch.plugin_manager import PluginManager

    server = importlib.import_module(
        "imswitch.imcontrol.controller.server.ImSwitchServer")

    monkeypatch.setenv("IMSWITCH_PLUGIN_DIR", str(plugin_root))

    app = FastAPI(root_path="/imswitch")
    api_router = APIRouter(prefix="/api")
    server.register_plugin_routes(api_router)
    app.include_router(api_router)

    manager = PluginManager(
        master=FakeMaster(), setup_info=FakeSetupInfo(availableWidgets))
    manager.discover()
    manager.attach_to_app(app)

    server.set_plugin_manager(manager)
    monkeypatch.setattr(server, "_PLUGIN_MANAGER", manager, raising=False)

    return manager, TestClient(app)


@pytest.fixture(autouse=True)
def _clear_plugin_manager():
    yield
    importlib.import_module(
        "imswitch.imcontrol.controller.server.ImSwitchServer"
    ).set_plugin_manager(None)


# ── 1-3: the happy path, in the order the browser walks it ──────────────────
def test_manifest_lists_everything_the_frontend_consumes(plugin_root, monkeypatch):
    write_plugin(plugin_root, "e2e-plugin", "imswitch_plugin_e2e")
    _, client = build_host(plugin_root, monkeypatch, ["Settings", "e2e"])

    body = client.get("/imswitch/api/plugins").json()
    assert body["errors"] == []
    assert len(body["plugins"]) == 1

    entry = body["plugins"][0]
    assert entry["status"] == "loaded"
    # Exactly the keys usePluginWidgets()/makeRegistryEntryFromManifest() read.
    assert entry["name"] == "e2e"
    assert entry["display_name"] == "E2E Plugin"
    assert entry["version"] == "1.2.3"
    assert entry["scope"] == "e2e_plugin"
    assert entry["exposed"] == "./Widget"
    assert entry["remote_entry"] == "/imswitch/plugin/e2e/ui/remoteEntry.js"
    assert entry["api_base"] == "/imswitch/plugin/e2e/api"
    assert entry["socket_ns"] == "/plugin/e2e"
    assert entry["menu"] == {
        "label": "E2E", "icon": "Science", "group": "apps", "order": 7,
    }


def test_plugin_endpoints_respond(plugin_root, monkeypatch):
    write_plugin(plugin_root, "e2e-plugin", "imswitch_plugin_e2e")
    _, client = build_host(plugin_root, monkeypatch, ["e2e"])

    api_base = client.get("/imswitch/api/plugins").json()["plugins"][0]["api_base"]

    got = client.get(f"{api_base}/status")
    assert got.status_code == 200
    # Hardware resolved by role, all the way through to the controller.
    assert got.json() == {"ok": True, "camera": "WidefieldCamera"}

    posted = client.post(f"{api_base}/compute?value=21")
    assert posted.status_code == 200
    assert posted.json() == {"value": 42}


def test_bind_mounted_frontend_bundle_is_actually_served(plugin_root, monkeypatch):
    """The assertion that proves the whole delivery model.

    The bundle was never installed, never copied into the image and never built
    by the host — it came from a directory on disk, and the browser can fetch it
    same-origin at the URL the manifest advertises.
    """
    write_plugin(plugin_root, "e2e-plugin", "imswitch_plugin_e2e")
    _, client = build_host(plugin_root, monkeypatch, ["e2e"])

    remote_entry = client.get(
        "/imswitch/api/plugins").json()["plugins"][0]["remote_entry"]

    served = client.get(remote_entry)
    assert served.status_code == 200
    assert MARKER in served.text


# ── 4: the availableWidgets gate ────────────────────────────────────────────
def test_gated_plugin_is_disabled_unmounted_and_never_constructed(
        plugin_root, monkeypatch):
    pkg = write_plugin(plugin_root, "e2e-plugin", "imswitch_plugin_e2e")
    _, client = build_host(plugin_root, monkeypatch, ["Settings"])

    entry = client.get("/imswitch/api/plugins").json()["plugins"][0]
    assert entry["status"] == "disabled"
    assert "availableWidgets" in entry["reason"]
    assert entry["remote_entry"] is None
    assert entry["api_base"] is None

    # Its API is gone...
    assert client.get("/imswitch/plugin/e2e/api/status").status_code == 404
    # ...and its controller never ran, so it never claimed the camera.
    assert not (pkg / "constructed.marker").exists()


# ── 5: a broken plugin must not take a healthy one down ─────────────────────
def test_malformed_manifest_is_reported_and_the_other_plugin_still_loads(
        plugin_root, monkeypatch):
    write_plugin(
        plugin_root, "aaa-broken", "imswitch_plugin_broken",
        manifest="[plugin]\nname = \"broken\"\n# no version, no display_name, no ui\n",
    )
    write_plugin(plugin_root, "zzz-good", "imswitch_plugin_e2e")

    manager, client = build_host(plugin_root, monkeypatch, True)
    body = client.get("/imswitch/api/plugins").json()

    assert [p["name"] for p in body["plugins"]] == ["e2e"]
    assert body["plugins"][0]["status"] == "loaded"

    assert len(body["errors"]) == 1
    error = body["errors"][0]
    assert "aaa-broken" in error["source"]
    # Readable: names the failing field rather than just a traceback type.
    assert "display_name" in error["error"] or "ValidationError" in error["error"]


# ── 6: unmet hardware is an error, not a crash ──────────────────────────────
def test_unmet_required_hardware_is_reported_not_raised(plugin_root, monkeypatch):
    write_plugin(plugin_root, "e2e-plugin", "imswitch_plugin_e2e")

    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    from imswitch.plugin_manager import PluginManager

    server = importlib.import_module(
        "imswitch.imcontrol.controller.server.ImSwitchServer")
    monkeypatch.setenv("IMSWITCH_PLUGIN_DIR", str(plugin_root))

    class EmptyMaster:
        """A host with no detectors at all."""

    app = FastAPI(root_path="/imswitch")
    router = APIRouter(prefix="/api")
    server.register_plugin_routes(router)
    app.include_router(router)

    # discover() must not raise, even though the plugin's camera is required.
    manager = PluginManager(master=EmptyMaster(), setup_info=FakeSetupInfo(True))
    manager.discover()
    manager.attach_to_app(app)
    server.set_plugin_manager(manager)

    body = TestClient(app).get("/imswitch/api/plugins").json()
    assert body["plugins"] == []
    assert len(body["errors"]) == 1
    assert "detector:camera" in body["errors"][0]["error"]


# ── the plugin directory being absent is normal, not an error ───────────────
def test_no_plugin_directory_is_not_an_error(tmp_path, monkeypatch):
    _, client = build_host(tmp_path / "does-not-exist", monkeypatch, True)
    assert client.get("/imswitch/api/plugins").json() == {
        "plugins": [], "errors": []}
