"""Tests for the v2 plugin manifest HTTP API.

Covers the wiring added in WP1: ``GET /imswitch/api/plugins`` (and the
single-plugin variant) must be reachable, must never 500, and must return
exactly the payload ``usePluginWidgets()`` in ``frontend/src/App.jsx`` consumes.

Hermetic: no uvicorn, no hardware, no plugin on disk. The PluginManager is
populated directly with a ``LoadedPlugin`` record so ``manifest_list()`` — the
real one — produces the response body under test.

Every ImSwitch import in this module is deliberately *lazy*. Importing
``ImSwitchServer`` mounts Socket.IO and the static dirs onto its module-global
FastAPI app as a side effect; doing that at collection time breaks the
session-scoped real-server fixture in ``_test/api/`` that boots ImSwitch via
``imswitch.__main__.main()``. Keep these imports inside functions.
"""
import collections
import importlib

import pytest

pytest.importorskip("fastapi")


PLUGIN_NAME = "demo"

# Keys the frontend reads off each manifest entry. Frozen surface — see
# docs/plugins/DECISIONS.md.
FRONTEND_KEYS = {
    "name", "display_name", "version", "status", "menu",
    "remote_entry", "scope", "exposed", "api_base", "socket_ns",
}


def _server_module():
    """The ImSwitchServer *module* (not the class the package re-exports)."""
    return importlib.import_module(
        "imswitch.imcontrol.controller.server.ImSwitchServer")


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def server():
    """The module under test, with its global plugin-manager holder cleared
    before and after so tests stay independent."""
    module = _server_module()
    module.set_plugin_manager(None)
    yield module
    module.set_plugin_manager(None)


@pytest.fixture
def client(server):
    """App mounted exactly the way ImSwitchServer mounts the real one, so the
    URL under test is literally the one the browser requests."""
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI(root_path="/imswitch")
    router = APIRouter(prefix="/api")
    server.register_plugin_routes(router)
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def loaded_manager(tmp_path):
    """A PluginManager holding one fully-formed plugin record."""
    from fastapi import APIRouter

    from imswitch.plugin_manager import LoadedPlugin, PluginManager
    from imswitch.plugin_sdk import PluginManifest

    ui_dir = tmp_path / "ui" / "dist"
    ui_dir.mkdir(parents=True)
    (ui_dir / "remoteEntry.js").write_text("// stub")

    manifest = PluginManifest(
        name=PLUGIN_NAME,
        display_name="Demo Plugin",
        version="0.1.0",
        ui={"scope": "demo_plugin", "menu_label": "Demo", "menu_icon": "ScienceIcon"},
    )
    manager = PluginManager(master=None, setup_info=None)
    manager._plugins[PLUGIN_NAME] = LoadedPlugin(
        manifest=manifest,
        controller=object(),
        router=APIRouter(),
        ui_dir=ui_dir,
        mount=f"/plugin/{PLUGIN_NAME}",
        source=f"dropin:/opt/imswitch/plugins/{PLUGIN_NAME}",
    )
    return manager


class _BrokenManager:
    """Stand-in for a PluginManager that blows up while serialising."""

    def manifest_list(self):
        raise RuntimeError("manifest serialisation exploded")

    def errors(self):
        return []


# ── the route exists and degrades gracefully ────────────────────────────────
def test_plugins_route_exists_with_no_manager(client):
    """Clean install: discovery has not run. Must be 200, not 404 and not 500."""
    response = client.get("/imswitch/api/plugins")
    assert response.status_code == 200
    assert response.json() == {"plugins": [], "errors": []}


def test_plugins_route_reachable_without_root_path_prefix(client):
    """Starlette strips root_path, so both spellings resolve to the same route."""
    assert client.get("/api/plugins").status_code == 200


def test_plugins_route_never_500s(server, client):
    """A manager that raises must degrade to an error entry, not a 500 — the
    frontend has to be able to render "no plugins" instead of breaking."""
    server.set_plugin_manager(_BrokenManager())
    response = client.get("/imswitch/api/plugins")
    assert response.status_code == 200
    body = response.json()
    assert body["plugins"] == []
    assert len(body["errors"]) == 1
    assert "manifest serialisation exploded" in body["errors"][0]["error"]


# ── the route reports a loaded plugin ───────────────────────────────────────
def test_plugins_route_lists_loaded_plugin(server, client, loaded_manager):
    server.set_plugin_manager(loaded_manager)
    body = client.get("/imswitch/api/plugins").json()

    assert body["errors"] == []
    assert len(body["plugins"]) == 1
    entry = body["plugins"][0]

    assert FRONTEND_KEYS <= set(entry)
    assert entry["name"] == PLUGIN_NAME
    assert entry["display_name"] == "Demo Plugin"
    assert entry["version"] == "0.1.0"
    # WP2 replaces this with "loaded" / "disabled" / "error".
    assert entry["status"] == "ok"
    assert entry["scope"] == "demo_plugin"
    assert entry["exposed"] == "./Widget"
    assert entry["remote_entry"] == f"/plugin/{PLUGIN_NAME}/ui/remoteEntry.js"
    assert entry["api_base"] == f"/plugin/{PLUGIN_NAME}/api"
    assert entry["socket_ns"] == f"/plugin/{PLUGIN_NAME}"
    assert entry["menu"]["label"] == "Demo"
    assert entry["menu"]["icon"] == "ScienceIcon"


def test_plugins_route_reports_discovery_errors(server, client, loaded_manager):
    loaded_manager._record_error("dropin:/opt/imswitch/plugins/bad",
                                 ValueError("no register() function"))
    server.set_plugin_manager(loaded_manager)
    body = client.get("/imswitch/api/plugins").json()

    assert len(body["plugins"]) == 1
    assert body["errors"] == [{
        "source": "dropin:/opt/imswitch/plugins/bad",
        "error": "ValueError: no register() function",
    }]


# ── single-plugin route ─────────────────────────────────────────────────────
def test_single_plugin_route_returns_manifest(server, client, loaded_manager):
    server.set_plugin_manager(loaded_manager)
    response = client.get(f"/imswitch/api/plugins/{PLUGIN_NAME}")
    assert response.status_code == 200
    assert response.json()["name"] == PLUGIN_NAME


def test_single_plugin_route_404s_for_unknown_name(server, client, loaded_manager):
    server.set_plugin_manager(loaded_manager)
    assert client.get("/imswitch/api/plugins/nope").status_code == 404


def test_single_plugin_route_404s_with_no_manager(client):
    assert client.get(f"/imswitch/api/plugins/{PLUGIN_NAME}").status_code == 404


# ── registration order (the B3 regression guard) ────────────────────────────
def test_create_api_registers_plugin_routes_on_api_router(server):
    """The routes must land on `api_router` inside createAPI(), i.e. before
    ImSwitchServer.run() calls app.include_router(api_router). Registering them
    any later means FastAPI never sees them.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    empty_api = collections.namedtuple("Api", [])()
    srv = server.ImSwitchServer(empty_api, setupInfo=None, master=None)
    srv.createAPI()

    app = FastAPI(root_path="/imswitch")
    app.include_router(server.api_router)

    response = TestClient(app).get("/imswitch/api/plugins")
    assert response.status_code == 200
    assert response.json() == {"plugins": [], "errors": []}
