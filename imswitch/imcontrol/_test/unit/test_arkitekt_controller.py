"""ArkitektController and ArkitektManager without hardware or a server.

Without the arkitekt package (the default test environment) the panel must
still load, say what to install and list what would be offered. With
arkitekt installed, the declared actions are also validated by rekuest's
definition builder and called locally against a fake microscope.
"""
import importlib
import types

import numpy as np
import pytest

from imswitch.imcontrol.controller.controllers import ArkitektController as controller_module
from imswitch.imcontrol.model.SetupInfo import ArkitektInfo

manager_module = importlib.import_module("imswitch.imcontrol.model.managers.ArkitektManager")


class Devices:
    def __init__(self, devices):
        self.devices = devices

    def getAllDeviceNames(self):
        return list(self.devices)

    def __getitem__(self, name):
        return self.devices[name]


class PositionerController:
    def __init__(self):
        self.pos = {"X": 100.0, "Y": 200.0, "Z": 50.0, "A": 0.0}
        self.calls = []

    def getPositionerPositions(self):
        return {"Stage": dict(self.pos)}

    def movePositioner(self, positionerName, axis, dist, isAbsolute, isBlocking, speed):
        self.calls.append(("move", axis, dist, isAbsolute, isBlocking))
        self.pos[axis] = dist if isAbsolute else self.pos[axis] + dist

    def movePositionerXYZ(self, positionerName, x, y, isAbsolute, isBlocking, speed):
        self.calls.append(("xy", x, y, isAbsolute, isBlocking))
        self.pos["X"], self.pos["Y"] = x, y


class Laser:
    valueRangeMin, valueRangeMax = 0, 1023

    def __init__(self):
        self.enabled, self.power = False, 10

    def setEnabled(self, on):
        self.enabled = on

    def setValue(self, value):
        self.power = value


class Detector:
    pixelSizeUm = [1, 0.5, 0.5]
    _running = True

    def __init__(self):
        self.n = 0

    def getLatestFrame(self, returnFrameNumber=False):
        self.n += 1
        frame = np.full((60, 80), self.n, dtype=np.uint16)
        return (frame, self.n) if returnFrameNumber else frame

    def getParameter(self, name):
        return 10.0


@pytest.fixture
def no_arkitekt(monkeypatch):
    real = manager_module.importlib.util.find_spec
    monkeypatch.setattr(manager_module.importlib.util, "find_spec",
                        lambda name, *a: None if name in ("arkitekt", "rekuest") else real(name))


@pytest.fixture
def saved(monkeypatch):
    calls = []
    monkeypatch.setattr(controller_module.configfiletools, "saveSetupInfo",
                        lambda options, info: calls.append(info))
    monkeypatch.setattr(controller_module.configfiletools, "loadOptions", lambda: (None, False))
    return calls


class Stage:
    """The stage manager: its cached position is what moves keep current."""
    axes = ["X", "Y", "Z"]

    def __init__(self, controller):
        self._controller = controller

    @property
    def position(self):
        return dict(self._controller.pos)

    def getPosition(self):
        raise AssertionError("a serial round trip: only get_stage_position may ask the device")


def make_controller(info=None, lasers=("LED", "Laser 488"), stage=True, camera=True,
                    controllers=None):
    positioner = PositionerController()
    registry = {"Positioner": positioner, **(controllers or {})}
    master = types.SimpleNamespace(
        positionersManager=Devices({"Stage": Stage(positioner)} if stage else {}),
        lasersManager=Devices({name: Laser() for name in lasers}),
        detectorsManager=Devices({"Cam": Detector()} if camera else {}),
        getController=registry.get)
    master.arkitektManager = manager_module.ArkitektManager(
        info or ArkitektInfo(autoConnect=False))
    setup = types.SimpleNamespace(arkitekt=None)
    controller = controller_module.ArkitektController(setup, None, master, widget=None,
                                                      factory=None, moduleCommChannel=None)
    return controller, positioner, master


# ── without the arkitekt package ────────────────────────────────────────────

def test_without_the_package_the_panel_says_what_to_install(no_arkitekt, saved):
    controller, _, _ = make_controller()
    status = controller.getArkitektStatus()
    assert status["state"] == "unavailable" and not status["available"]
    assert "arkitekt[rekuest,mikro]" in status["message"]
    assert not status["hasStoredLogin"]  # no package, no session to look for
    refused = controller.bindArkitekt(url="http://nas.local")
    assert refused["status"] == "error" and "not installed" in refused["message"]
    assert saved[-1].arkitekt.url == "http://nas.local"  # the choice is kept anyway


def test_the_setup_gets_its_arkitekt_block_so_panel_changes_are_saved(no_arkitekt, saved):
    controller, _, _ = make_controller()
    assert controller._setupInfo.arkitekt is controller._manager.info
    result = controller.setArkitektSettings(appName="FRAME Fork Approval", useMikro=False,
                                            allowInsecureTransport=True)
    assert result["status"] == "success" and result["appName"] == "frame-fork-approval"
    assert saved[-1].arkitekt.useMikro is False
    assert saved[-1].arkitekt.allowInsecureTransport is True


def test_disabled_in_the_setup(no_arkitekt):
    manager = manager_module.ArkitektManager(ArkitektInfo(enabled=False))
    assert manager.status()["enabled"] is False


@pytest.mark.parametrize("kwargs, expected", [
    ({}, {"get_stage_position", "move_stage", "go_to_xy", "home_axis",
          "move_to_sample_loading_position", "set_illumination", "set_camera",
          "acquire_frame", "run_tile_scan"}),
    ({"lasers": ()}, {"get_stage_position", "move_stage", "go_to_xy", "home_axis",
                      "move_to_sample_loading_position", "set_camera", "acquire_frame",
                      "run_tile_scan"}),
    ({"stage": False}, {"set_illumination", "set_camera", "acquire_frame"}),
    ({"info": ArkitektInfo(useMikro=False, autoConnect=False)},
     {"get_stage_position", "move_stage", "go_to_xy", "home_axis",
      "move_to_sample_loading_position", "set_illumination", "set_camera"}),
])
def test_offered_actions_follow_the_setup(no_arkitekt, kwargs, expected):
    controller, _, _ = make_controller(**kwargs)
    actions = controller.getArkitektStatus()["actions"]
    assert {a["name"] for a in actions} == expected
    assert all(a["moves"] for a in actions if a["name"] in ("move_stage", "run_tile_scan"))


def test_device_names_become_enum_choices():
    choices = controller_module._choices("Illumination", ["LED", "Laser 488", "488", "LED!"])
    assert [m.name for m in choices] == ["LED", "LASER_488", "_488", "LED_"]
    assert [m.value for m in choices] == ["LED", "Laser 488", "488", "LED!"]


def test_remote_calls_are_logged_and_generators_keep_streaming(no_arkitekt):
    controller, _, _ = make_controller()

    def scan(n):
        yield from range(n)

    def broken():
        raise ValueError("boom")

    assert list(controller._tracked(scan)(n=3)) == [0, 1, 2]
    with pytest.raises(ValueError):
        controller._tracked(broken)()
    newest, oldest = controller.getArkitektStatus()["activity"]
    assert oldest["action"] == "scan" and oldest["results"] == 3 and oldest["status"] == "done"
    assert oldest["arguments"] == {"n": 3}
    assert newest["status"] == "failed" and newest["error"] == "ValueError: boom"
    assert controller.clearArkitektActivity()["status"] == "success"
    assert controller.getArkitektStatus()["activity"] == []


def test_busy_microscope_refuses_remote_calls(no_arkitekt):
    experiment = types.SimpleNamespace(getExperimentStatus=lambda: {"status": "running"})
    controller, _, _ = make_controller(controllers={"Experiment": experiment})
    with pytest.raises(RuntimeError, match="experiment is running"):
        controller._refuse_if_busy()


def test_tile_grid_is_a_snake_around_the_current_position(no_arkitekt):
    controller, _, _ = make_controller()
    grid = controller._tile_grid(60.0, 30.0, None, None, 50.0, None, None)
    # 80 x 60 px at 0.5 µm = 40 x 30 µm; 50 % overlap -> 20 x 15 µm steps
    assert (grid["nx"], grid["ny"], grid["step"]) == (4, 3, (20.0, 15.0))
    xs = [x for _, _, x, _ in grid["tiles"]]
    assert xs[:4] == [70.0, 90.0, 110.0, 130.0] and xs[4:8] == [130.0, 110.0, 90.0, 70.0]
    assert grid["tiles"][0][3] == 185.0 and grid["start"]["x"] == 100.0


def test_thumbnail_is_a_small_jpeg():
    pytest.importorskip("cv2")
    url = controller_module._thumbnail(np.random.randint(0, 4096, (600, 800), np.uint16))
    assert url.startswith("data:image/jpeg;base64,")


def test_illumination_outside_its_range_is_refused(no_arkitekt):
    controller, _, master = make_controller()
    with pytest.raises(ValueError, match="outside its range"):
        controller._set_illumination("LED", True, 5000)
    controller._set_illumination("LED", True, 512)
    assert master.lasersManager["LED"].enabled and master.lasersManager["LED"].power == 512


# ── with the arkitekt package ───────────────────────────────────────────────

def test_declarations_validate_against_rekuest():
    pytest.importorskip("arkitekt")
    rekuest = pytest.importorskip("rekuest.arkitekt")
    controller, _, _ = make_controller(info=ArkitektInfo(useMikro=False, autoConnect=False))
    app = controller.build_app("imswitch", "2.0.0")
    implementations = app.snapshot(provider=rekuest.rekuest_provider).registry.implementations
    assert "move_stage" in implementations and "acquire_frame" not in implementations
    home = next(p for p in implementations["home_axis"].definition.args if p.key == "axis")
    assert [c.value for c in home.choices] == ["X", "Y"]


def test_every_type_identifier_is_package_slash_key():
    """The server refuses the agent's registration otherwise (MODEL ports
    defaulted to a bare 'stage_position')."""
    import re
    pytest.importorskip("arkitekt")
    rekuest = pytest.importorskip("rekuest.arkitekt")
    pytest.importorskip("mikro")
    controller, _, _ = make_controller()
    app = controller.build_app("imswitch", "2.0.0")
    registry = app.snapshot(provider=rekuest.rekuest_provider).registry

    def ports(items):
        for port in items or ():
            yield port
            yield from ports(port.children)

    identifiers = {(name, port.key, port.identifier)
                   for name, implementation in registry.implementations.items()
                   for port in ports([*implementation.definition.args,
                                      *implementation.definition.returns])
                   if port.identifier is not None}
    assert ("get_stage_position", "return0", "@imswitch/stage_position") in identifiers
    assert all(re.fullmatch(r"@[^/\s]+/[^/\s]+", i) for _, _, i in identifiers), identifiers


def test_state_publishing_never_reads_the_device(no_arkitekt):
    controller, positioner, _ = make_controller()
    positioner.pos["X"] = 42.0
    assert controller._state_snapshot()["x_um"] == 42.0  # Stage.getPosition would raise
    assert controller._position(None, fresh=True)["x"] == 42.0  # via PositionerController


class FakeMikro:
    """create_array_dataset and create_coordinate_system, recorded."""

    def __init__(self):
        self.datasets, self.spaces = [], []

    def create_array_dataset(self, **kwargs):
        dataset = types.SimpleNamespace(id=f"ds{len(self.datasets)}", kwargs=kwargs,
                                        lens=lambda: types.SimpleNamespace(id="lens"))
        self.datasets.append(dataset)
        return dataset

    def create_coordinate_system(self, name, axes, registrations, epoch):
        space = types.SimpleNamespace(name=name, registered=[], staged=[])
        space.register = lambda dataset, scale=None, **offsets: space.registered.append(
            (dataset.id, scale, offsets))
        space.stage = lambda name=None: space.staged.append(name)
        self.spaces.append(space)
        return space


def test_tile_scan_places_tiles_in_one_space_and_restores_everything():
    pytest.importorskip("arkitekt")
    pytest.importorskip("mikro")
    from arkitekt import Task
    controller, positioner, master = make_controller()
    controller.build_app("imswitch", "2.0.0")
    led = next(m for m in controller._actions["set_illumination"].__wrapped__
               .__annotations__["channel"] if m.value == "LED")
    mikro = FakeMikro()
    tiles = list(controller._actions["run_tile_scan"](
        mikro=mikro, task=Task.local(), range_x_um=60.0, range_y_um=30.0, overlap_percent=50,
        illumination=led, intensity=300, settle_s=0))
    assert len(tiles) == 12 and len(mikro.datasets) == 12
    xs = [offsets["x"] for _, _, offsets in mikro.spaces[0].registered]
    assert xs[:4] == [70.0, 90.0, 110.0, 130.0] and xs[4:8] == [130.0, 110.0, 90.0, 70.0]
    assert mikro.spaces[0].registered[0][1] == {"y": 0.5, "x": 0.5} and mikro.spaces[0].staged
    assert mikro.datasets[0].kwargs["axes"] == ["c", "y", "x"]
    assert positioner.calls[-1][:3] == ("xy", 100.0, 200.0)  # back where it started
    led_device = master.lasersManager["LED"]
    assert led_device.enabled is False and led_device.power == 10  # restored


class FakeFakts:
    """/.well-known/fakts, device authorization and token endpoints on 127.0.0.1."""

    def __init__(self, token_error="authorization_pending"):
        import asyncio
        import socket
        import threading
        from aiohttp import web

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.base = f"http://127.0.0.1:{self.port}"
        base, ready = self.base, threading.Event()

        async def well_known(request):
            return web.json_response({
                "name": "Test NAS", "base_url": f"{base}/f/", "protocol_version": "2",
                "configure": f"{base}/f/configure/{{code}}", "token_endpoint": f"{base}/f/token/",
                "device_authorization_endpoint": f"{base}/f/device/"})

        async def device(request):
            return web.json_response({"device_code": "dev-1", "user_code": "ABCD-1234",
                                      "client_id": "client-1", "expires_in": 600, "interval": 1})

        async def token(request):
            return web.json_response({"error": token_error}, status=400)

        def serve():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app = web.Application()
            app.router.add_get("/.well-known/fakts", well_known)
            app.router.add_post("/f/device/", device)
            app.router.add_post("/f/token/", token)
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", self.port).start())
            ready.set()
            loop.run_forever()

        threading.Thread(target=serve, daemon=True).start()
        ready.wait(5)


@pytest.fixture
def private_state_dir(tmp_path, monkeypatch):
    """Sessions go to tmp_path, never the user's arkitekt state directory."""
    pytest.importorskip("arkitekt")
    import importlib
    import platformdirs
    private = lambda *a, **k: str(tmp_path)  # noqa: E731
    monkeypatch.setattr(platformdirs, "user_state_dir", private)
    # arkitekt does `from platformdirs import user_state_dir`, so patching
    # platformdirs alone does not reach it: patch the module that binds it.
    # That module moved between releases (5.x: app.fakts, 6.x: app.sessions),
    # and CI resolves "arkitekt>=5.0.1" to whatever is newest.
    for name in ("arkitekt.app.fakts", "arkitekt.app.sessions"):
        module = importlib.import_module(name)
        if hasattr(module, "user_state_dir"):
            monkeypatch.setattr(module, "user_state_dir", private)
    # Fail loudly rather than write test logins into the real state directory
    # if a future arkitekt keeps the path somewhere this fixture does not reach.
    from arkitekt.app.fakts import _cache_path
    probe = _cache_path(types.SimpleNamespace(identifier="probe", version="0"), "http://probe")
    assert str(probe).startswith(str(tmp_path)), (
        f"arkitekt still resolves sessions outside the test dir: {probe}")
    monkeypatch.setenv("ARKITEKT_DEVICE_ID", "test-microscope")


def wait_for(manager, states, timeout=15):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if manager.status()["state"] in states:
            return manager.status()
        time.sleep(0.05)
    raise AssertionError(f"still {manager.status()['state']}: {manager.status().get('message')}")


def test_stored_login_path_matches_arkitekt(private_state_dir):
    from arkitekt.app.fakts import _cache_path
    url = "http://nas.local"
    assert manager_module.cache_file("imswitch", "2.0.0", url) == _cache_path(
        types.SimpleNamespace(identifier="imswitch", version="2.0.0"), url)


def test_bind_shows_the_device_code_and_cancel_stops_the_login(private_state_dir, saved):
    server = FakeFakts()
    controller, _, _ = make_controller()
    assert controller.bindArkitekt(url=server.base)["status"] == "started"
    status = wait_for(controller._manager, {"awaiting_login", "error"})
    assert status["state"] == "awaiting_login", status["message"]
    assert status["userCode"] == "ABCD-1234" and status["serverName"] == "Test NAS"
    assert status["approveUrl"] == f"{server.base}/f/configure/ABCD-1234"
    assert controller.bindArkitekt()["status"] == "error"  # one connection at a time
    assert controller.cancelArkitekt()["state"] == "unbound"
    assert not controller._manager._thread.is_alive()


def test_expired_code_is_explained(private_state_dir, saved):
    server = FakeFakts(token_error="expired_token")
    controller, _, _ = make_controller()
    controller.bindArkitekt(url=server.base)
    assert "expired" in wait_for(controller._manager, {"error"})["message"]


def test_automatic_reconnect_with_an_invalid_stored_login_does_not_prompt(private_state_dir):
    import json
    import os
    server = FakeFakts()
    path = manager_module.cache_file("imswitch", "2.0.0", server.base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({}, f)  # a session fakts cannot use: it would fall back to a device code
    controller, _, _ = make_controller(info=ArkitektInfo(url=server.base, autoConnect=True))
    status = wait_for(controller._manager, {"error", "awaiting_login"})
    assert status["state"] == "error" and "no longer valid" in status["message"]
    assert controller._manager._retry is None  # no retry loop piling up codes
    assert controller.unbindArkitekt()["hasStoredLogin"] is False


def test_only_http_approval_links_reach_the_panel(no_arkitekt):
    import asyncio
    manager = manager_module.ArkitektManager(ArkitektInfo(autoConnect=False))
    endpoint = types.SimpleNamespace(configure="javascript:alert('{code}')", base_url="x",
                                     name="evil")
    asyncio.run(manager._on_device_code(endpoint, "ABCD"))
    assert manager.status()["userCode"] == "ABCD" and manager.status()["approveUrl"] is None
    endpoint.configure = "https://lok.example/configure/{code}"
    asyncio.run(manager._on_device_code(endpoint, "ABCD"))
    assert manager.status()["approveUrl"] == "https://lok.example/configure/ABCD"


def test_device_code_hook_takes_the_fakts_5_4_challenge(no_arkitekt):
    """fakts 5.4+ (arkitekt 6) calls the hook with one DeviceCodeChallenge, not
    (endpoint, code): the Bind button failed with a TypeError before this."""
    import asyncio
    manager = manager_module.ArkitektManager(ArkitektInfo(autoConnect=False))
    endpoint = types.SimpleNamespace(configure="https://lok.example/configure/{code}",
                                     base_url="https://lok.example", name="Lok")

    def challenge(link):
        return types.SimpleNamespace(endpoint=endpoint, user_code="WXYZ-1234",
                                     verification_uri_complete=link, expires_in=600)

    asyncio.run(manager._on_device_code(challenge("https://lok.example/approve?c=WXYZ-1234")))
    status = manager.status()
    assert status["userCode"] == "WXYZ-1234" and status["serverName"] == "Lok"
    assert status["approveUrl"] == "https://lok.example/approve?c=WXYZ-1234"

    # No ready-made link from the server: fall back to the endpoint's template.
    asyncio.run(manager._on_device_code(challenge("")))
    assert manager.status()["approveUrl"] == "https://lok.example/configure/WXYZ-1234"

    # The panel renders the link, so a non-http one never reaches it.
    asyncio.run(manager._on_device_code(challenge("javascript:alert(1)")))
    assert manager.status()["approveUrl"] is None


def test_frame_grab_without_frame_numbers(no_arkitekt):
    controller, _, _ = make_controller()

    class Plain:
        _running = True

        def getLatestFrame(self):
            return np.ones((4, 4))

        def getParameter(self, name):
            return 1.0

    controller.mDetector = Plain()
    assert controller.grabCameraFrame(frameSync=1).shape == (4, 4)
