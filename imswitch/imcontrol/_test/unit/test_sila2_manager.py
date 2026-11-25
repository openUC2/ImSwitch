"""SiLa2Manager configuration: UUID, discovery, cloud endpoint (no server started)."""
import pytest

pytest.importorskip("unitelabs.cdk")

from imswitch.imcontrol.model.SetupInfo import SiLA2Info  # noqa: E402
from imswitch.imcontrol.model.managers.SiLa2Manager import SiLa2Manager  # noqa: E402


def test_uuid_is_stable_per_port_and_overridable():
    a, b = SiLa2Manager(SiLA2Info(serverPort=50052)), SiLa2Manager(SiLA2Info(serverPort=50052))
    assert a.server_uuid == b.server_uuid
    assert SiLa2Manager(SiLA2Info(serverPort=50053)).server_uuid != a.server_uuid
    fixed = "11111111-2222-4333-8444-555555555555"
    assert SiLa2Manager(SiLA2Info(serverUUID=fixed)).server_uuid == fixed


def test_connector_config():
    manager = SiLa2Manager(SiLA2Info(discovery=False))
    config = manager._connector_config()
    assert config.discovery is None
    assert not config.cloud_servers.enabled
    manager = SiLa2Manager(SiLA2Info(cloudEndpoint="abc.unitelabs.io"))
    config = manager._connector_config()
    assert config.cloud_servers.enabled
    assert config.cloud_servers.endpoints[0].hostname == "abc.unitelabs.io"
    assert config.sila_server.uuid == manager.server_uuid


def test_disabled_without_setup_block():
    manager = SiLa2Manager(None)
    assert not manager.is_enabled()
    assert manager.get_config() == {"enabled": False}
    manager.start_server()  # no-op
    assert not manager.is_running()


def test_workcell_lock_http_rule():
    from imswitch.imcontrol.model import workcell_lock as wl
    api = "/imswitch/api/PositionerController/"
    wl.set_lock_state(lambda: "SiLA client lock")
    try:
        assert wl.http_refusal(api + "movePositioner")                      # moves: refused
        assert wl.http_refusal("/imswitch/api/LaserController/setLaserValue")
        for open_path in (api + "getPositionerPositions", api + "stopAllAxes",
                          "/imswitch/api/SiLa2Controller/releaseWorkcellLock",
                          "/imswitch/api/FileManager/upload", "/imswitch/ui/index.html"):
            assert wl.http_refusal(open_path) == "", open_path
        token = wl.sila_call.set(True)   # a call made on the workcell's behalf
        assert wl.http_refusal(api + "movePositioner") == ""
        wl.sila_call.reset(token)
        wl.set_lock_state(lambda: "")
        assert wl.http_refusal(api + "movePositioner") == ""
    finally:
        wl.set_lock_state(None)
