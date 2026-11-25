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
