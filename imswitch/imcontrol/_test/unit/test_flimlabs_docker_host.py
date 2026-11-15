"""FLIM server on the Docker host: localhost falls back to host.docker.internal."""
from unittest.mock import MagicMock

import pytest

from imswitch.imcontrol.model.interfaces.flimlabsclient import FLIMLabsClient
from imswitch.imcontrol.model.managers.detectors.FLIMLabsDetectorManager import FLIMLabsDetectorManager


def make(monkeypatch, reachable, host="localhost"):
    monkeypatch.setattr(FLIMLabsClient, "health", lambda self: self.host in reachable)
    monkeypatch.setattr(FLIMLabsClient, "check_card", lambda self, use_cache=False: None)
    info = MagicMock()
    info.managerProperties = {"host": host, "port": 5249}
    return FLIMLabsDetectorManager(info, "FLIM").client


@pytest.mark.parametrize("reachable, expected", [
    ({"localhost"}, "http://localhost:5249"),
    ({"host.docker.internal"}, "http://host.docker.internal:5249"),
    (set(), "http://localhost:5249"),   # nothing answers: keep what the setup says
])
def test_localhost_falls_back_to_docker_host(monkeypatch, reachable, expected):
    assert make(monkeypatch, reachable).base_url == expected


def test_explicit_host_is_never_replaced(monkeypatch):
    assert make(monkeypatch, {"host.docker.internal"}, host="10.0.0.5").base_url == "http://10.0.0.5:5249"
