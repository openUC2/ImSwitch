"""Forklift pallet upgrade (UC2ConfigController): the picked version must never reach the host
unit name or forklift unless it is a plain branch, tag or commit name."""
import logging

import pytest

from imswitch.imcommon.model import dirtools
from imswitch.imcontrol.controller.controllers.UC2ConfigController import (
    PALLET_UPGRADE_UNIT, UC2ConfigController, palletUpgradeUnit, readPalletVersioning)


@pytest.mark.parametrize("version, unit", [
    ("", PALLET_UPGRADE_UNIT),
    ("edge", "imswitch-pallet-upgrade@edge.service"),
    ("v26.1.0-alpha.1", r"imswitch-pallet-upgrade@v26.1.0\x2dalpha.1.service"),
    # same as `systemd-escape beniroquai-patch-1` / `systemd-escape feat/arkitekt-hub`
    ("beniroquai-patch-1", r"imswitch-pallet-upgrade@beniroquai\x2dpatch\x2d1.service"),
    ("feat/arkitekt-hub", r"imswitch-pallet-upgrade@feat-arkitekt\x2dhub.service"),
    ("d9f6311", "imswitch-pallet-upgrade@d9f6311.service"),
])
def test_unit_for_version(version, unit):
    assert palletUpgradeUnit(version) == unit


@pytest.mark.parametrize("version", [
    "edge; reboot", "$(reboot)", "`id`", "edge'", 'edge"', "edge x", "-edge", "/edge", ".edge",
    "a..b", "edge\n", "@edge", "github.com/openUC2/os-rpi@edge", "a" * 101, "é",
])
def test_bad_versions_are_refused(version):
    with pytest.raises(ValueError):
        palletUpgradeUnit(version)


def test_versioning_file(tmp_path):
    f = tmp_path / "forklift.yml"
    f.write_text(
        'forklift: "0.9.0-alpha.1"\n'
        'current: "github.com/openUC2/os-rpi@v0.0.0-20261101-abc (branch: beniroquai-patch-1)"\n'
        'pallet: "github.com/openUC2/os-rpi@v26.0.0 + 20261101-abc"\n'
        "changes: |2\n  M x\n"
        'upgrade-source: "github.com/openUC2/os-rpi@beniroquai-patch-1"\n'
        'warnings:\n  - "something"\n')
    v = readPalletVersioning(str(f))
    assert v["upgrade-source"] == "github.com/openUC2/os-rpi@beniroquai-patch-1"
    assert v["current"].endswith("(branch: beniroquai-patch-1)")
    assert "changes" not in v and "warnings" not in v
    assert readPalletVersioning(str(tmp_path / "missing.yml")) == {}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(dirtools.UserFileDirs, "Root", str(tmp_path))
    c = object.__new__(UC2ConfigController)
    c._logger = logging.getLogger("test")
    c.calls = []

    def callSystemd(method, signature, body, **kwargs):
        c.calls.append((method, body))
        if method == "LoadUnit":
            return ("/unit/path",)
        if method == "Get":
            return (("s", "activating"),)
        return ()
    c._callSystemd = callSystemd
    return c


def test_start_refuses_bad_version_without_systemd(controller):
    assert controller.startPalletUpgrade("edge && reboot")["status"] == "error"
    assert controller.calls == []


def test_start_version_and_status_follow_the_unit(controller):
    assert controller.getPalletUpgradeStatus()["unit"] == PALLET_UPGRADE_UNIT
    res = controller.startPalletUpgrade(" beniroquai-patch-1 ")
    unit = r"imswitch-pallet-upgrade@beniroquai\x2dpatch\x2d1.service"
    assert res == {"status": "started", "unit": unit}
    assert controller.calls[-1] == ("StartUnit", (unit, "replace"))
    status = controller.getPalletUpgradeStatus()
    assert status["unit"] == unit and status["state"] == "activating"
    assert ("LoadUnit", (unit,)) in controller.calls
