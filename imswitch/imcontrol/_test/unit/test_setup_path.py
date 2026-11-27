"""A setup path relative to the working directory loads that file, not a same-named copy."""
import types

from imswitch.imcontrol.model import configfiletools


class _Setup:
    @classmethod
    def from_json(cls, text, infer_missing=True):
        obj = cls()
        obj.text = text
        return obj


def test_relative_config_file_is_the_one_in_the_working_directory(tmp_path, monkeypatch):
    setups = tmp_path / "ImSwitchConfig" / "imcontrol_setups"
    setups.mkdir(parents=True)
    (setups / "scope.json").write_text('{"which": "stale copy in the setups folder"}')
    work = tmp_path / "repo"
    (work / "setups").mkdir(parents=True)
    (work / "setups" / "scope.json").write_text('{"which": "the file that was asked for"}')
    monkeypatch.setattr(configfiletools, "_setupFilesDir", str(setups))
    monkeypatch.chdir(work)
    loaded = configfiletools.loadSetupInfo(
        types.SimpleNamespace(setupFileName="setups/scope.json"), _Setup)
    assert "asked for" in loaded.text
    # a bare name still means the setups folder
    loaded = configfiletools.loadSetupInfo(types.SimpleNamespace(setupFileName="scope.json"), _Setup)
    assert "stale copy" in loaded.text


def test_a_port_another_server_listens_on_counts_as_in_use():
    # A second ImSwitch on a busy port opened the board's serial port before
    # failing to bind: startup now refuses when the port is taken.
    import socket

    from imswitch.__main__ import _port_in_use
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert _port_in_use(port)
    finally:
        server.close()
    assert not _port_in_use(port)
