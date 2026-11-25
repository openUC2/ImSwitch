"""SiLA 2: the check client (sila2/check.py) against the virtual microscope.

Starts ImSwitch with the virtual setup plus a ``sila2`` block and runs every step of the
check, motion and light included (all virtual). Run this file on its own: the API test
server is one per session, and the other API tests start it without SiLA.

    pytest imswitch/imcontrol/_test/api/test_sila2.py -p no:arkitekt_next
"""
import asyncio
import json
import socket
import time
from pathlib import Path

import pytest

pytest.importorskip("unitelabs.cdk")

from sila.client import Client, ClientConfig  # noqa: E402

from . import get_test_server  # noqa: E402

VIRTUAL = (Path(__file__).parents[3] / "_data" / "user_defaults" / "imcontrol_setups"
           / "example_virtual_microscope.json")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"no SiLA server on port {port} after {timeout} s")


@pytest.fixture(scope="module")
def sila_port(tmp_path_factory):
    setup = json.loads(VIRTUAL.read_text())
    setup["availableWidgets"].append("SiLa2")
    port = _free_port()
    setup["sila2"] = {"serverPort": port, "discovery": False}
    path = tmp_path_factory.mktemp("sila") / "virtual_sila.json"
    path.write_text(json.dumps(setup))
    server = get_test_server(str(path))
    if server.config_file != str(path):
        pytest.skip("the API test server already runs without SiLA; run this file alone")
    server.start()
    _wait_for_port(port, 60)
    return server, port


def _client(port):
    return Client(ClientConfig(hostname="127.0.0.1", port=port, tls=False,
                               default_timeout=60_000))


def test_every_sila_endpoint(sila_port):
    # Imported after the server started: imswitch.imcontrol.controller imports ImSwitchServer,
    # which reads the HTTP port and SSL flag at import time.
    from imswitch.imcontrol.controller.controllers.sila2.check import run_check

    async def run():
        async with _client(sila_port[1]) as client:
            return await run_check(client, motion=True, light=True, teach=True, intensity=10)

    check = asyncio.run(run())
    failed = [r for r in check.results if r[1] == "fail"]
    assert not failed, failed
    skipped = {r[0] for r in check.results if r[1] == "skip"}
    assert skipped == {"StageController.Home"}  # end switches: only with --home


def test_lock_keeps_the_ui_out(sila_port):
    """While a SiLA client holds the lock, HTTP may read and stop but not move (423)."""
    server, port = sila_port
    api = "/imswitch/api/PositionerController"

    async def locked(fn):
        async with _client(port) as client:
            await client.LockController.LockServer(LockIdentifier="test", Timeout=0)
            try:
                return await asyncio.to_thread(fn)
            finally:
                await client.LockController.UnlockServer(LockIdentifier="test")

    def http_calls():
        move = server.get(f"{api}/movePositioner", params={"axis": "X", "dist": 1})
        read = server.get(f"{api}/getPositionerPositions")
        stop = server.get(f"{api}/stopAllAxes")
        status = server.get("/imswitch/api/SiLa2Controller/getSiLA2Status").json()
        return move.status_code, read.status_code, stop.status_code, status["locked"]

    assert asyncio.run(locked(http_calls)) == (423, 200, 200, True)
    after = server.get(f"{api}/movePositioner", params={"axis": "X", "dist": 0})
    assert after.status_code == 200
