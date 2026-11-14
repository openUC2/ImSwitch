"""HTTP access to the ImSwitch under test, and the setup it has loaded.

An endpoint is written "/Controller/method", as in ImSwitch's API documentation:

    client.get_json("/PositionerController/getPositionerPositions")
    client.get_json("/SettingsController/getCameraStatus", {"detectorName": "WidefieldCamera"})

What hardware the rig has is read from ImSwitch's own setup file:

    client.setup()["lasers"]        # {"LED": {"valueRangeMax": 1023, ...}, ...}

Only the server URL comes from the environment, so the suite can also run from a laptop.
"""
import functools
import os
import time

import pytest
import requests

IMSWITCH_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001").rstrip("/")


def get(endpoint, **kwargs):
    """``requests.get`` on an endpoint; returns the response whatever its status."""
    kwargs.setdefault("timeout", 30)
    return requests.get(f"{IMSWITCH_URL}/api{endpoint}", **kwargs)


def post(endpoint, **kwargs):
    """``requests.post`` on an endpoint; returns the response whatever its status."""
    kwargs.setdefault("timeout", 30)
    return requests.post(f"{IMSWITCH_URL}/api{endpoint}", **kwargs)


def json_of(response):
    """The JSON of a response; any status but 200 fails the test."""
    assert response.status_code == 200, (
        f"{response.request.method} {response.url} -> {response.status_code}: {response.text}")
    return response.json()


def get_json(endpoint, params=None, timeout=30):
    return json_of(get(endpoint, params=params, timeout=timeout))


def post_json(endpoint, params=None, json=None, timeout=30):
    return json_of(post(endpoint, params=params, json=json, timeout=timeout))


def reachable():
    try:
        return get("/version", timeout=10).status_code == 200
    except requests.RequestException:
        return False


@functools.lru_cache(maxsize=None)  # asked once per run; an error is not cached
def setup_file():
    """The name of the setup file ImSwitch runs with, as imcontrol_options.json names it.

    The file is ImSwitchConfig/imcontrol_setups/<name>; the name may also be a full path.
    """
    return get_json("/UC2ConfigController/getCurrentSetupFilename")["current_setup"]


@functools.lru_cache(maxsize=None)
def setup():
    """The content of setup_file(), as a dict.

    ImSwitch parses the file before it answers, so a field the file leaves out comes back with
    ImSwitch's default (e.g. "forAcquisition": false). The sections the tests use are
    "detectors", "lasers", "LEDMatrixs", "positioners" and "objective". A section the rig does
    not have is empty or None.
    """
    return get_json("/UC2ConfigController/readSetupFile", {"setupFileName": setup_file()})


def wait_until(check, timeout, interval=0.1):
    """Call ``check()`` until it returns something true and return that; None after ``timeout``."""
    deadline = time.monotonic() + timeout
    while True:
        result = check()
        if result or time.monotonic() >= deadline:
            return result or None
        time.sleep(interval)


def parametrize_from_rig(argnames, fetch, reason):
    """``pytest.mark.parametrize(argnames, fetch())`` for values the rig reports, e.g. its axes.

    pytest reads the values while it collects the tests, where an error would stop the whole
    run. So without a server, or with nothing to test, the test is skipped with ``reason``.
    """
    try:
        values = fetch()
    except (requests.RequestException, AssertionError) as error:
        values, reason = [], f"ImSwitch not reachable: {error}"
    if not values:
        nothing = [None] * len(argnames.split(","))
        values = [pytest.param(*nothing, id="none", marks=pytest.mark.skip(reason=reason))]
    return pytest.mark.parametrize(argnames, values)
