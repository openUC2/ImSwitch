"""Check that every configured laser/LED switches through the ImSwitch API."""

import os

import pytest
import requests


# ImSwitch runs on :8001 without the caddy prefix inside the container; from
# outside the Pi it is http://<pi>:8000/imswitch, so set IMSWITCH_URL then.
BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

LASER_API = f"{BASE_URL}/api/LaserController"


def call(method, **params):
    """Call one LaserController endpoint and return its JSON."""
    response = requests.get(
        f"{LASER_API}/{method}",
        params=params,
        timeout=5,
    )

    assert response.status_code == 200, (
        f"{method} -> {response.status_code}: {response.text}"
    )

    return response.json()


def get_lightsource_params():
    """One pytest parameter per laser/LED of the active setup.

    Read at collection time. ImSwitch reads the setup at startup, so it must be
    restarted before a changed light source list appears here.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/api/AcceptanceTestController/getAvailableLightSources",
            timeout=5,
        )
        response.raise_for_status()

        data = response.json()
        names = [
            source["name"]
            for source in data.get("light_sources", [])
        ]

    except requests.RequestException as exc:
        return [
            pytest.param(
                None,
                marks=pytest.mark.skip(
                    reason=f"ImSwitch not reachable at {BASE_URL}: {exc}"
                ),
                id="ImSwitch-unreachable",
            )
        ]

    if not names:
        return [
            pytest.param(
                None,
                marks=pytest.mark.skip(
                    reason="active setup has no lasers/LEDs"
                ),
                id="no-lightsources",
            )
        ]

    # The light source name becomes the test ID, so it is visible in the output.
    return [
        pytest.param(
            name,
            id=str(name),
        )
        for name in names
    ]


@pytest.fixture
def safe_lightsource(request):
    """Hand over one light source and force it back to 0/inactive afterwards.

    Teardown runs even when the test fails, so none is left emitting.
    """
    name = request.param

    yield name

    if name is not None:
        try:
            call(
                "setLaserValue",
                laserName=name,
                value=0,
            )
        finally:
            call(
                "setLaserActive",
                laserName=name,
                active=False,
            )


@pytest.mark.hardware
@pytest.mark.parametrize(
    "safe_lightsource",
    get_lightsource_params(),
    indirect=True,
)
def test_lightsource_reports_active(safe_lightsource):
    """Enable one light source, read active and value back, disable it again.

    This proves the API path and that ImSwitch updates its own state. It does
    not prove that light was emitted - the readbacks are ImSwitch-side, not an
    optical measurement; test_lightsource_photon.py covers that.
    """
    name = safe_lightsource

    call(
        "setLaserValue",
        laserName=name,
        value=1,
    )

    call(
        "setLaserActive",
        laserName=name,
        active=True,
    )

    assert call(
        "getLaserActive",
        laserName=name,
    ) is True, f"{name}: did not become active"

    assert (
        call(
            "getLaserValue",
            laserName=name,
        ) or 0
    ) > 0, f"{name}: value is not positive"

    call(
        "setLaserActive",
        laserName=name,
        active=False,
    )

    assert call(
        "getLaserActive",
        laserName=name,
    ) is False, f"{name}: did not become inactive"

    call(
        "setLaserValue",
        laserName=name,
        value=0,
    )
