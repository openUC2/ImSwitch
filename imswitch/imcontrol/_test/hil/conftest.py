"""What applies to every HIL test. Nothing here knows what a test checks."""
from pathlib import Path

import pytest

import hil_client as client


def pytest_collection_modifyitems(items):
    """Mark every HIL test ``hardware``, so ``pytest -m "not hardware"`` leaves the suite out."""
    for item in items:
        if Path(__file__).parent in Path(item.fspath).parents:
            item.add_marker(pytest.mark.hardware)


@pytest.fixture(scope="session", autouse=True)
def imswitch_reachable():
    """Without a server the whole suite skips."""
    if not client.reachable():
        pytest.skip(f"ImSwitch not reachable at {client.IMSWITCH_URL}")
