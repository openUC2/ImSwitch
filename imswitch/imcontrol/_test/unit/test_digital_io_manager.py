"""Named digital lines over the (virtual) UC2 board."""
import pytest

from imswitch.imcontrol.model.SetupInfo import DigitalIOInfo, RS232Info, SetupInfo
from imswitch.imcontrol.model.managers.DigitalIOManager import DigitalIOManager
from imswitch.imcontrol.model.managers.rs232.VirtualESP32Manager import VirtualESP32Manager


@pytest.fixture
def board():
    # input 1 follows output 1, like a loop-back cable
    info = RS232Info(managerName="VirtualESP32Manager",
                     managerProperties={"digitalIn": {"1": {"mirrorOut": 1}}})
    manager = VirtualESP32Manager(info, "ESP32")
    yield {"rs232sManager": {"ESP32": manager}}
    manager.finalize()


def test_raw_lines_without_a_block(board):
    dio = DigitalIOManager(None, **board)
    assert dio.outputs == ["1", "2", "3"] and dio.inputs == ["1", "2", "3"]
    dio.set_output("1", True)
    assert dio.read_input("1") is True


def test_named_and_inverted_lines(board):
    dio = DigitalIOManager(DigitalIOInfo(outputs={"shutter": 1}, inputs={"door_open": 1},
                                         invert=["door_open"]), **board)
    assert dio.outputs == ["shutter"] and dio.inputs == ["door_open"]
    dio.set_output("shutter", True)
    assert dio.read_input("door_open") is False         # line high, active low
    assert dio.wait_for_input("door_open", "low", timeout_s=0.5)
    assert not dio.wait_for_input("door_open", "rising", timeout_s=0.1)
    with pytest.raises(ValueError, match="no digital output '1'"):
        dio.set_output("1", True)


def test_no_board_means_no_lines():
    dio = DigitalIOManager(DigitalIOInfo(outputs={"shutter": 1}))
    assert not dio.available and dio.outputs == []
    with pytest.raises(RuntimeError):
        dio.set_output("shutter", True)


def test_setup_block_parses():
    setup = SetupInfo.from_dict({"digitalIO": {"outputs": {"shutter": 1}, "invert": ["shutter"]}})
    assert setup.digitalIO.outputs == {"shutter": 1} and setup.digitalIO.rs232device == "ESP32"
    assert SetupInfo.from_dict({}).digitalIO is None
