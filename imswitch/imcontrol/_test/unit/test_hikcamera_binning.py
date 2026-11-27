"""CameraHIK.setBinning against fake cameras: GenICam node names, legacy names, read-back."""
import logging

import imswitch.imcontrol.model.interfaces.hikcamera as hik

MV_E_SUPPORT, MV_E_GC_GENERIC = 0x80000001, 0x80000100
SENSOR = (4000, 3000)


class FakeCamera:
    def __init__(self, nodes, applied_ret=0):
        self.nodes, self.applied_ret = dict(nodes), applied_ret

    def MV_CC_SetEnumValueByString(self, node, value):
        return 0

    def MV_CC_SetIntValue(self, node, value):
        if node in ("Width", "Height"):
            return 0
        if node not in self.nodes:
            return MV_E_SUPPORT
        self.nodes[node] = value
        return self.applied_ret

    def MV_CC_GetIntValue(self, node, val):
        b = next(iter(self.nodes.values()))
        if node in ("WidthMax", "HeightMax"):
            val.nCurValue = SENSOR[node == "HeightMax"] // b
            return 0
        if node not in self.nodes:
            return MV_E_SUPPORT
        val.nCurValue = self.nodes[node]
        return 0


def camera(fake):
    cam = hik.CameraHIK.__new__(hik.CameraHIK)
    cam.camera = fake
    cam._CameraHIK__logger = logging.getLogger("test_hikcamera_binning")
    return cam


def test_genicam_names_trusted_by_readback_not_ret_code():
    # MV-CS200-10UC style: BinningHorizontal/Vertical, error code although applied
    cam = camera(FakeCamera({"BinningHorizontal": 1, "BinningVertical": 1}, MV_E_GC_GENERIC))
    cam.setBinning(2)
    assert cam.binning == 2
    assert (cam.SensorWidth, cam.SensorHeight) == (2000, 1500)


def test_legacy_names():
    cam = camera(FakeCamera({"BinningX": 1, "BinningY": 1}))
    cam.setBinning(4)
    assert cam.binning == 4 and cam.camera.nodes == {"BinningX": 4, "BinningY": 4}
    assert (cam.SensorWidth, cam.SensorHeight) == (1000, 750)


def test_refused_binning_reports_what_the_camera_does():
    class Refusing(FakeCamera):
        def MV_CC_SetIntValue(self, node, value):
            return 0 if node in ("Width", "Height") else MV_E_SUPPORT
    cam = camera(Refusing({"BinningX": 1, "BinningY": 1}))
    cam.setBinning(2)
    assert cam.binning == 1
