"""Virtual microscope: one light channel does not switch another off; snaps are fresh."""
import numpy as np

from imswitch.imcontrol.model.managers.rs232.VirtualMicroscopeManager import VirtualMicroscopy


def test_switching_one_source_off_keeps_the_other_on():
    scope = VirtualMicroscopy("smlm")
    light = scope.illuminator
    light.set_intensity(1, 300)        # LED
    light.set_intensity(2, 0)          # laser switched off afterwards
    assert light.get_intensity(1) == 300 and light.get_intensity() == 300
    light.set_intensity(2, 50)
    assert light.get_intensity() == 350 and light.intensity == 350


def test_get_last_returns_the_newest_queued_frame():
    camera = VirtualMicroscopy("smlm").camera
    camera.acquisition_active = True   # frames come from the queue, no producer thread here
    for n in range(3):
        camera.frame_queue.put((np.full((4, 4), n, np.uint16), n))
    frame, number = camera.getLast(returnFrameNumber=True)
    assert number == 2 and frame[0, 0] == 2 and camera.frame_queue.empty()
    camera.acquisition_active = False
