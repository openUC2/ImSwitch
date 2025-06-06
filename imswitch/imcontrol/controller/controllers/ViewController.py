from imswitch.imcommon.model import APIExport
from imswitch import IS_HEADLESS
from ..basecontrollers import ImConWidgetController


class ViewController(ImConWidgetController):
    """ Linked to ViewWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acqHandle = None
        self._commChannel.sigStartLiveAcquistion.connect(self.startLiveView)
        self._commChannel.sigStopLiveAcquisition.connect(self.stopLiveView)

        if IS_HEADLESS: return
        self._widget.setViewToolsEnabled(False)

        # Connect ViewWidget signals
        self._widget.sigGridToggled.connect(self.gridToggle)
        self._widget.sigCrosshairToggled.connect(self.crosshairToggle)
        self._widget.sigLiveviewToggled.connect(self.liveview)

    def startLiveView(self):
        self.liveview(enabled=True)

    def stopLiveView(self):
        self.liveview(enabled=False)

    def liveview(self, enabled):
        """ Start liveview and activate detector acquisition. """
        if enabled and self._acqHandle is None:
            self._acqHandle = self._master.detectorsManager.startAcquisition(liveView=True)
        elif not enabled and self._acqHandle is not None:
            self._master.detectorsManager.stopAcquisition(self._acqHandle, liveView=True)
            self._acqHandle = None
        if not IS_HEADLESS:
            self._widget.setViewToolsEnabled(enabled)
            self._widget.setLiveViewActive(enabled)

    def gridToggle(self, enabled):
        """ Connect with grid toggle from Image Widget through communication channel. """
        self._commChannel.sigGridToggled.emit(enabled)

    def crosshairToggle(self, enabled):
        """ Connect with crosshair toggle from Image Widget through communication channel. """
        self._commChannel.sigCrosshairToggled.emit(enabled)

    def closeEvent(self):
        if self._acqHandle is not None:
            self._master.detectorsManager.stopAcquisition(self._acqHandle, liveView=True)

    def get_image(self, detectorName):
        if detectorName is None:
            return self._master.detectorsManager.execOnCurrent(lambda c: c.getLatestFrame())
        else:
            return self._master.detectorsManager[detectorName].getLatestFrame()

    @APIExport(runOnUIThread=True)
    def setLiveViewActive(self, active: bool) -> None:
        """ Sets whether the LiveView is active and updating. """
        if IS_HEADLESS:
            self.liveview(active)
        else:
            self._widget.setLiveViewActive(active)

    @APIExport(runOnUIThread=False)
    def getLiveViewActive(self) -> bool:
        """ Returns whether the LiveView is active and updating. """
        if not IS_HEADLESS:
            return self._widget.getLiveViewActive()
        else:
            return self._acqHandle is not None

    @APIExport(runOnUIThread=True)
    def setLiveViewGridVisible(self, visible: bool) -> None:
        """ Sets whether the LiveView grid is visible. """
        if not IS_HEADLESS: self._widget.setLiveViewGridVisible(visible)

    @APIExport(runOnUIThread=True)
    def setLiveViewCrosshairVisible(self, visible: bool) -> None:
        """ Sets whether the LiveView crosshair is visible. """
        if not IS_HEADLESS: self._widget.setLiveViewCrosshairVisible(visible)


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
