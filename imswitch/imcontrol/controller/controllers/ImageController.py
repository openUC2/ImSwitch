from imswitch.imcontrol.view import guitools
from ..basecontrollers import LiveUpdatedController
from imswitch.imcommon.model import initLogger, APIExport
import numpy as np
import time

class ImageController(LiveUpdatedController):
    """ Linked to ImageWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__logger = initLogger(self, tryInheritParent=False)
        if not self._master.detectorsManager.hasDevices():
            return

        self._lastShape = self._master.detectorsManager.execOnCurrent(lambda c: c.shape)
        self._shouldResetView = False

        names =  self._master.detectorsManager.getAllDeviceNames(lambda c: c.forAcquisition)
        if type(names) is not list:
            names = [names]

        isRGB = []
        for name in names:
            try:
                isRGB.append(self._master.detectorsManager[name]._isRGB)
            except:
                isRGB.append(False)

        self._widget.setLiveViewLayers(
            self._master.detectorsManager.getAllDeviceNames(lambda c: c.forAcquisition), isRGB
        )

        # Connect CommunicationChannel signals
        self._commChannel.sigUpdateImage.connect(self.update)
        self._commChannel.sigAdjustFrame.connect(self.adjustFrame)
        self._commChannel.sigGridToggled.connect(self.gridToggle)
        self._commChannel.sigCrosshairToggled.connect(self.crosshairToggle)
        self._commChannel.sigAddItemToVb.connect(self.addItemToVb)
        self._commChannel.sigRemoveItemFromVb.connect(self.removeItemFromVb)
        self._commChannel.sigMemorySnapAvailable.connect(self.memorySnapAvailable)
        self._commChannel.sigSetExposure.connect(lambda t: self.setExposure(t))


    @APIExport(runOnUIThread=False)
    def displayImageNapari(self, layerName, mImage, isRGB=False, scale=(1,1), isCurrentDetector=None): # TODO: Flag of RGB is not used!
        self._commChannel.sigUpdateImage.emit(layerName, mImage, scale, isCurrentDetector)

    def autoLevels(self, detectorNames=None, im=None):
        """ Set histogram levels automatically with current detector image."""
        if detectorNames is None:
            detectorNames = self._master.detectorsManager.getAllDeviceNames(
                lambda c: c.forAcquisition
            )

        for detectorName in detectorNames:
            if im is None:
                im = self._widget.getImage(detectorName)

            # self._widget.setImageDisplayLevels(detectorName, *guitools.bestLevels(im))
            self._widget.setImageDisplayLevels(detectorName, *guitools.minmaxLevels(im))

    def addItemToVb(self, item):
        """ Add item from communication channel to viewbox."""
        item.hide()
        self._widget.addItem(item)

    def removeItemFromVb(self, item):
        """ Remove item from communication channel to viewbox."""
        self._widget.removeItem(item)

    def update(self, detectorName, im, init, scale, isCurrentDetector):
        """ Update new image in the viewbox. """
        if np.prod(im.shape)>1: # TODO: This seems weird!

            if not init:
                self.autoLevels([detectorName], im)

            self._widget.setImage(detectorName, im, scale)

            if not init or self._shouldResetView:
                self.adjustFrame(instantResetView=True)

    def adjustFrame(self, shape=None, instantResetView=False):
        """ Adjusts the viewbox to a new width and height. """

        if shape is None:
            shape = self._lastShape

        self._widget.updateGrid(shape)
        if instantResetView:
            self._widget.resetView()
            self._shouldResetView = False
        else:
            self._shouldResetView = True

        self._lastShape = shape

    def getCenterViewbox(self):
        """ Returns center of viewbox to center a ROI. """
        return self._widget.getCenterViewbox()

    def gridToggle(self, enabled):
        """ Shows or hides grid. """
        self._widget.setGridVisible(enabled)

    def crosshairToggle(self, enabled):
        """ Shows or hides crosshair. """
        self._widget.setCrosshairVisible(enabled)

    def memorySnapAvailable(self, name, image, _, __):
        """ Adds captured image to widget. """
        self._widget.addStaticLayer(name, image)
        if self._shouldResetView:
            self.adjustFrame(image.shape, instantResetView=True)

    def setExposure(self, exp):
        detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
        self.__logger.debug(f"Change exposure of {detectorName}, to {str(exp)}")
        #self._master.detectorsManager[detectorName].setParameter('Readout time', exp)

    @APIExport(runOnUIThread=True)
    def setImageLayer(self, layerName, image, isRGB=False): #(layername, image, isRGB)
        """ Set image layer to widget. """
        self._commChannel.sigUpdateImage.emit(layerName, image, True, (1,1), True)
        ## (detectorName, image, init, scale, isCurrentDetector)

    @APIExport(runOnUIThread=False)
    def snapFrame(self, detector_name: str = None, format: str = "png16") -> dict:
        """
        Capture a single frame and return it in the specified format.
        
        Args:
            detector_name: Name of detector (use current if None)
            format: Output format ("png16", "tiff", "binary")
            
        Returns:
            Dictionary with frame data and metadata
        """
        try:
            # Get current detector if not specified
            if detector_name is None:
                detector_name = self._master.detectorsManager.getCurrentDetectorName()

            # Get latest frame
            detector = self._master.detectorsManager[detector_name]
            frame = detector.getLatestFrame()

            if frame is None:
                raise ValueError("No frame available")

            # Ensure uint16 format for consistency
            if frame.dtype != np.uint16:
                if frame.dtype == np.uint8:
                    frame = frame.astype(np.uint16) << 8
                else:
                    frame = frame.astype(np.uint16)

            result = {
                "detector": detector_name,
                "format": format,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "timestamp": time.time()
            }

            if format == "binary":
                # Use binary encoder
                from imswitch.imcommon.framework.binary_streaming import BinaryFrameEncoder
                from imswitch.config import get_config

                config = get_config()
                encoder = BinaryFrameEncoder(
                    compression_algorithm="none",  # No compression for snapshots
                    subsampling_factor=1,  # No subsampling for snapshots
                    bitdepth=config.stream_binary_bitdepth_in,
                    pixfmt=config.stream_binary_pixfmt
                )

                packet, metadata = encoder.encode_frame(frame)
                result["data"] = packet
                result["metadata"] = metadata

            elif format in ["png16", "tiff"]:
                # Save to temporary file and return path/data
                import tempfile
                from PIL import Image
                import tifffile

                if format == "png16":
                    # Save as 16-bit PNG
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        # PIL doesn't handle 16-bit grayscale well, use tifffile
                        if len(frame.shape) == 2:
                            img = Image.fromarray(frame, mode='I;16')
                            img.save(tmp.name, "PNG")
                        else:
                            raise ValueError("RGB PNG16 not implemented")
                        result["file_path"] = tmp.name

                elif format == "tiff":
                    # Save as TIFF
                    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
                        tifffile.imwrite(tmp.name, frame)
                        result["file_path"] = tmp.name

            else:
                raise ValueError(f"Unsupported format: {format}")

            return result

        except Exception as e:
            self.__logger.error(f"Error capturing frame: {e}")
            return {"error": str(e)}



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
