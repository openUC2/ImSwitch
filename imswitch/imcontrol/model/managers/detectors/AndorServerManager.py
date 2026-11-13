from imswitch.imcommon.model import initLogger
from .DetectorManager import (DetectorManager, DetectorAction, DetectorNumberParameter,
                              DetectorListParameter, DetectorBooleanParameter)

# Read back from andor-server by refreshParameters(); booleans are synced separately
# because _refreshParametersFromCamera skips False values.
_SERVER_READABLE_PARAMS = (
    'exposure', 'gain', 'blacklevel', 'frame_rate', 'frame_number', 'image_width', 'image_height',
    'trigger_source', 'temperature', 'target_temperature', 'fan_mode', 'temperature_status',
    'output_amplifier', 'readout_rate', 'preamp_gain', 'vs_speed', 'vs_amplitude', 'em_gain_mode',
    'shutter_mode',
)
_SERVER_BOOLEAN_PARAMS = ('cooler', 'em_advanced', 'frame_transfer', 'baseline_clamp')
_TEMPERATURE_STATES = ['off', 'not_stabilized', 'stabilized', 'not_reached', 'drift', 'out_of_range',
                       'not_supported', 'acquiring', 'unknown']


class AndorServerManager(DetectorManager):
    """ DetectorManager for an Andor SDK2 camera (iXon) served by andor-server,
    the HTTP sidecar that runs the x86_64-only Andor SDK (emulated on arm64).
    Parameter names follow ToupCamManager where the meaning is the same
    (exposure, gain, blacklevel, frame_rate, trigger_source, target_temperature,
    temperature, ...), so both cameras can be driven and compared the same way.
    Here ``gain`` is the EM gain.

    Manager properties:

    - ``host`` / ``port`` / ``basePath`` -- where andor-server listens
      (default ``andor-server``, ``8000``, ``""``)
    - ``connectTimeout`` -- seconds to wait for the server at start-up
      (default 30); after that the manager comes up disconnected and keeps
      reconnecting in the background
    - ``andor`` -- andor-server settings applied on (re)connect, e.g.
      ``{"exposure_ms": 100, "output_amplifier": "em", "em_gain": 2,
      "hs_speed_mhz": 10, "preamp_gain": 2, "baseline_clamp": true}``
      (see GET /settings on the server for all of them)
    - ``cooler`` -- applied on (re)connect, e.g.
      ``{"on": true, "setpoint_c": -60, "fan_mode": "full"}``
    - ``supportedBinnings`` -- default ``[1, 2, 4, 8]``; ``binning`` -- start-up
      binning (default 1)
    - ``longExposureMs`` -- from this exposure on (default 2000), acquisitions
      run as single exposures on the server (saved there as TIFF when
      ANDOR_SAVE_DIR is set, with progress in its /status) instead of a live
      stream
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo
        props = detectorInfo.managerProperties

        self._camera = self._getCameraObj(props)
        self._running = False
        self._adjustingParameters = False

        binning = int(props.get('binning', 1))
        try:
            supportedBinnings = [int(b) for b in props['supportedBinnings']]
            if not supportedBinnings:
                raise ValueError('empty supportedBinnings')
        except Exception:
            supportedBinnings = [b for b in (1, 2, 4, 8)
                                 if b in (self._camera.capabilities.get('binnings') or [1, 2, 4, 8])]
        if binning not in supportedBinnings:
            supportedBinnings.insert(0, binning)

        fullShape = (self._camera.SensorWidth, self._camera.SensorHeight)
        parameters = self._buildParameters(name)
        actions = {
            'More properties': DetectorAction(group='Misc', func=self._camera.openPropertiesGUI)
        }
        super().__init__(detectorInfo, name, fullShape=fullShape, supportedBinnings=supportedBinnings,
                         model=self._camera.model, parameters=parameters, actions=actions, croppable=True)

        # DetectorManager.__init__ applied supportedBinnings[0]; end up on the
        # requested binning, full frame.
        if binning != self.binning:
            self.setBinning(binning)
        if self._camera.is_connected:
            self.crop(hpos=0, vpos=0, hsize=self.fullShape[0], vsize=self.fullShape[1])
        self._camera.addConnectListener(self._onConnected)

    # ------------------------------------------------------------------ parameters
    def _buildParameters(self, name):
        cam = self._camera
        s, caps = cam.settings, cam.capabilities
        t = cam.status.get('temperature') or {}
        emLow, emHigh = caps.get('em_gain_range') or (0, 1000)
        tLow, tHigh = caps.get('temperature_range_c') or (-120, 20)
        bitDepth = max(caps.get('bit_depths') or [16])
        parameters = {
            'exposure': DetectorNumberParameter(group='Misc', value=s.get('exposure_ms', 100), valueUnits='ms',
                                                editable=True, valueMin=0, valueMax=24 * 3600 * 1000),
            'gain': DetectorNumberParameter(group='Misc', value=s.get('em_gain', 0), valueUnits='EM gain',
                                            editable=True, valueMin=emLow, valueMax=emHigh),
            'image_width': DetectorNumberParameter(group='Misc', value=cam.imageShape[0], valueUnits='px',
                                                   editable=False),
            'image_height': DetectorNumberParameter(group='Misc', value=cam.imageShape[1], valueUnits='px',
                                                    editable=False),
            'frame_rate': DetectorNumberParameter(group='Misc', value=s.get('live_max_fps', 5), valueUnits='fps',
                                                  editable=True, valueMin=0.1, valueMax=30),
            'frame_number': DetectorNumberParameter(group='Misc', value=0, valueUnits='frames', editable=False),
            'previewMinValue': DetectorNumberParameter(group='Misc', value=0, valueUnits='arb.u.', editable=True),
            'previewMaxValue': DetectorNumberParameter(group='Misc', value=2 ** bitDepth - 1, valueUnits='arb.u.',
                                                       editable=True),
            'trigger_source': DetectorListParameter(group='Acquisition mode', value=cam.getTriggerSource(),
                                                    options=cam.getTriggerTypes(), editable=True),
            'shutter_mode': DetectorListParameter(group='Acquisition mode', value=s.get('shutter_mode', 'auto'),
                                                  options=['auto', 'open', 'closed'], editable=True),
            'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=cam.pixelSizeUm,
                                                         valueUnits='µm', editable=True),
            'pixel_format': DetectorListParameter(group='Image format', value='mono16', options=['mono16'],
                                                  editable=False),
            'output_amplifier': DetectorListParameter(group='Readout', value=cam.getPropertyValue('output_amplifier')
                                                      or 'Electron Multiplying',
                                                      options=['Electron Multiplying', 'Conventional'], editable=True),
            'readout_rate': DetectorListParameter(group='Readout', value=cam.getPropertyValue('readout_rate') or '',
                                                  options=cam.readoutRateOptions(), editable=True),
            'preamp_gain': DetectorListParameter(group='Readout', value=cam.getPropertyValue('preamp_gain') or '',
                                                 options=cam.preampGainOptions(), editable=True),
            'vs_speed': DetectorListParameter(group='Readout', value=cam.getPropertyValue('vs_speed') or '',
                                              options=cam.vsSpeedOptions(), editable=True),
            'vs_amplitude': DetectorListParameter(group='Readout', value=cam.getPropertyValue('vs_amplitude') or 'Normal',
                                                  options=caps.get('vs_amplitudes') or ['Normal'], editable=True),
            'em_gain_mode': DetectorListParameter(group='Readout', value=s.get('em_gain_mode', 'real'),
                                                  options=caps.get('em_gain_modes') or ['real'], editable=True),
            'em_advanced': DetectorBooleanParameter(group='Readout', value=bool(s.get('em_advanced')), editable=True),
            'frame_transfer': DetectorBooleanParameter(group='Readout', value=bool(s.get('frame_transfer')),
                                                       editable=True),
            'baseline_clamp': DetectorBooleanParameter(group='Readout', value=bool(s.get('baseline_clamp', True)),
                                                       editable=True),
            'cooler': DetectorBooleanParameter(group='Cooling', value=bool(t.get('cooler_on')), editable=True),
            'target_temperature': DetectorNumberParameter(group='Cooling', value=t.get('setpoint_c') or -60,
                                                          valueUnits='°C', editable=True, valueMin=tLow, valueMax=tHigh),
            'temperature': DetectorNumberParameter(group='Cooling', value=t.get('celsius') or 0, valueUnits='°C',
                                                   editable=False),
            'temperature_status': DetectorListParameter(group='Cooling', value=t.get('status') or 'unknown',
                                                        options=_TEMPERATURE_STATES, editable=False),
            'fan_mode': DetectorListParameter(group='Cooling', value=t.get('fan_mode') or 'full',
                                              options=['full', 'low', 'off'], editable=True),
        }
        # Only heads with SetBaselineOffset have a black level setting (the iXon Ultra 888 has not).
        if (caps.get('supports') or {}).get('baseline_offset'):
            parameters['blacklevel'] = DetectorNumberParameter(group='Misc', value=s.get('baseline_offset') or 0,
                                                               valueUnits='counts', editable=True,
                                                               valueMin=-1000, valueMax=1000)
        return parameters

    def _syncFromServer(self):
        """Mirror the server's settings into the parameters: values, the options that depend
        on other settings (pre-amp gains valid at a readout rate), ranges and the frame shape."""
        cam, params = self._camera, self.parameters
        for name in _SERVER_READABLE_PARAMS:
            if name in params:
                value = cam.getPropertyValue(name)
                if value is not None:
                    params[name].value = value
        s = cam.settings
        params['cooler'].value = bool((cam.status.get('temperature') or {}).get('cooler_on'))
        for name, key in (('em_advanced', 'em_advanced'), ('frame_transfer', 'frame_transfer'),
                          ('baseline_clamp', 'baseline_clamp')):
            params[name].value = bool(s.get(key))
        params['readout_rate'].options = cam.readoutRateOptions()
        params['preamp_gain'].options = cam.preampGainOptions()
        params['vs_speed'].options = cam.vsSpeedOptions()
        emRange = (cam.status.get('image') or {}).get('em_gain_range') or cam.capabilities.get('em_gain_range')
        if emRange:
            params['gain'].valueMin, params['gain'].valueMax = emRange
        width, height = cam.imageShape
        params['image_width'].value, params['image_height'].value = width, height
        params['frame_number'].value = cam.getFrameNumber()

    def _onConnected(self):
        """(Re)connected to andor-server: refresh options and values from the real camera."""
        try:
            self._syncFromServer()
            self._setFullShape((self._camera.SensorWidth // self._camera.binning,
                                self._camera.SensorHeight // self._camera.binning))
        except Exception as e:  # noqa: BLE001
            self.__logger.debug(f'Could not refresh parameters after connecting: {e}')

    def setParameter(self, name, value):
        """Sets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""
        super().setParameter(name, value)
        if name in ('previewMinValue', 'previewMaxValue', 'Camera pixel size'):
            return value
        if name.find('posure') > 0 and name != 'exposure_mode':
            name = 'exposure'
        if name not in self._DetectorManager__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        from imswitch.imcontrol.model.interfaces.andorservercamera import AndorServerError
        try:
            value = self._camera.setPropertyValue(name, value)
        except AndorServerError as e:
            # e.g. 409 while a long exposure runs: keep showing what the camera really does
            self.__logger.error(f'Setting {name} failed: {e}')
        try:
            self._syncFromServer()
        except Exception as e:  # noqa: BLE001
            self.__logger.debug(f'Could not refresh parameters: {e}')
        return self.parameters[name].value if name in self.parameters else value

    def getParameter(self, name):
        """Gets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""
        if name.find('posure') > 0 and name != 'exposure_mode':
            name = 'exposure'
        if name not in self.parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')
        value = self._camera.getPropertyValue(name)
        return self.parameters[name].value if value is None else value

    def refreshParameters(self):
        """Re-read temperature, settings and frame counters from andor-server."""
        try:
            self._camera.refreshSettings()
            self._syncFromServer()
        except Exception as e:  # noqa: BLE001
            self.__logger.debug(f'Could not refresh parameters: {e}')
        return self.parameters

    def getParameterRanges(self) -> dict:
        params = self.parameters
        return {name: {'min': params[name].valueMin, 'max': params[name].valueMax,
                       'units': params[name].valueUnits}
                for name in ('exposure', 'gain', 'blacklevel', 'target_temperature') if name in params}

    def isLongExposure(self) -> bool:
        return self._camera.isLongExposure()

    # ------------------------------------------------------------------ frames
    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        return self._camera.getLast(returnFrameNumber=returnFrameNumber)

    def getChunk(self):
        try:
            return self._camera.getLastChunk()
        except Exception:
            return None

    def flushBuffer(self):
        """Drop buffered frames so the next grab is guaranteed post-move/-settle."""
        self._camera.flushBuffer()

    def flushBuffers(self):
        self._camera.flushBuffer()

    def snapSync(self, timeout: float = 2.0):
        """Take one exposure and return it (software-trigger mode)."""
        return self._camera.snapSoftwareTrigger(timeout=timeout)

    def getFrameNumber(self):
        return self._camera.getFrameNumber()

    def getStreamDiagnostics(self):
        return self._camera.getStreamDiagnostics()

    # ------------------------------------------------------------------ acquisition
    def startAcquisition(self):
        if not self._running:
            self._camera.start_live()
            self._running = True
            self.__logger.debug('startlive')

    def stopAcquisition(self):
        if self._running:
            self._running = False
            self._camera.suspend_live()
            self.__logger.debug('suspendlive')

    def stopAcquisitionForROIChange(self):
        self._running = False
        self._camera.stop_live()

    def _performSafeCameraAction(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        self._adjustingParameters = True
        wasrunning = self._running
        self.stopAcquisitionForROIChange()
        try:
            function()
        finally:
            if wasrunning:
                self.startAcquisition()
            self._adjustingParameters = False

    def setTriggerSource(self, source):
        self.__logger.debug(f'Setting trigger source to {source}')
        self._camera.setTriggerSource(source)
        self.parameters['trigger_source'].value = self._camera.getTriggerSource()
        return True

    def sendSoftwareTrigger(self):
        """Start one exposure; its frame arrives through getLatestFrame()."""
        if self._camera.send_trigger():
            self.__logger.debug('Software trigger sent successfully.')
        else:
            self.__logger.warning('Failed to send software trigger.')

    def getCurrentTriggerType(self):
        return self._camera.getTriggerSource()

    def getTriggerTypes(self):
        return self._camera.getTriggerTypes()

    # ------------------------------------------------------------------ geometry
    def setBinning(self, binning):
        """Square binning on the camera; the full frame shrinks accordingly."""
        super().setBinning(binning)
        if self._camera.binning == binning:
            return

        def binningAction():
            self._camera.setBinning(binning)
            binnedShape = (self._camera.SensorWidth // binning, self._camera.SensorHeight // binning)
            self._shape = self._camera.imageShape
            self._setFullShape(binnedShape)

        try:
            self._performSafeCameraAction(binningAction)
        except Exception as e:  # noqa: BLE001
            self.__logger.error(f'Failed to set binning {binning}: {e}')

    def crop(self, hpos, vpos, hsize, vsize):
        '''
        hpos - horizontal start position of crop window
        vpos - vertical start position of crop window
        hsize - horizontal size of crop window
        vsize - vertical size of crop window
        All in pixels of the (binned) full frame.
        '''
        def cropAction():
            b = self._camera.binning
            self.__logger.debug(f'{self._camera.model}: crop frame to {hsize}x{vsize} at {hpos},{vpos} (binning {b}).')
            self._camera.setROI(hpos * b, vpos * b, hsize * b, vsize * b)
            self._shape = self._camera.imageShape
            self._frameStart = (hpos, vpos)
        try:
            self._performSafeCameraAction(cropAction)
        except Exception as e:  # noqa: BLE001
            self.__logger.error(e)

    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def setPixelSizeUm(self, pixelSizeUm):
        self.parameters['Camera pixel size'].value = pixelSizeUm

    def setFlipImage(self, flipY: bool, flipX: bool):
        self._camera.flipImage = (flipY, flipX)
        self.__logger.info(f"Updated flip settings: flipY={flipY}, flipX={flipX}")

    # ------------------------------------------------------------------ lifecycle, status
    def openPropertiesDialog(self):
        self._camera.openPropertiesGUI()

    def finalize(self) -> None:
        super().finalize()
        self.__logger.debug('Disconnecting from andor-server (the camera stays initialised there)...')
        self._camera.close()

    def closeEvent(self):
        self._camera.close()

    def _getCameraObj(self, props):
        from imswitch.imcontrol.model.interfaces.andorservercamera import CameraAndorServer
        camera = CameraAndorServer(
            host=props.get('host', 'andor-server'), port=props.get('port', 8000),
            basePath=props.get('basePath', ''), connectTimeout=float(props.get('connectTimeout', 30)),
            requestTimeout=float(props.get('requestTimeout', 10)),
            settings=props.get('andor'), cooler=props.get('cooler'),
            longExposureMs=float(props.get('longExposureMs', 2000)))
        self.__logger.info(f'Initialized andor-server camera {camera.baseUrl}, model: {camera.model}, '
                           f'connected: {camera.is_connected}')
        return camera

    def getCameraStatus(self):
        """ Returns comprehensive camera status information. """
        status = super().getCameraStatus()
        cam = self._camera
        status['cameraType'] = 'AndorServer'
        status['isMock'] = (cam.info.get('server') or {}).get('backend') == 'mock'
        status['isConnected'] = cam.is_connected
        status['parameterRanges'] = self.getParameterRanges()
        status['isLongExposure'] = self.isLongExposure()
        status['isAcquiring'] = self._running
        status['isAdjustingParameters'] = self._adjustingParameters
        status['server'] = {k: cam.status.get(k) for k in ('state', 'error', 'exposure', 'live', 'temperature')}
        status['serverUrl'] = cam.baseUrl
        status['currentTriggerSource'] = cam.getTriggerSource()
        status['availableTriggerTypes'] = cam.getTriggerTypes()
        return status

# Copyright (C) ImSwitch developers 2021
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
