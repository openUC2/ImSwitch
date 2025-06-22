from logging import raiseExceptions
import numpy as np
import time
import cv2
from imswitch.imcommon.model import initLogger
from PIL import Image
from typing import List

from skimage.filters import gaussian, median
import imswitch.imcontrol.model.interfaces.gxipy as gx
import collections

class TriggerMode:
    SOFTWARE = 'Software Trigger'
    HARDWARE = 'Hardware Trigger'
    CONTINUOUS = 'Continuous Acquisition'

class CameraGXIPY:
    def __init__(self,cameraNo=None, exposure_time = 10000, gain = 0, frame_rate=-1, blacklevel=100, binning=1, flipImage=(False, False), isRGB=False):
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=True)

        # many to be purged
        self.model = "CameraGXIPY"
        self.shape = (0, 0)
        self.isRGB = isRGB
        self.is_connected = False
        self.is_streaming = False

        # unload CPU?
        self.downsamplepreview = 1

        # camera parameters
        self.blacklevel = blacklevel
        self.exposure_time = exposure_time
        self.gain = gain
        self.preview_width = 600
        self.preview_height = 600
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo
        self.flipImage = flipImage

        # reserve some space for the framebuffer
        self.NBuffer = 10
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.flatfieldImage = None
        self.isFlatfielding = False
        self.lastFrameId = -1
        self.frameNumber = -1
        self.frame = None

        # For RGB
        self.contrast_lut = None
        self.gamma_lut = None
        self.color_correction_param = 0

        # Initialize trigger source
        self.trigger_source = 'Continuous'


        #%% starting the camera thread
        self.camera = None

        # binning
        self.binning = binning

        self.device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.device_manager.update_device_list()

        if dev_num  != 0:
            self.__logger.debug("Trying to connect to camera: ")
            self.__logger.debug(dev_info_list)
            self._init_cam(cameraNo=self.cameraNo, binning=self.binning, callback_fct=self.set_frame)
        else :
            raise Exception("No camera GXIPY connected")


    def _init_cam(self, cameraNo=1, binning = 1, callback_fct=None):
        # start camera
        self.is_connected = True

        self.camera = self.device_manager.open_device_by_index(cameraNo)

        # reduce pixel number
        self.setBinning(binning)

        # set triggermode
        self.camera.TriggerMode.set(gx.GxSwitchEntry.OFF)

        # set exposure
        self.camera.ExposureTime.set(self.exposure_time)

        # set gain
        self.camera.Gain.set(self.gain)

        # set framerate
        self.set_frame_rate(self.frame_rate)

        # set blacklevel
        self.camera.BlackLevel.set(self.blacklevel)

        # set camera to mono12 mode
        availablePixelFormats = self.camera.PixelFormat.get_range()
        
        # Improve RGB detection logic
        self.isRGB = self._detect_rgb_camera()
        
        try:
            self.set_pixel_format(list(availablePixelFormats)[-1]) # last one is at highest bitrate
        except Exception as e:
            self.__logger.error(e)

        # get framesize
        self.SensorHeight = self.camera.HeightMax.get()//self.binning
        self.SensorWidth = self.camera.WidthMax.get()//self.binning

        # set the acq buffer count and register callback for improved frame handling
        user_param = None
        # Increase buffer count to avoid frame loss
        self.camera.data_stream[0].set_acquisition_buffer_number(self.NBuffer)

        # Register callback for frame capture
        try:
            self.camera.register_capture_callback(user_param, callback_fct)
        except Exception as e:
            self.__logger.warning(f"Failed to register capture callback: {e}")
            # Fall back to polling mode if callback fails

        # set things if RGB camera is used
        # get param of improving image quality
        if self.camera.GammaParam.is_readable():
            gamma_value = self.camera.GammaParam.get()
            self.gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        if self.camera.ContrastParam.is_readable():
            contrast_value = self.camera.ContrastParam.get()
            self.contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        if self.camera.ColorCorrectionParam.is_readable():
            self.color_correction_param = self.camera.ColorCorrectionParam.get()

    def start_live(self):
        if not self.is_streaming:
            # start data acquisition
            self.camera.stream_on()
            self.is_streaming = True

    def stop_live(self):
        if self.is_streaming:
            # start data acquisition
            self.camera.stream_off()
            self.is_streaming = False

    def suspend_live(self):
        if self.is_streaming:
        # start data acquisition
            try:
                self.camera.stream_off()
            except:
                # camera was disconnected?
                self.camera.unregister_capture_callback()
                self.camera.close_device()
                self._init_cam(cameraNo=self.cameraNo, binning=self.binning, callback_fct=self.set_frame)

            self.is_streaming = False

    def prepare_live(self):
        pass

    def close(self):
        self.camera.close_device()

    def set_flatfielding(self, is_flatfielding):
        self.isFlatfielding = is_flatfielding
        # record the flatfield image if needed
        if self.isFlatfielding:
            self.recordFlatfieldImage()

    def set_exposure_time(self,exposure_time):
        self.exposure_time = exposure_time
        self.camera.ExposureTime.set(self.exposure_time*1000)

    def set_exposure_mode(self, exposure_mode="manual"):
        """Set exposure mode to match HIK camera interface."""
        try:
            if exposure_mode == "manual":
                self.camera.ExposureAuto.set(gx.GxAutoEntry.OFF)
                self.__logger.debug("Exposure mode set to manual")
            elif exposure_mode == "auto":
                self.camera.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS)
                self.__logger.debug("Exposure mode set to auto continuous")
            elif exposure_mode == "once":
                self.camera.ExposureAuto.set(gx.GxAutoEntry.ONCE)
                self.__logger.debug("Exposure mode set to auto once")
            else:
                self.__logger.warning(f"Exposure mode '{exposure_mode}' not recognized. Valid modes: manual, auto, once")
                return False
            return True
        except Exception as e:
            self.__logger.error(f"Failed to set exposure mode '{exposure_mode}': {e}")
            return False

    def set_camera_mode(self, isAutomatic):
        """Set camera mode (automatic/manual) for compatibility with HIK interface."""
        exposure_mode = "auto" if str(isAutomatic).lower() in ('true', '1', 'automatic', 'auto') else "manual"
        return self.set_exposure_mode(exposure_mode)

    def set_gain(self,gain):
        self.gain = gain
        self.camera.Gain.set(self.gain)

    def set_frame_rate(self, frame_rate):
        if frame_rate == -1:
            frame_rate = 10000 # go as fast as you can
        self.frame_rate = frame_rate

        # temporary
        self.camera.AcquisitionFrameRate.set(self.frame_rate)
        self.camera.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)

    def set_blacklevel(self,blacklevel):
        self.blacklevel = blacklevel
        self.camera.BlackLevel.set(self.blacklevel)

    def set_pixel_format(self,format):
        format = format.upper()
        if self.camera.PixelFormat.is_implemented() and self.camera.PixelFormat.is_writable():
            if format == 'MONO8':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
            if format == 'MONO10':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO10)
            if format == 'MONO12':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO12)
            if format == 'MONO14':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO14)
            if format == 'MONO16':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO16)
            if format == 'BAYER_RG8':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG8)
            if format == 'BAYER_RG10':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG10)
            if format == 'BAYER_RG12':
                return self.camera.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG12)
        else:
            self.__logger.debug("pixel format is not implemented or not writable")
            return -1

    def setBinning(self, binning=1):
        # Unfortunately this does not work
        self.camera.BinningHorizontal.set(binning)
        self.camera.BinningVertical.set(binning)
        self.binning = binning

    def getLast(self, is_resize=True, returnFrameNumber=False, timeout=1):
        # get frame and save
        # only return fresh frames
        # print(self.lastFrameId, self.frameNumber)
        cTime = time.time()
        while(self.lastFrameId > self.frameNumber and self.frame is None):
            time.sleep(.01) # wait for fresh frame
            if time.time()-cTime > timeout:
                self.__logger.warning("Timeout in getLast")
                if returnFrameNumber:
                    return None, -1
        if self.isFlatfielding and self.flatfieldImage is not None:
            self.frame = self.frame/self.flatfieldImage
        self.lastFrameId = self.frameNumber
        if returnFrameNumber:
            return self.frame, self.frameNumber
        return self.frame

    def getLatestFrame(self, returnFrameNumber=False):
        """Alias for getLast to match detector interface."""
        return self.getLast(returnFrameNumber=returnFrameNumber)

    def flushBuffer(self):
        self.frameid_buffer.clear()
        self.frame_buffer.clear()

    def flushBuffers(self):
        """Alias for flushBuffer to match detector interface."""
        self.flushBuffer()

    def getLastChunk(self):
        chunk = np.array(self.frame_buffer)
        frameids = np.array(self.frameid_buffer)
        self.flushBuffer()
        self.__logger.debug("Buffer: "+str(chunk.shape)+" IDs: " + str(frameids))
        return chunk

    def getChunk(self):
        """Alias for getLastChunk to match detector interface."""
        return self.getLastChunk()

    def setROI(self,hpos=None,vpos=None,hsize=None,vsize=None):
        #hsize = max(hsize, 25)*10  # minimum ROI size
        #vsize = max(vsize, 3)*10  # minimum ROI size
        hpos = self.camera.OffsetX.get_range()["inc"]*((hpos)//self.camera.OffsetX.get_range()["inc"])
        vpos = self.camera.OffsetY.get_range()["inc"]*((vpos)//self.camera.OffsetY.get_range()["inc"])
        hsize = int(np.min((self.camera.Width.get_range()["inc"]*((hsize*self.binning)//self.camera.Width.get_range()["inc"]),self.camera.WidthMax.get())))
        vsize = int(np.min((self.camera.Height.get_range()["inc"]*((vsize*self.binning)//self.camera.Height.get_range()["inc"]),self.camera.HeightMax.get())))

        if vsize is not None:
            self.ROI_width = hsize
            # update the camera setting
            if self.camera.Width.is_implemented() and self.camera.Width.is_writable():
                message = self.camera.Width.set(self.ROI_width)
                self.__logger.debug(message)
            else:
                self.__logger.debug("OffsetX is not implemented or not writable")

        if hsize is not None:
            self.ROI_height = vsize
            # update the camera setting
            if self.camera.Height.is_implemented() and self.camera.Height.is_writable():
                message = self.camera.Height.set(self.ROI_height)
                self.__logger.debug(message)
            else:
                self.__logger.debug("Height is not implemented or not writable")

        if hpos is not None:
            self.ROI_hpos = hpos
            # update the camera setting
            if self.camera.OffsetX.is_implemented() and self.camera.OffsetX.is_writable():
                message = self.camera.OffsetX.set(self.ROI_hpos)
                self.__logger.debug(message)
            else:
                self.__logger.debug("OffsetX is not implemented or not writable")

        if vpos is not None:
            self.ROI_vpos = vpos
            # update the camera setting
            if self.camera.OffsetY.is_implemented() and self.camera.OffsetY.is_writable():
                message = self.camera.OffsetY.set(self.ROI_vpos)
                self.__logger.debug(message)
            else:
                self.__logger.debug("OffsetX is not implemented or not writable")

        return hpos,vpos,hsize,vsize


    def setPropertyValue(self, property_name, property_value):
        # Check if the property exists.
        if property_name == "gain":
            self.set_gain(property_value)
        elif property_name == "exposure":
            self.set_exposure_time(property_value)
        elif property_name == "exposure_mode":
            self.set_exposure_mode(property_value)
        elif property_name == "blacklevel":
            self.set_blacklevel(property_value)
        elif property_name == "flat_fielding":
            self.set_flatfielding(property_value)
        elif property_name == "roi_size":
            self.roi_size = property_value
        elif property_name == "frame_rate":
            self.set_frame_rate(property_value)
        elif property_name == "trigger_source":
            self.setTriggerSource(property_value)
        elif property_name == 'mode':
            self.set_camera_mode(property_value)
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def getPropertyValue(self, property_name):
        """Get camera property values with improved error handling.""" 
        try:
            # Check if the property exists.
            if property_name == "gain":
                property_value = self.camera.Gain.get()
            elif property_name == "exposure":
                property_value = self.camera.ExposureTime.get()
            elif property_name == "exposure_mode":
                # Get current exposure auto mode and convert to string
                exposure_auto_value = self.camera.ExposureAuto.get()
                if exposure_auto_value == gx.GxAutoEntry.OFF:
                    property_value = "manual"
                elif exposure_auto_value == gx.GxAutoEntry.CONTINUOUS:
                    property_value = "auto"
                elif exposure_auto_value == gx.GxAutoEntry.ONCE:
                    property_value = "once"
                else:
                    property_value = "unknown"
            elif property_name == "blacklevel":
                property_value = self.camera.BlackLevel.get()
            elif property_name == "image_width":
                property_value = self.camera.Width.get()//self.binning
            elif property_name == "image_height":
                property_value = self.camera.Height.get()//self.binning
            elif property_name == "roi_size":
                property_value = getattr(self, 'roi_size', None)
            elif property_name == "frame_rate":
                property_value = self.frame_rate
            elif property_name == "trigger_source":
                property_value = getattr(self, 'trigger_source', 'Continuous')
            elif property_name == "frame_number":
                property_value = self.getFrameNumber()
            else:
                self.__logger.warning(f'Property {property_name} does not exist')
                return False
            return property_value
        except Exception as e:
            self.__logger.error(f'Error getting property {property_name}: {e}')
            return False

    def setTriggerSource(self, trigger_source):
        """Set trigger source with standardized interface matching HIK camera."""
        was_streaming = self.is_streaming
        if was_streaming:
            self.suspend_live()
            
        tlow = str(trigger_source).lower()
        try:
            if tlow.find("cont") >= 0 or trigger_source == 'Continuous':
                self.set_continuous_acquisition()
                self.trigger_source = 'Continuous'
            elif tlow.find("int") >= 0 or tlow.find("soft") >= 0 or trigger_source == 'Internal trigger':
                self.set_software_triggered_acquisition()
                self.trigger_source = 'Internal trigger'
            elif tlow.find("ext") >= 0 or trigger_source == 'External trigger':
                self.set_hardware_triggered_acquisition()
                self.trigger_source = 'External trigger'
            else:
                self.__logger.warning(f"Unknown trigger source: {trigger_source}")
                return False
            return True
        finally:
            if was_streaming:
                self.start_live()

    def set_continuous_acquisition(self):
        """Configure camera for continuous (free-run) acquisition."""
        try:
            self.camera.TriggerMode.set(gx.GxSwitchEntry.OFF)
            self.trigger_mode = TriggerMode.CONTINUOUS
            self.__logger.debug("Continuous acquisition configured successfully")
        except Exception as e:
            self.__logger.error(f"Failed to configure continuous acquisition: {e}")

    def set_software_triggered_acquisition(self):
        """Configure camera for software trigger acquisition."""
        try:
            self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
            self.trigger_mode = TriggerMode.SOFTWARE
            self.__logger.debug("Software trigger acquisition configured successfully")
        except Exception as e:
            self.__logger.error(f"Failed to configure software trigger: {e}")

    def set_hardware_triggered_acquisition(self):
        """Configure camera for external hardware trigger with improved error handling."""
        try:
            # Enable trigger mode
            self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)

            # Set trigger source to LINE2 (external trigger)
            self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.LINE2)
            
            # Configure trigger line settings
            if self.camera.LineSelector.is_implemented():
                self.camera.LineSelector.set(2)  # Select LINE2
                
            if self.camera.LineMode.is_implemented():
                self.camera.LineMode.set(0)  # Set as input
                
            # Set trigger activation to rising edge (default)
            if self.camera.TriggerActivation.is_implemented():
                self.camera.TriggerActivation.set(gx.GxTriggerActivationEntry.RISING_EDGE)

            # Save settings to user set if available
            try:
                if self.camera.UserSetSelector.is_implemented():
                    self.camera.UserSetSelector.set(1)
                if self.camera.UserSetSave.is_implemented():
                    self.camera.UserSetSave.send_command()
            except Exception as e:
                self.__logger.debug(f"Could not save user settings: {e}")

            self.trigger_mode = TriggerMode.HARDWARE
            self.flushBuffer()
            self.__logger.debug("Hardware trigger acquisition configured successfully")
            
        except Exception as e:
            self.__logger.error(f"Failed to configure hardware trigger: {e}")
            # Fall back to software trigger
            self.set_software_triggered_acquisition()

    def getFrameNumber(self):
        return self.frameNumber

    def _detect_rgb_camera(self) -> bool:
        """Improved RGB camera detection logic."""
        try:
            # First check if PixelColorFilter is implemented (indicates Bayer pattern sensor)
            if self.camera.PixelColorFilter.is_implemented():
                return True
                
            # Check available pixel formats for color formats
            available_formats = self.camera.PixelFormat.get_range()
            rgb_keywords = ['RGB', 'BGR', 'BAYER', 'COLOR']
            for fmt in available_formats:
                fmt_str = str(fmt).upper()
                if any(keyword in fmt_str for keyword in rgb_keywords):
                    return True
                    
            # Check device model name for RGB indicators
            if hasattr(self.camera, 'DeviceModelName') and self.camera.DeviceModelName.is_readable():
                model_name = self.camera.DeviceModelName.get()
                if 'RGB' in model_name.upper() or 'COLOR' in model_name.upper():
                    return True
                    
        except Exception as e:
            self.__logger.debug(f"Error during RGB detection: {e}")
            
        return False

    def getTriggerTypes(self) -> List[str]:
        """Return a list of available trigger types to match HIK interface."""
        try:
            if not self.is_connected:
                return ["Camera not connected"]
            return [
                "Continuous",
                "Internal trigger", 
                "External trigger"
            ]
        except Exception as e:
            self.__logger.error(f"Error getting trigger types: {e}")
            return ["Error getting trigger types"]

    def getTriggerSource(self) -> str:
        """Return the current trigger source as a string to match HIK interface."""
        try:
            if not self.is_connected:
                return "Camera not connected"
            return getattr(self, 'trigger_source', 'Continuous')
        except Exception as e:
            self.__logger.error(f"Error getting trigger source: {e}")
            return "Error getting trigger source"

    def sendSoftwareTrigger(self):
        """Send software trigger for compatibility with detector interface."""
        return self.send_trigger()

    def send_trigger(self):
        """Fire software trigger pulse.""" 
        if self.is_streaming:
            self.camera.TriggerSoftware.send_command()
            return True
        else:
            self.__logger.debug('trigger not sent - camera is not streaming')
            return False

    def openPropertiesGUI(self):
        pass

    def set_frame(self, params, frame):
        """Callback function to process frames from the camera."""
        if frame is None:
            self.__logger.error("Getting image failed.")
            return
        if frame.get_status() != 0:
            self.__logger.error("Got an incomplete frame")
            return

        try:
            # if RGB camera
            if self.isRGB:
                rgb_image = frame.convert("RGB")
                if rgb_image is None:
                    return

                # improve image quality
                try:
                    rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)
                except Exception as e:
                    self.__logger.debug(f"Image improvement failed: {e}")

                # create numpy array with data from raw image
                numpy_image = rgb_image.get_numpy_array()
                if numpy_image is None:
                    return

            else:
                numpy_image = frame.get_numpy_array()

            # flip image if needed
            if self.flipImage[0]: # Y
                numpy_image = np.flip(numpy_image, axis=0)
            if self.flipImage[1]: # X
                numpy_image = np.flip(numpy_image, axis=1)
            if numpy_image is None:
                self.__logger.error("Got a None frame")
                return
                
            self.frame = numpy_image.copy()
            self.frameNumber = frame.get_frame_id()
            self.timestamp = time.time()

            # Add to ring buffer 
            self.frame_buffer.append(numpy_image)
            self.frameid_buffer.append(self.frameNumber)
            
        except Exception as e:
            self.__logger.error(f"Error processing frame: {e}")
            return

    def recordFlatfieldImage(self, nFrames=10, nGauss=5, nMedian=5):
        # record a flatfield image and save it in the flatfield variable
        flatfield = []
        for iFrame in range(nFrames):
            flatfield.append(self.getLast())
        flatfield = np.mean(np.array(flatfield),0)
        # normalize and smooth using scikit image
        flatfield = gaussian(flatfield, sigma=nGauss)
        flatfield = median(flatfield, selem=np.ones((nMedian, nMedian)))
        self.flatfieldImage = flatfield

    def setFlatfieldImage(self, flatfieldImage, isFlatfieldEnabeled=True):
        self.flatfieldImage = flatfieldImage
        self.isFlatfielding = isFlatfieldEnabeled


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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
