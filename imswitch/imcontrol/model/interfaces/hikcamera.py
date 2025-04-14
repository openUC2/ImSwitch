from logging import raiseExceptions
import numpy as np
import time
import cv2
from imswitch.imcommon.model import initLogger
from skimage.filters import gaussian, median

import sys
import threading
from ctypes import *
import collections

from sys import platform
try:
    if platform == "linux" or platform == "linux2":
        # linux
        from imswitch.imcontrol.model.interfaces.hikrobotMac.MvCameraControl_class import *
    elif platform == "darwin":
        # OS X
        from imswitch.imcontrol.model.interfaces.hikrobotMac.MvCameraControl_class import *
        pass
    elif platform == "win32":
        import msvcrt
        from imswitch.imcontrol.model.interfaces.hikrobotWin.MvCameraControl_class import *
except Exception as e:
    print(e)

# Some possible YUV pixel formats:
PixelType_Gvsp_YUV444_Packed = 35127328
PixelType_Gvsp_YUV422_YUYV_Packed = 34603058
PixelType_Gvsp_YUV422_Packed = 34603039
PixelType_Gvsp_YUV411_Packed = 34340894

class CameraHIK:
    def __init__(self,cameraNo=None, exposure_time = 10000, gain = 0, frame_rate=-1, blacklevel=100, isRGB=False, binning=2):
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=False)

        self.model = "CameraHIK"
        self.shape = (0, 0)
        self.is_connected = False
        self.is_streaming = False
        self.downsamplepreview = 1

        self.blacklevel = blacklevel
        self.exposure_time = exposure_time
        self.gain = gain
        self.preview_width = 600
        self.preview_height = 600
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo

        self.NBuffer = 1
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.flatfieldImage = None
        self.camera = None

        # Binning
        if platform in ("darwin", "linux2", "linux"):
            binning = 2
        self.binning = binning

        self.SensorHeight = 0
        self.SensorWidth = 0
        self.frame = np.zeros((self.SensorHeight, self.SensorWidth))

        self.lastFrameId = -1
        self.frameNumber = -1
        self.g_bExit = False

        self.isRGB = isRGB
        self.isFlatfielding = False

        self._init_cam(cameraNo=self.cameraNo, callback_fct=None)

    def _init_cam(self, cameraNo=1, callback_fct=None):


        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise Exception("Enum devices fail! ret[0x%x]" % ret)

        if deviceList.nDeviceNum == 0:
            raise Exception("No camera HIK connected")

        self.camera = MvCamera()
        self.stDeviceList = cast(deviceList.pDeviceInfo[int(cameraNo)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.camera.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            raise Exception("Create handle fail! ret[0x%x]" % ret)

        ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception("Open device fail! ret[0x%x]" % ret)

        # get available parameters
        self.mParameters = self.get_camera_parameters()
        self.isRGB = self.mParameters["isRGB"]
        
        # set parameters
        self.setBinning(binning=self.binning)

        stBool = c_bool(False)
        ret = self.camera.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
        if ret != 0:
            self.__logger.debug("Get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)

        ret = self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.__logger.debug("Set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # Use YUV format if isRGB is True (instead of Bayer)
        if self.isRGB:
            self.camera.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_YUV422_YUYV_Packed)

        stIntValue_height = MVCC_INTVALUE()
        memset(byref(stIntValue_height), 0, sizeof(MVCC_INTVALUE))
        stIntValue_width = MVCC_INTVALUE()
        memset(byref(stIntValue_width), 0, sizeof(MVCC_INTVALUE))

        ret = self.camera.MV_CC_GetIntValue("Height", stIntValue_height)
        if ret != 0:
            raise Exception("Get height fail! ret[0x%x]" % ret)
        self.SensorHeight = stIntValue_height.nCurValue

        ret = self.camera.MV_CC_GetIntValue("Width", stIntValue_width)
        if ret != 0:
            raise Exception("Get width fail! ret[0x%x]" % ret)
        self.SensorWidth = stIntValue_width.nCurValue
        self.is_connected = True
        print(f"Current number of pixels: Width = {self.SensorWidth}, Height = {self.SensorHeight}")

    def reconnectCamera(self):
        # Safely close any existing handle
        if self.camera is not None:
            try:
                self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
            except Exception as e:
                self.__logger.error(f"Error while closing camera handle: {e}")
            self.camera = None

        # Re-initialize camera with original cameraNo
        try:
            self._init_cam(cameraNo=self.cameraNo, callback_fct=None)
            self.__logger.debug("Camera reconnected successfully.")
        except Exception as e:
            self.__logger.error(f"Failed to reconnect camera: {e}")
            
    def get_camera_parameters(self):
        param_dict = {}

        # PixelFormat and check if color
        stPixelFormat = MVCC_ENUMVALUE()
        ret = self.camera.MV_CC_GetEnumValue("PixelFormat", stPixelFormat)
        if ret == 0:
            param_dict["pixel_format"] = stPixelFormat.nCurValue
        
        # camera Name 
        stName = MVCC_STRINGVALUE()
        param_dict["isRGB"] = False            
        ret = self.camera.MV_CC_GetStringValue("DeviceModelName", stName)
        if ret == 0:
            param_dict["model_name"] = stName.chCurValue.decode("utf-8")
            if param_dict["model_name"].find("UC")>0:
                param_dict["isRGB"] = True

        # Image Width
        stWidth = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("Width", stWidth)
        if ret == 0:
            param_dict["width"] = stWidth.nCurValue

        # Image Height
        stHeight = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("Height", stHeight)
        if ret == 0:
            param_dict["height"] = stHeight.nCurValue

        # Current / Min / Max Gain
        stGain = MVCC_FLOATVALUE()
        ret = self.camera.MV_CC_GetFloatValue("Gain", stGain)
        if ret == 0:
            param_dict["gain_current"] = stGain.fCurValue
            param_dict["gain_min"] = stGain.fMin
            param_dict["gain_max"] = stGain.fMax

        # Current / Min / Max Exposure
        stExposure = MVCC_FLOATVALUE()
        ret = self.camera.MV_CC_GetFloatValue("ExposureTime", stExposure)
        if ret == 0:
            param_dict["exposure_current"] = stExposure.fCurValue
            param_dict["exposure_min"] = stExposure.fMin
            param_dict["exposure_max"] = stExposure.fMax

        return param_dict


    def start_live(self):
        if not self.is_streaming:
            self.g_bExit = False
            ret = self.camera.MV_CC_StartGrabbing()
            self.__logger.debug("start grabbing")
            self.__logger.debug(ret)
            try:
                self.hThreadHandle = threading.Thread(target=self.work_thread, args=(self.camera, None, None))
                self.hThreadHandle.start()
            except Exception:
                self.__logger.error("Could not start frame grabbing")

            if ret != 0:
                self.__logger.debug("start grabbing fail! ret[0x%x]" % ret)
                return
            self.is_streaming = True

    def stop_live(self):
        if self.is_streaming:
            self.g_bExit = True
            self.hThreadHandle.join()
            self.is_streaming = False

    def suspend_live(self):
        if self.is_streaming:
            self.g_bExit = True
            try:
                self.hThreadHandle.join()
                ret = self.camera.MV_CC_StopGrabbing()
            except:
                pass
            self.is_streaming = False

    def prepare_live(self):
        pass

    def close(self):
        ret = self.camera.MV_CC_CloseDevice()
        ret = self.camera.MV_CC_DestroyHandle()

    def set_exposure_time(self, exposure_time):
        self.exposure_time = exposure_time
        self.camera.MV_CC_SetFloatValue("ExposureTime", self.exposure_time * 1000)

    def set_exposure_mode(self, exposure_mode="manual"):
        if exposure_mode == "manual":
            self.camera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        elif exposure_mode == "auto":
            self.camera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        elif exposure_mode == "once":
            self.camera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_ONCE)
        else:
            self.__logger.warning("Exposure mode not recognized")

    def set_camera_mode(self, isAutomatic):
        self.set_exposure_mode("auto" if isAutomatic else "manual")

    def set_gain(self, gain):
        self.gain = gain
        self.camera.MV_CC_SetFloatValue("Gain", self.gain)

    def set_frame_rate(self, frame_rate):
        ret = self.camera.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        if ret != 0:
            self.__logger.error("set AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        ret = self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", 5.0)
        if ret != 0:
            self.__logger.error("set AcquisitionFrameRate fail! ret[0x%x]" % ret)

    def set_flatfielding(self, is_flatfielding):
        self.isFlatfielding = is_flatfielding
        if self.isFlatfielding:
            self.recordFlatfieldImage()

    def setFlatfieldImage(self, flatfieldImage, isFlatfieldEnabeled=True):
        self.flatfieldImage = flatfieldImage
        self.isFlatfielding = isFlatfieldEnabeled

    def set_blacklevel(self, blacklevel):
        self.blacklevel = blacklevel
        self.camera.MV_CC_SetFloatValue("BlackLevel", self.blacklevel)

    def set_pixel_format(self, format):
        # Example pixel format setting for mono:
        self.camera.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8_Signed)

    def setBinning(self, binning=1):
        try:
            self.camera.MV_CC_SetIntValue("BinningX", binning)
            self.camera.MV_CC_SetIntValue("BinningY", binning)
            self.binning = binning
        except Exception as e:
            self.__logger.error(e)

    def getLast(self, returnFrameNumber=False, timeout=1):
        start_time = time.time()
        while len(self.frame_buffer) == 0:
            time.sleep(0.02)
            if time.time() - start_time > timeout:
                return None

        frame = self.frame_buffer[-1]
        frameNumber = self.frameid_buffer[-1]
        if returnFrameNumber:
            return np.array(frame), frameNumber
        return np.array(frame)

    def flushBuffer(self):
        self.frameid_buffer.clear()
        self.frame_buffer.clear()

    def getLastChunk(self):
        chunk = np.array(self.frame_buffer)
        frameids = np.array(self.frameid_buffer)
        self.flushBuffer()
        self.__logger.debug("Buffer: "+str(chunk.shape)+" IDs: " + str(frameids))
        return chunk

    def setROI(self,hpos=None,vpos=None,hsize=None,vsize=None):
        # Not updated. Provided as example
        hpos = self.camera.OffsetX.get_range()["inc"]*((hpos)//self.camera.OffsetX.get_range()["inc"])
        vpos = self.camera.OffsetY.get_range()["inc"]*((vpos)//self.camera.OffsetY.get_range()["inc"])
        hsize = int(np.min((self.camera.Width.get_range()["inc"]*((hsize*self.binning)//self.camera.Width.get_range()["inc"]),self.camera.WidthMax.get())))
        vsize = int(np.min((self.camera.Height.get_range()["inc"]*((vsize*self.binning)//self.camera.Height.get_range()["inc"]),self.camera.HeightMax.get())))

        if vsize is not None:
            self.ROI_width = hsize
            if self.camera.Width.is_implemented() and self.camera.Width.is_writable():
                message = self.camera.Width.set(self.ROI_width)
                self.__logger.debug(message)
            else:
                self.__logger.debug("OffsetX is not implemented or not writable")

        if hsize is not None:
            self.ROI_height = vsize
            if self.camera.Height.is_implemented() and self.camera.Height.is_writable():
                message = self.camera.Height.set(self.ROI_height)
                self.__logger.debug(message)
            else:
                self.__logger.debug("Height is not implemented or not writable")

        if hpos is not None:
            self.ROI_hpos = hpos
            if self.camera.OffsetX.is_implemented() and self.camera.OffsetX.is_writable():
                message = self.camera.OffsetX.set(self.ROI_hpos)
                self.__logger.debug(message)
            else:
                self.__logger.debug("OffsetX is not implemented or not writable")

        if vpos is not None:
            self.ROI_vpos = vpos
            if self.camera.OffsetY.is_implemented() and self.camera.OffsetY.is_writable():
                message = self.camera.OffsetY.set(self.ROI_vpos)
                self.__logger.debug(message)
            else:
                self.__logger.debug("OffsetY is not implemented or not writable")

        return hpos,vpos,hsize,vsize

    def setPropertyValue(self, property_name, property_value):
        if property_name == "gain":
            self.set_gain(property_value)
        elif property_name == "exposure":
            self.set_exposure_time(property_value)
        elif property_name == "exposure_mode":
            self.set_exposure_mode(property_value)
        elif property_name == "blacklevel":
            self.set_blacklevel(property_value)
        elif property_name == "roi_size":
            self.roi_size = property_value
        elif property_name == "frame_rate":
            self.set_frame_rate(property_value)
        elif property_name == "flat_fielding":
            self.set_flatfielding(property_value)
        elif property_name == "trigger_source":
            self.setTriggerSource(property_value)
        elif property_name == 'mode':
            self.set_camera_mode(property_value)
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def getPropertyValue(self, property_name):
        if property_name == "gain":
            property_value = self.camera.Gain.get()
        elif property_name == "exposure":
            property_value = self.camera.ExposureTime.get()
        elif property_name == "frame_number":
            property_value = self.getFrameNumber()
        elif property_name == "exposure_mode":
            property_value = self.camera.ExposureAuto.get()
        elif property_name == "blacklevel":
            property_value = self.camera.BlackLevel.get()
        elif property_name == "image_width":
            property_value = self.camera.Width.get()//self.binning
        elif property_name == "image_height":
            property_value = self.camera.Height.get()//self.binning
        elif property_name == "roi_size":
            property_value = self.roi_size
        elif property_name == "frame_Rate":
            property_value = self.frame_rate
        elif property_name == "trigger_source":
            property_value = self.trigger_source
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def setTriggerSource(self, trigger_source):
        pass

    def send_trigger(self):
        pass

    def openPropertiesGUI(self):
        pass

    def work_thread(self, cam=0, pData=0, nDataSize=0):
        if platform == "win32":
            stOutFrame = MV_FRAME_OUT()
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))

            while True:
                if self.g_bExit:
                    break
                if not self.is_connected:
                    # reconnect the camera 
                    self.__logger.debug("Camera disconnected, trying to reconnect...")
                    self.reconnectCamera()
                ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
                if (stOutFrame.pBufAddr is not None) and (ret == 0):
                    if self.isRGB:
                        nRGBSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
                        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
                        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                        stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
                        stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
                        stConvertParam.pSrcData = stOutFrame.pBufAddr
                        stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
                        stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
                        stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
                        stConvertParam.nDstBufferSize = nRGBSize

                        ret = cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
                        if ret != 0:
                            self.__logger.error("convert pixel fail! ret[0x%x]" % ret)
                            return

                        cam.MV_CC_FreeImageBuffer(stOutFrame)

                        try:
                            img_buff = (c_ubyte * stConvertParam.nDstLen)()
                            cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                            data = np.frombuffer(img_buff, count=int(nRGBSize), dtype=np.uint8)
                            self.frame = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, -1))
                            self.SensorHeight, self.SensorWidth = self.frame.shape[0], self.frame.shape[1]
                            self.frameNumber = stOutFrame.stFrameInfo.nFrameNum
                            self.timestamp = time.time()
                            self.frame_buffer.append(self.frame)
                            self.frameid_buffer.append(self.frameNumber)

                        except Exception as e:
                            self.__logger.error(e)
                            self.is_connected = False
                            self.__logger.error("Get image fail! ret[0x%x]" % ret)
                    else:
                        cam.MV_CC_FreeImageBuffer(stOutFrame)
                        pData = (c_ubyte * (stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight))()
                        cdll.msvcrt.memcpy(
                            byref(pData),
                            stOutFrame.pBufAddr,
                            stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
                        )
                        data = np.frombuffer(
                            pData,
                            count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                            dtype=np.uint8
                        )
                        self.frame = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                        self.SensorHeight, self.SensorWidth = self.frame.shape[0], self.frame.shape[1]
                        self.frameNumber = stOutFrame.stFrameInfo.nFrameNum
                        self.timestamp = time.time()
                        self.frame_buffer.append(self.frame)
                        self.frameid_buffer.append(self.frameNumber)

                else:
                    pass
                if self.g_bExit:
                    break

        if platform in ("darwin", "linux2", "linux"):
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                self.__logger.error("get payload size fail! ret[0x%x]" % ret)

            nPayloadSize = stParam.nCurValue
            stDeviceList = MV_FRAME_OUT_INFO_EX()
            memset(byref(stDeviceList), 0, sizeof(stDeviceList))

            while True:
                if not self.is_connected:
                    # reconnect the camera 
                    self.__logger.debug("Camera disconnected, trying to reconnect...")
                    self.reconnectCamera()
                if self.g_bExit:
                    break
                if self.isRGB:
                    try:
                        stDeviceList = MV_FRAME_OUT_INFO_EX()
                        memset(byref(stDeviceList), 0, sizeof(stDeviceList))
                        data_buf = (c_ubyte * nPayloadSize)()
                        ret = cam.MV_CC_GetOneFrameTimeout(
                            byref(data_buf), nPayloadSize, stDeviceList, 1000
                        )
                        if ret == 0:
                            nRGBSize = stDeviceList.nWidth * stDeviceList.nHeight * 3
                            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                            stConvertParam.nWidth = stDeviceList.nWidth
                            stConvertParam.nHeight = stDeviceList.nHeight
                            stConvertParam.pSrcData = data_buf
                            stConvertParam.nSrcDataLen = stDeviceList.nFrameLen
                            stConvertParam.enSrcPixelType = stDeviceList.enPixelType
                            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                            stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
                            stConvertParam.nDstBufferSize = nRGBSize

                            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
                            if ret != 0:
                                self.__logger.error("convert pixel fail! ret[0x%x]" % ret)
                                del data_buf
                                sys.exit()

                            img_buff = (c_ubyte * stConvertParam.nDstLen)()
                            memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                            data = np.frombuffer(img_buff, count=int(nRGBSize), dtype=np.uint8)
                            self.frame = data.reshape((stDeviceList.nHeight, stDeviceList.nWidth, -1))
                            self.lastFrameId = stDeviceList.nFrameNum

                            self.SensorHeight, self.SensorWidth = stDeviceList.nWidth, stDeviceList.nHeight
                            self.frameNumber = stDeviceList.nFrameNum
                            self.timestamp = time.time()
                            self.frame_buffer.append(self.frame)
                            self.frameid_buffer.append(self.frameNumber)

                    except Exception as e:
                        self.__logger.error("Get image fail! ret[0x%x]" % ret)
                        self.__logger.error(e)
                        del data_buf
                        self.is_connected = False
                else:
                    data_buf = (c_ubyte * nPayloadSize)()
                    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stDeviceList, 100)
                    if ret == 0:
                        data = np.frombuffer(
                            data_buf, count=int(stDeviceList.nWidth * stDeviceList.nHeight), dtype=np.uint8
                        )
                        self.frame = data.reshape((stDeviceList.nHeight, stDeviceList.nWidth))
                        self.SensorHeight, self.SensorWidth = stDeviceList.nWidth, stDeviceList.nHeight
                        self.frameNumber = stDeviceList.nFrameNum
                        self.timestamp = time.time()
                        self.frame_buffer.append(self.frame)
                        self.frameid_buffer.append(self.frameNumber)
                    else:
                        self.is_connected = False
                        self.__logger.error("Get image fail! ret[0x%x]" % ret)
                        del data_buf
                        
            if self.g_bExit:
                return

    def recordFlatfieldImage(self, nFrames=10, nGauss=5, nMedian=5):
        for iFrame in range(nFrames):
            frame = self.getLast()
            if frame is None:
                continue
            if iFrame == 0:
                flatfield = frame
            else:
                flatfield += frame
        flatfield = flatfield / nFrames
        flatfield = gaussian(flatfield, sigma=nGauss)
        flatfield = median(flatfield, selem=np.ones((nMedian, nMedian)))
        self.flatfieldImage = flatfield
        
    def getFrameNumber(self):
        return self.frameNumber

