from ctypes import *
import ctypes.util
from imswitch.imcontrol.model.interfaces.pyAndorSDK2 import atmcd_errors
import platform
import os
import sys
from typing import Union, List
import numpy as np

MAX_PATH = 256


class ColorDemosaicInfo(Structure):
    _fields_ = [("iX", c_int),
                ("iY", c_int),
                ("iAlgorithm", c_int),
                ("iXPhase", c_int),
                ("iYPhase", c_int),
                ("iBackground", c_int)]


class AndorCapabilities(Structure):
    _fields_ = [("ulSize", c_ulong),
                ("ulAcqModes", c_ulong),
                ("ulReadModes", c_ulong),
                ("ulTriggerModes", c_ulong),
                ("ulCameraType", c_ulong),
                ("ulPixelMode", c_ulong),
                ("ulSetFunctions", c_ulong),
                ("ulGetFunctions", c_ulong),
                ("ulFeatures", c_ulong),
                ("ulPCICard", c_ulong),
                ("ulEMGainCapability", c_ulong),
                ("ulFTReadModes", c_ulong),
                ("ulFeatures2", c_ulong)]


class WhiteBalanceInfo(Structure):
    _fields_ = [("iSize", c_int),
                ("iX", c_int),
                ("iY", c_int),
                ("iAlgorithm", c_int),
                ("iROI_left", c_int),
                ("iROI_right", c_int),
                ("iROI_top", c_int),
                ("iROI_bottom", c_int),
                ("iOperation", c_int)]


class SYSTEMTIME(Structure):
    _fields_ = [("wYear", c_short),
                ("wMonth", c_short),
                ("wDayOfWeek", c_short),
                ("wDay", c_short),
                ("wHour", c_short),
                ("wMinute", c_short),
                ("wSecond", c_short),
                ("wMilliseconds", c_short)]


class ATMCDException(Exception):
    pass


class atmcd:
    __version__ = '0.1'

    def __init__(self, userPath=None):
        self.dll = self._load_library(userPath)

    def _load_library(self, userPath):
        if sys.platform == "linux":
            return cdll.LoadLibrary("libandor.so")
        elif sys.platform == "win32":
            if userPath is not None:
                _path = userPath + ';' + os.environ['PATH']
                os.environ['PATH'] = _path
            if platform.machine() == "AMD64":
                dllname = "atmcd64d.dll"
            else:
                dllname = "atmcd32d.dll"
            path = ctypes.util.find_library(dllname)
            return windll.LoadLibrary(path)
        else:
            raise ATMCDException(
                "Cannot detect operating system, will now stop")

    def handle_return(self, ret_value: int) -> int:
        if ret_value != atmcd_errors.Error_Codes.DRV_SUCCESS:
            raise ATMCDException('{}'.format(ret_value))
        return ret_value

    def acquire(self) -> np.ndarray:
        ret, xpixels, ypixels = self.GetDetector()
        self.handle_return(ret)

        self.handle_return(self.PrepareAcquisition())
        self.handle_return(self.StartAcquisition())
        self.handle_return(self.WaitForAcquisition())

        imageSize = xpixels * ypixels
        ret, fullFrameBuffer = self.GetMostRecentImage(imageSize)
        self.handle_return(ret)

        return np.ctypeslib.as_array(fullFrameBuffer)

    def acquire_series(self, length: int) -> List[np.ndarray]:
        series = []
        for _ in range(length):
            series.append(self.acquire())
        return series

    def AbortAcquisition(self):
        """
            Description:
            This function aborts the current acquisition if one is active.

            Synopsis:
            ret = AbortAcquisition()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition aborted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_IDLE - The system is not currently acquiring.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_ERROR_ACK - Unable to communicate with card.

            C++ Equiv:
            unsigned int AbortAcquisition(void);

            See Also:
            GetStatus StartAcquisition

        """
        ret = self.dll.AbortAcquisition()
        return (ret)

    def CancelWait(self):
        """
            Description:
            This function restarts a thread which is sleeping within the WaitForAcquisition function. The sleeping thread will return from WaitForAcquisition with a value not equal to DRV_SUCCESS.

            Synopsis:
            ret = CancelWait()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Thread restarted successfully.

            C++ Equiv:
            unsigned int CancelWait(void);

            See Also:
            WaitForAcquisition

        """
        ret = self.dll.CancelWait()
        return (ret)

    def CoolerOFF(self):
        """
            Description:
            Switches OFF the cooling. The rate of temperature change is controlled in some models until the temperature reaches 0C. Control is returned immediately to the calling application.

            Synopsis:
            ret = CoolerOFF()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Temperature controller switched OFF.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_NOT_SUPPORTED - Camera does not support switching cooler off.

            C++ Equiv:
            unsigned int CoolerOFF(void);

            See Also:
            CoolerON SetTemperature GetTemperature GetTemperatureF GetTemperatureRange GetStatus

            Note: Not available on Luca R cameras - always cooled to -20C.

                NOTE: (Classic & ICCD only)  1. When the temperature control is switched off the temperature of the sensor is gradually raised to 0C to ensure no thermal stresses are set up in the sensor.  2. When closing down the program via ShutDown you must ensure that the temperature of the detector is above -20C, otherwise calling ShutDown while the detector is still cooled will cause the temperature to rise faster than certified.


        """
        ret = self.dll.CoolerOFF()
        return (ret)

    def CoolerON(self):
        """
            Description:
            Switches ON the cooling. On some systems the rate of temperature change is controlled until the temperature is within 3C of the set value.
            Control is returned immediately to the calling application.

            Synopsis:
            ret = CoolerON()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Temperature controller switched ON.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.

            C++ Equiv:
            unsigned int CoolerON(void);

            See Also:
            CoolerOFF SetTemperature GetTemperature GetTemperatureF GetTemperatureRange GetStatus

            Note: The temperature to which the detector will be cooled is set via SetTemperature.
                The temperature stabilization is controlled via hardware, and the current temperature can be obtained via GetTemperature.
                The temperature of the sensor is gradually brought to the desired temperature to ensure no thermal stresses are set up in the sensor.
                Can be called for certain systems during an acquisition. This can be tested for using GetCapabilities.

        """
        ret = self.dll.CoolerON()
        return (ret)

    def DemosaicImage(self, grey, info):
        """
            Description:
            For colour sensors only
            Demosaics an image taken with a CYMG CCD into RGB using the parameters stored in info. Below is the ColorDemosaicInfo structure definition and a description of its members:
            struct COLORDEMOSAICINFO {
            int iX; // Number of X pixels. Must be >2.
            int iY; // Number of Y pixels. Must be >2.
            int iAlgorithm; // Algorithm to demosaic image.
            int iXPhase; // First pixel in data (Cyan or Yellow/Magenta or Green).
            int iYPhase; // First pixel in data (Cyan or Yellow/Magenta or Green).
            int iBackground; // Background to remove from raw data when demosaicing.
            ColorDemosaicInfo;
            * iX and iY are the image dimensions. The number of elements in the input red, green and blue arrays is iX x iY.
            * iAlgorithm sets the algorithm to use: 0 for a 2x2 matrix demosaic algorithm or 1 for a 3x3 one.
            The CYMG CCD pattern can be broken into cells of 2x4 pixels, e.g.:
            * iXPhase and iYPhase store what colour is the bottom-left pixel.
            * iBackground sets the numerical value to be removed from every pixel in the input image before demosaicing is done.

            Synopsis:
            (ret, red, green, blue) = DemosaicImage(grey, info)

            Inputs:
            grey - pointer to image to demosaic
            info - pointer to demosaic information structure.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Image demosaiced
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Invalid pointer (i.e. NULL).
                DRV_P3INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Invalid pointer (i.e. NULL).
                DRV_P5INVALID - One or more parameters in info is out of range
            red - pointer to the red plane storage allocated by the user.
            green - pointer to the green plane storage allocated by the user.
            blue - pointer to the blue plane storage allocated by the user.

            C++ Equiv:
            unsigned int DemosaicImage(WORD * grey, WORD * red, WORD * green, WORD * blue, ColorDemosaicInfo * info);

            See Also:
            GetMostRecentColorImage16 WhiteBalance

        """
        cgrey = (c_short * info.iX * info.iY)(*grey)
        cred = (c_short * info.iX * info.iY)()
        cgreen = (c_short * info.iX * info.iY)()
        cblue = (c_short * info.iX * info.iY)()
        cinfo = ColorDemosaicInfo(info)
        ret = self.dll.DemosaicImage(cgrey, cred, cgreen, cblue, byref(cinfo))
        return (ret, cred, cgreen, cblue)

    def EnableKeepCleans(self, mode):
        """
            Description:
            This function is only available on certain cameras operating in FVB external trigger mode.  It determines if the camera keep clean cycle will run between acquisitions.
            When keep cleans are disabled in this way the exposure time is effectively the exposure time between triggers.
            The Keep Clean cycle is enabled by default.
            The feature capability AC_FEATURES_KEEPCLEANCONTROL determines if this function can be called for the camera.

            Synopsis:
            ret = EnableKeepCleans(mode)

            Inputs:
            mode - The keep clean mode.:
                0 - OFF
                1 - ON

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Keep clean cycle mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_AVAILABLE - Feature not available.

            C++ Equiv:
            unsigned int EnableKeepCleans(int mode);

            See Also:
            GetCapabilities

            Note: Currently only available on Newton and iKon cameras operating in FVB external trigger mode.

        """
        cmode = c_int(mode)
        ret = self.dll.EnableKeepCleans(cmode)
        return (ret)

    def EnableSensorCompensation(self, mode):
        """
            Description:
            This function enables/disables the on camera sensor compensation.

            Synopsis:
            ret = EnableSensorCompensation(mode)

            Inputs:
            mode - :
                unsigned int -

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Mode successfully selected
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported on this camera
                DRV_P1INVALID - Requested mode isn’t valid

            C++ Equiv:
            unsigned int EnableSensorCompensation(int mode);

            See Also:
            GetCapabilities

            Note: This function enables/disables the on camera sensor compensation.

        """
        cmode = c_int(mode)
        ret = self.dll.EnableSensorCompensation(cmode)
        return (ret)

    def Filter_GetAveragingFactor(self):
        """
            Description:
            Returns the current averaging factor value.

            Synopsis:
            (ret, averagingFactor) = Filter_GetAveragingFactor()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Frame count returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid averagingFactor (i.e. NULL pointer).
            averagingFactor - The current averaging factor value.

            C++ Equiv:
            unsigned int Filter_GetAveragingFactor(int * averagingFactor);

            See Also:
            Filter_SetAveragingFactor

        """
        caveragingFactor = c_int()
        ret = self.dll.Filter_GetAveragingFactor(byref(caveragingFactor))
        return (ret, caveragingFactor.value)

    def Filter_GetAveragingFrameCount(self):
        """
            Description:
            Returns the current frame count value.

            Synopsis:
            (ret, frames) = Filter_GetAveragingFrameCount()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Frame count returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid frame count (i.e. NULL pointer).
            frames - The current frame count value.

            C++ Equiv:
            unsigned int Filter_GetAveragingFrameCount(int * frames);

            See Also:
            Filter_SetAveragingFrameCount

        """
        cframes = c_int()
        ret = self.dll.Filter_GetAveragingFrameCount(byref(cframes))
        return (ret, cframes.value)

    def Filter_GetDataAveragingMode(self):
        """
            Description:
            Returns the current averaging mode.

            Synopsis:
            (ret, mode) = Filter_GetDataAveragingMode()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Averaging mode returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid threshold (i.e. NULL pointer).
            mode - The current averaging mode.

            C++ Equiv:
            unsigned int Filter_GetDataAveragingMode(int * mode);

            See Also:
            Filter_SetDataAveragingMode

        """
        cmode = c_int()
        ret = self.dll.Filter_GetDataAveragingMode(byref(cmode))
        return (ret, cmode.value)

    def Filter_GetMode(self):
        """
            Description:
            Returns the current Noise Filter mode.

            Synopsis:
            (ret, mode) = Filter_GetMode()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Filter mode returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Noise Filter processing not available for this camera.
                DRV_P1INVALID - Invalid mode (i.e. NULL pointer)
            mode - Noise Filter mode.

            C++ Equiv:
            unsigned int Filter_GetMode(unsigned int * mode);

            See Also:
            Filter_SetMode

        """
        cmode = c_uint()
        ret = self.dll.Filter_GetMode(byref(cmode))
        return (ret, cmode.value)

    def Filter_GetThreshold(self):
        """
            Description:
            Returns the current Noise Filter threshold value.

            Synopsis:
            (ret, threshold) = Filter_GetThreshold()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Threshold returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Noise Filter processing not available for this camera.
                DRV_P1INVALID - Invalid threshold (i.e. NULL pointer).
            threshold - The current threshold value.

            C++ Equiv:
            unsigned int Filter_GetThreshold(float * threshold);

            See Also:
            Filter_SetThreshold

        """
        cthreshold = c_float()
        ret = self.dll.Filter_GetThreshold(byref(cthreshold))
        return (ret, cthreshold.value)

    def Filter_SetAveragingFactor(self, averagingFactor):
        """
            Description:
            Sets the averaging factor.

            Synopsis:
            ret = Filter_SetAveragingFactor(averagingFactor)

            Inputs:
            averagingFactor - The averaging factor to use.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Averaging factor set.
                DRV_NOT_INITIALIZED DRV_ACQUIRING - System not initialized.
                DRV_P1INVALID - Acquisition in progress.

            C++ Equiv:
            unsigned int Filter_SetAveragingFactor(int averagingFactor);

            See Also:
            Filter_GetAveragingFactor

        """
        caveragingFactor = c_int(averagingFactor)
        ret = self.dll.Filter_SetAveragingFactor(caveragingFactor)
        return (ret)

    def Filter_SetAveragingFrameCount(self, frames):
        """
            Description:
            Sets the averaging frame count.

            Synopsis:
            ret = Filter_SetAveragingFrameCount(frames)

            Inputs:
            frames - The averaging frame count to use.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Averaging frame count set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid frame count.

            C++ Equiv:
            unsigned int Filter_SetAveragingFrameCount(int frames);

            See Also:
            Filter_GetAveragingFrameCount

        """
        cframes = c_int(frames)
        ret = self.dll.Filter_SetAveragingFrameCount(cframes)
        return (ret)

    def Filter_SetDataAveragingMode(self, mode):
        """
            Description:
            Sets the current data averaging mode.

            Synopsis:
            ret = Filter_SetDataAveragingMode(mode)

            Inputs:
            mode - The averaging  factor mode to use.:
                0 - No Averaging Filter
                5 - Recursive Averaging Filter
                6 - Frame Averaging Filter

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Averaging mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid mode.

            C++ Equiv:
            unsigned int Filter_SetDataAveragingMode(int mode);

            See Also:
            Filter_GetDataAveragingMode

        """
        cmode = c_int(mode)
        ret = self.dll.Filter_SetDataAveragingMode(cmode)
        return (ret)

    def Filter_SetMode(self, mode):
        """
            Description:
            Set the Noise Filter to use.

            Synopsis:
            ret = Filter_SetMode(mode)

            Inputs:
            mode - Filter mode to use.:
                0 - No Filter
                1 - Median Filter
                2 - Level Above Filter
                3 - interquartile Range Filter
                4 - Noise Threshold Filter

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Filter set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Noise Filter processing not available for this camera.
                DRV_P1INVALID - Invalid mode.

            C++ Equiv:
            unsigned int Filter_SetMode(int mode);

            See Also:
            Filter_GetMode

        """
        cmode = c_int(mode)
        ret = self.dll.Filter_SetMode(cmode)
        return (ret)

    def Filter_SetThreshold(self, threshold):
        """
            Description:
            Sets the threshold value for the Noise Filter.

            Synopsis:
            ret = Filter_SetThreshold(threshold)

            Inputs:
            threshold - Threshold value used to process image.:
                0 - 65535  for Level Above filter.
                0 - 10 for all other filters.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Threshold set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Noise Filter processing not available for this camera.
                DRV_P1INVALID - Invalid threshold.

            C++ Equiv:
            unsigned int Filter_SetThreshold(float threshold);

            See Also:
            Filter_GetThreshold

        """
        cthreshold = c_float(threshold)
        ret = self.dll.Filter_SetThreshold(cthreshold)
        return (ret)

    def FreeInternalMemory(self):
        """
            Description:
            The FreeinternalMemory function will deallocate any memory used internally to store the previously acquired data. Note that once this function has been called, data from last acquisition cannot be retrieved.

            Synopsis:
            ret = FreeInternalMemory()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Memory freed.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.

            C++ Equiv:
            unsigned int FreeInternalMemory(void);

            See Also:
            GetImages PrepareAcquisition

        """
        ret = self.dll.FreeInternalMemory()
        return (ret)

    def GetAcquiredData(self, size):
        """
            Description:
            This function will return the data from the last acquisition. The data are returned as long integers (32-bit signed integers). The array must be large enough to hold the complete data set.

            Synopsis:
            (ret, arr) = GetAcquiredData(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data copied.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - No acquisition has taken place
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetAcquiredData(at_32 * arr, unsigned long size);

            See Also:
            GetStatus StartAcquisition GetAcquiredData16

        """
        carr = (c_int * size)()
        csize = c_ulong(size)
        ret = self.dll.GetAcquiredData(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetAcquiredData16(self, size):
        """
            Description:
            16-bit version of the GetAcquiredData function. The array must be large enough to hold the complete data set.

            Synopsis:
            (ret, arr) = GetAcquiredData16(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data copied.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size isincorrect.
                DRV_NO_NEW_DATA - No acquisition has taken place
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetAcquiredData16(WORD * arr, unsigned long size);

            See Also:
            GetStatus StartAcquisition GetAcquiredData

        """
        carr = (c_short * size)()
        csize = c_ulong(size)
        ret = self.dll.GetAcquiredData16(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetAcquiredFloatData(self, size):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, arr) = GetAcquiredFloatData(size)

            Inputs:
            size -

            Outputs:
            ret - Function Return Code
            arr -

            C++ Equiv:
            unsigned int GetAcquiredFloatData(float * arr, unsigned long size);

        """
        carr = (c_float * size)()
        csize = c_ulong(size)
        ret = self.dll.GetAcquiredFloatData(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetAcquisitionProgress(self):
        """
            Description:
            This function will return information on the progress of the current acquisition. It can be called at any time but is best used in conjunction with SetDriverEventSetDriverEvent.
            The values returned show the number of completed scans in the current acquisition.
            If 0 is returned for both accum and series then either:-
            * No acquisition is currently running
            * The acquisition has just completed
            * The very first scan of an acquisition has just started and not yet completed
            GetStatus can be used to confirm if the first scan has just started, returning
            DRV_ACQUIRING, otherwise it will return DRV_IDLE.
            For example, if [i]accum[/i]=2 and [i]series[/i]=3 then the acquisition has completed 3 in the series and 2 accumulations in the 4 scan of the series.

            Synopsis:
            (ret, acc, series) = GetAcquisitionProgress()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of accumulation and series scans completed.
                DRV_NOT_INITIALIZED - System not initialized.
            acc - returns the number of accumulations completed in the current kinetic scan.
            series - series the number of kinetic scans completed

            C++ Equiv:
            unsigned int GetAcquisitionProgress(long * acc, long * series);

            See Also:
            SetAcquisitionMode SetNumberAccumulations SetNumberKinetics SetDriverEvent

        """
        cacc = c_int()
        cseries = c_int()
        ret = self.dll.GetAcquisitionProgress(byref(cacc), byref(cseries))
        return (ret, cacc.value, cseries.value)

    def GetAcquisitionTimings(self):
        """
            Description:
            This function will return the current "valid" acquisition timing information.
            This function should be used after all the acquisitions settings have been set, e.g.
            SetExposureTimeSetExposureTime, SetKineticCycleTimeSetKineticCycleTime and SetReadModeSetReadMode etc.
            The values returned are the actual times used in subsequent acquisitions.
            This function is required as it is possible to set the exposure time to 20ms, accumulate cycle time to 30ms and then set the readout mode to full image.
            As it can take 250ms to read out an image it is not possible to have a cycle time of 30ms.

            Synopsis:
            (ret, exposure, accumulate, kinetic) = GetAcquisitionTimings()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timing information returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_INVALID_MODE - Acquisition or readout mode is not available.
            exposure - valid exposure time in seconds
            accumulate - valid accumulate cycle time in seconds
            kinetic - valid kinetic cycle time in seconds

            C++ Equiv:
            unsigned int GetAcquisitionTimings(float * exposure, float * accumulate, float * kinetic);

            See Also:
            SetAccumulationCycleTime SetAcquisitionMode SetExposureTime SetHSSpeed SetKineticCycleTime SetMultiTrack SetNumberAccumulations SetNumberKinetics SetReadMode SetSingleTrack SetTriggerMode SetVSSpeed

        """
        cexposure = c_float()
        caccumulate = c_float()
        ckinetic = c_float()
        ret = self.dll.GetAcquisitionTimings(
            byref(cexposure), byref(caccumulate), byref(ckinetic))
        return (ret, round(cexposure.value, 6), round(caccumulate.value, 6), round(ckinetic.value, 6))

    def GetAdjustedRingExposureTimes(self, inumTimes):
        """
            Description:
            This function will return the actual exposure times that the camera will use. There may be differences between requested exposures and the actual exposures.

            Synopsis:
            (ret, fptimes) = GetAdjustedRingExposureTimes(inumTimes)

            Inputs:
            inumTimes - inumTimesNumbers of times requested.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Success.
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - Invalid number of exposures requested
            fptimes - Pointer to an array large enough to hold inumTimes floats.

            C++ Equiv:
            unsigned int GetAdjustedRingExposureTimes(int inumTimes, float * fptimes);

            See Also:
            GetNumberRingExposureTimes SetRingExposureTimes

        """
        cinumTimes = c_int(inumTimes)
        cfptimes = (c_float * inumTimes)()
        ret = self.dll.GetAdjustedRingExposureTimes(cinumTimes, cfptimes)
        
        arr = np.ndarray((inumTimes, ), 'f', cfptimes, order='C')
        return (ret, arr)

    def GetAllDMAData(self, size):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, arr) = GetAllDMAData(size)

            Inputs:
            size -

            Outputs:
            ret - Function Return Code
            arr -

            C++ Equiv:
            unsigned int GetAllDMAData(at_32 * arr, long size);

        """
        carr = c_int()
        csize = c_int(size)
        ret = self.dll.GetAllDMAData(byref(carr), csize)
        return (ret, carr.value)

    def GetAmpDesc(self, index, length):
        """
            Description:
            This function will return a string with an amplifier description. The amplifier is selected using the index.
            The SDK has a string associated with each of its amplifiers.
            The maximum number of characters needed to store the amplifier descriptions is 21.
            The user has to specify the number of characters they wish to have returned to them from this function.

            Synopsis:
            (ret, name) = GetAmpDesc(index, length)

            Inputs:
            index - The amplifier index.
            length - The length of the user allocated character array.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Description returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - The amplifier index is not valid.
                DRV_P2INVALID - The desc pointer is null.
                DRV_P3INVALID - The length parameter is invalid (less than 1)
            name - A user allocated array of characters for storage of the description.

            C++ Equiv:
            unsigned int GetAmpDesc(int index, char * name, int length);

            See Also:
            GetNumberAmp

        """
        cindex = c_int(index)
        cname = create_string_buffer(length)
        clength = c_int(length)
        ret = self.dll.GetAmpDesc(cindex, cname, clength)

        arr = np.char.asarray(cname, unicode="utc=8")
        str1 = ""
        str1 = str1.join(arr)
        cname = str1

        return (ret, cname)

    def GetAmpMaxSpeed(self, index):
        """
            Description:
            This function will return the maximum available horizontal shift speed for the amplifier selected by the index parameter.

            Synopsis:
            (ret, speed) = GetAmpMaxSpeed(index)

            Inputs:
            index - amplifier index

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - The amplifier index is not valid
            speed - horizontal shift speed

            C++ Equiv:
            unsigned int GetAmpMaxSpeed(int index, float * speed);

            See Also:
            GetNumberAmp

        """
        cindex = c_int(index)
        cspeed = c_float()
        ret = self.dll.GetAmpMaxSpeed(cindex, byref(cspeed))
        return (ret, cspeed.value)

    def GetAvailableCameras(self):
        """
            Description:
            This function returns the total number of Andor cameras currently installed. It is possible to call this function before any of the cameras are initialized.

            Synopsis:
            (ret, totalCameras) = GetAvailableCameras()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of available cameras returned.
                DRV_GENERAL_ERRORS - An error occurred while obtaining the number of available cameras.
            totalCameras - the number of cameras currently installed

            C++ Equiv:
            unsigned int GetAvailableCameras(long * totalCameras);

            See Also:
            SetCurrentCamera GetCurrentCamera GetCameraHandle

        """
        ctotalCameras = c_int()
        ret = self.dll.GetAvailableCameras(byref(ctotalCameras))
        return (ret, ctotalCameras.value)

    def GetBackground(self, size):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, arr) = GetBackground(size)

            Inputs:
            size -

            Outputs:
            ret - Function Return Code
            arr -

            C++ Equiv:
            unsigned int GetBackground(at_32 * arr, long size);

        """
        carr = c_int()
        csize = c_int(size)
        ret = self.dll.GetBackground(byref(carr), csize)
        return (ret, carr.value)

    def GetBaselineClamp(self):
        """
            Description:
            This function returns the status of the baseline clamp functionality. With this feature enabled the baseline level of each scan in a kinetic series will be more consistent across the sequence.

            Synopsis:
            (ret, state) = GetBaselineClamp()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - Baseline Clamp not supported on this camera
                DRV_P1INVALID - State parameter was not zero or one.
            state - Baseline clamp functionality Enabled/Disabled:
                1 - Baseline Clamp Enabled
                0 - Baseline Clamp Disabled

            C++ Equiv:
            unsigned int GetBaselineClamp(int * state);

            See Also:
            SetBaselineClamp SetBaselineOffset

        """
        cstate = c_int()
        ret = self.dll.GetBaselineClamp(byref(cstate))
        return (ret, cstate.value)

    def GetBitDepth(self, channel):
        """
            Description:
            This function will retrieve the size in bits of the dynamic range for any available AD channel.

            Synopsis:
            (ret, depth) = GetBitDepth(channel)

            Inputs:
            channel - the AD channel.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Depth returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid channel
            depth - dynamic range in bits

            C++ Equiv:
            unsigned int GetBitDepth(int channel, int * depth);

            See Also:
            GetNumberADChannels SetADChannel

        """
        cchannel = c_int(channel)
        cdepth = c_int()
        ret = self.dll.GetBitDepth(cchannel, byref(cdepth))
        return (ret, cdepth.value)

    def GetBitsPerPixel(self, readoutIndex, index):
        """
            Description:
            This function will get the size in bits of the dynamic range for the current shift speed

            Synopsis:
            (ret, value) = GetBitsPerPixel(readoutIndex, index)

            Inputs:
            readoutIndex - the readout index
            index - the index to bit depth

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Depth returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_SUPPORTED - Variable bit depth not available for this camera
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - Invalid readout index
                DRV_P2INVALID - Invalid bit depth index
            value - the dynamic range in bits

            C++ Equiv:
            unsigned int GetBitsPerPixel(int readoutIndex, int index, int * value);

            See Also:
            SetHSSpeed SetADChannel GetCapabilities SetBitsPerPixel

            Note: This function will get the size in bits of the dynamic range for the current shift speed

        """
        creadoutIndex = c_int(readoutIndex)
        cindex = c_int(index)
        cvalue = c_int()
        ret = self.dll.GetBitsPerPixel(creadoutIndex, cindex, byref(cvalue))
        return (ret, cvalue.value)

    def GetCameraEventStatus(self):
        """
            Description:
            This function will return if the system is exposing or not.

            Synopsis:
            (ret, camStatus) = GetCameraEventStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Status returned
                DRV_NOT_INITIALIZED - System not initialized
            camStatus - The status of the firepulse will be returned that the firepulse is low:
                0 - Fire pulse low
                1 - Fire pulse high

            C++ Equiv:
            unsigned int GetCameraEventStatus(DWORD * camStatus);

            See Also:
            SetAcqStatusEvent SetPCIMode

            Note: This is only supported by the CCI23 card.

        """
        ccamStatus = ()
        ret = self.dll.GetCameraEventStatus(byref(ccamStatus))
        return (ret, ccamStatus.value)

    def GetCameraHandle(self, cameraIndex):
        """
            Description:
            This function returns the handle for the camera specified by cameraIndex.  When multiple Andor cameras are installed the handle of each camera must be retrieved in order to select a camera using the SetCurrentCamera function.
            The number of cameras can be obtained using the GetAvailableCameras function.

            Synopsis:
            (ret, cameraHandle) = GetCameraHandle(cameraIndex)

            Inputs:
            cameraIndex - index of any of the installed cameras. 0 to NumberCameras-1 where NumberCameras is the value returned by the GetAvailableCamerasGetAvailableCameras functionGetAvailableCamerasGetNumberVerticalSpeedsGetNumberHSSpeeds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Camera handle returned.
                DRV_P1INVALID - Invalid camera index.
            cameraHandle - handle of the camera.

            C++ Equiv:
            unsigned int GetCameraHandle(long cameraIndex, long * cameraHandle);

            See Also:
            SetCurrentCamera GetAvailableCameras GetCurrentCamera

        """
        ccameraIndex = c_int(cameraIndex)
        ccameraHandle = c_int()
        ret = self.dll.GetCameraHandle(ccameraIndex, byref(ccameraHandle))
        return (ret, ccameraHandle.value)

    def GetCameraInformation(self, index):
        """
            Description:
            This function will return information on a particular camera denoted by the index.

            Synopsis:
            (ret, information) = GetCameraInformation(index)

            Inputs:
            index - (reserved)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Driver status return
                DRV_VXDNOTINSTALLED - Driver not installed
                DRV_USBERROR - USB device error
            information - current state of camera:
                Bit:1 - USB camera present
                Bit:2 - All dlls loaded properly
                Bit:3 - Camera Initialized correctly

            C++ Equiv:
            unsigned int GetCameraInformation(int index, long * information);

            See Also:
            GetCameraHandle GetHeadModel GetCameraSerialNumber GetCapabilities

            Note: Only available in iDus. The index parameter is not used at present so should be set to 0. For any camera except the iDus The value of information following a call to this function will be zero.

        """
        cindex = c_int(index)
        cinformation = c_int()
        ret = self.dll.GetCameraInformation(cindex, byref(cinformation))
        return (ret, cinformation.value)

    def GetCameraSerialNumber(self):
        """
            Description:
            This function will retrieve camera's serial number.

            Synopsis:
            (ret, number) = GetCameraSerialNumber()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Serial Number returned.
                DRV_NOT_INITIALIZED - System not initialized.
            number - Serial Number.

            C++ Equiv:
            unsigned int GetCameraSerialNumber(int * number);

            See Also:
            GetCameraHandle GetHeadModel GetCameraInformation GetCapabilities

        """
        cnumber = c_int()
        ret = self.dll.GetCameraSerialNumber(byref(cnumber))
        return (ret, cnumber.value)

    def GetCapabilities(self):
        """
            Description:
            This function will fill in an AndorCapabilities structure with the capabilities associated with the connected camera.
            Before passing the address of an AndorCapabilites structure to the function the ulSize member of the structure
             should be set to the size of the structure. In C++  this can be done with the line:
            caps->ulSize = sizeof(AndorCapabilities);
            Individual capabilities are determined by examining certain bits and combinations of bits in the member variables of the AndorCapabilites structure.

            Synopsis:
            (ret, caps) = GetCapabilities()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_NOT_INITIALIZED - System not initialized
                DRV_SUCCESS - Capabilities returned.
                DRV_P1INVALID - Invalid caps parameter (i.e. NULL).
            caps - the capabilities structure to be filled in.

            C++ Equiv:
            unsigned int GetCapabilities(AndorCapabilities * caps);

            See Also:
            GetCameraHandle GetCameraSerialNumber GetHeadModel GetCameraInformation

        """
        ccaps = AndorCapabilities()
        ccaps.ulSize = sys.getsizeof(ccaps)
        ret = self.dll.GetCapabilities(byref(ccaps))
        return (ret, ccaps)

    def GetControllerCardModel(self):
        """
            Description:
            This function will retrieve the type of PCI controller card included in your system. This function is not applicable for USB systems. The maximum number of characters that can be returned from this function is 10.

            Synopsis:
            (ret, controllerCardModel) = GetControllerCardModel()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Name returned.
                DRV_NOT_INITIALIZED - System not initialized
            controllerCardModel - A user allocated array of characters for storage of the controller card model.

            C++ Equiv:
            unsigned int GetControllerCardModel(char * controllerCardModel);

            See Also:
            GetHeadModel GetCameraSerialNumber GetCameraInformation GetCapabilities

        """
        ccontrollerCardModel = create_string_buffer(10)
        ret = self.dll.GetControllerCardModel(ccontrollerCardModel)
        return (ret, ccontrollerCardModel)

    def GetCountConvertWavelengthRange(self):
        """
            Description:
            This function returns the valid wavelength range available in Count Convert mode.

            Synopsis:
            (ret, minval, maxval) = GetCountConvertWavelengthRange()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Count Convert wavelength set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Count Convert not supported on this camera
            minval - minimum wavelength permited.
            maxval - maximum wavelength permited.

            C++ Equiv:
            unsigned int GetCountConvertWavelengthRange(float * minval, float * maxval);

            See Also:
            GetCapabilities SetCountConvertMode SetCountConvertWavelength

        """
        cminval = c_float()
        cmaxval = c_float()
        ret = self.dll.GetCountConvertWavelengthRange(
            byref(cminval), byref(cmaxval))
        return (ret, cminval.value, cmaxval.value)

    def GetCurrentCamera(self):
        """
            Description:
            When multiple Andor cameras are installed this function returns the handle of the currently selected one.

            Synopsis:
            (ret, cameraHandle) = GetCurrentCamera()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Camera handle returned.
            cameraHandle - handle of the currently selected camera

            C++ Equiv:
            unsigned int GetCurrentCamera(long * cameraHandle);

            See Also:
            SetCurrentCamera GetAvailableCameras GetCameraHandle

        """
        ccameraHandle = c_int()
        ret = self.dll.GetCurrentCamera(byref(ccameraHandle))
        return (ret, ccameraHandle.value)

    def GetCurrentPreAmpGain(self, len):
        """
            Description:
            This function will return the currently active pre amp gain index and a string with its description.  The maximum number of characters needed to store the pre amp gain description is 30. The user has to specify the number of characters they wish to have returned to them from this function.

            Synopsis:
            (ret, index, name) = GetCurrentPreAmpGain(len)

            Inputs:
            len - The length of the user allocated character array.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Description returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - Invalid pointer (i.e. NULL)
                DRV_NOT_SUPPORTED - Function not supported with this camera
            index - current pre amp gain index
            name - A user allocated array of characters for storage of the description

            C++ Equiv:
            unsigned int GetCurrentPreAmpGain(int * index, char * name, int len);

            See Also:
            IsPreAmpGainAvailable GetNumberPreAmpGains SetPreAmpGain GetCapabilities

        """
        cindex = c_int()
        cname = create_string_buffer(len)
        clen = c_int(len)
        ret = self.dll.GetCurrentPreAmpGain(byref(cindex), byref(cname), clen)
        return (ret, cindex.value, cname.value)

    def GetCYMGShift(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, iXshift, iYShift) = GetCYMGShift()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            iXshift -
            iYShift -

            C++ Equiv:
            unsigned int GetCYMGShift(int * iXshift, int * iYShift);

        """
        ciXshift = c_int()
        ciYShift = c_int()
        ret = self.dll.GetCYMGShift(byref(ciXshift), byref(ciYShift))
        return (ret, ciXshift.value, ciYShift.value)

    def GetDDGExternalOutputEnabled(self, uiIndex):
        """
            Description:
            This function gets the current state of a selected external output.

            Synopsis:
            (ret, puiEnabled) = GetDDGExternalOutputEnabled(uiIndex)

            Inputs:
            uiIndex - index of external output.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with card.
                DRV_P2INVALID - Invalid external output index.
            puiEnabled - current state of external output (0 - Off, 1 - On).

            C++ Equiv:
            unsigned int GetDDGExternalOutputEnabled(at_u32 uiIndex, at_u32 * puiEnabled);

            See Also:
            GetCapabilities SetDDGExternalOutputEnabled SetDDGGateStep

            Note: Available on USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cpuiEnabled = c_uint()
        ret = self.dll.GetDDGExternalOutputEnabled(
            cuiIndex, byref(cpuiEnabled))
        return (ret, cpuiEnabled.value)

    def GetDDGExternalOutputPolarity(self, uiIndex):
        """
            Description:
            This function gets the current polarity of a selected external output.

            Synopsis:
            (ret, puiPolarity) = GetDDGExternalOutputPolarity(uiIndex)

            Inputs:
            uiIndex - index of external output.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Polarity returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid external output index.
            puiPolarity - current polarity of external output (0 - Positive, 1 - Negative).

            C++ Equiv:
            unsigned int GetDDGExternalOutputPolarity(at_u32 uiIndex, at_u32 * puiPolarity);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled SetDDGExternalOutputPolarity SetDDGGateStep

            Note: Available on USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cpuiPolarity = c_uint()
        ret = self.dll.GetDDGExternalOutputPolarity(
            cuiIndex, byref(cpuiPolarity))
        return (ret, cpuiPolarity.value)

    def GetDDGExternalOutputStepEnabled(self, uiIndex):
        """
            Description:
            Each external output has the option to track the gate step applied to the gater.  This function can be used to determine if this option is currently active.

            Synopsis:
            (ret, puiEnabled) = GetDDGExternalOutputStepEnabled(uiIndex)

            Inputs:
            uiIndex - index of external output.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid external output index.
            puiEnabled - current state of external output track step (0 - Off, 1 - On).

            C++ Equiv:
            unsigned int GetDDGExternalOutputStepEnabled(at_u32 uiIndex, at_u32 * puiEnabled);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled SetDDGExternalOutputStepEnabled SetDDGGateStep

            Note: Available on USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cpuiEnabled = c_uint()
        ret = self.dll.GetDDGExternalOutputStepEnabled(
            cuiIndex, byref(cpuiEnabled))
        return (ret, cpuiEnabled.value)

    def GetDDGExternalOutputTime(self, uiIndex):
        """
            Description:
            This function can be used to find the actual timings for a particular external output.

            Synopsis:
            (ret, puiDelay, puiWidth) = GetDDGExternalOutputTime(uiIndex)

            Inputs:
            uiIndex - index of external output.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid external output index.
                DRV_P3INVALID - Delay has invalid memory address.
            puiDelay - actual external output delay time in picoseconds.
            puiWidth - actual external output width time in picoseconds.

            C++ Equiv:
            unsigned int GetDDGExternalOutputTime(at_u32 uiIndex, at_u64 * puiDelay, at_u64 * puiWidth);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled SetDDGExternalOutputTime SetDDGGateStep

            Note: Available in USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cpuiDelay = c_ulonglong()
        cpuiWidth = c_ulonglong()
        ret = self.dll.GetDDGExternalOutputTime(
            cuiIndex, byref(cpuiDelay), byref(cpuiWidth))
        return (ret, cpuiDelay.value, cpuiWidth.value)

    def GetDDGGateTime(self):
        """
            Description:
            This function can be used to get the actual gate timings for a USB iStar.

            Synopsis:
            (ret, puiDelay, puiWidth) = GetDDGGateTime()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - USB iStar not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Delay has invalid memory address.
            puiDelay - gate delay time in picoseconds.
            puiWidth - gate width time in picoseconds.

            C++ Equiv:
            unsigned int GetDDGGateTime(at_u64 * puiDelay, at_u64 * puiWidth);

            See Also:
            GetCapabilities SetDDGGateTimeSetDDGGateStep

        """
        cpuiDelay = c_ulonglong()
        cpuiWidth = c_ulonglong()
        ret = self.dll.GetDDGGateTime(byref(cpuiDelay), byref(cpuiWidth))
        return (ret, cpuiDelay.value, cpuiWidth.value)

    def GetDDGInsertionDelay(self):
        """
            Description:
            This function gets the current state of the insertion delay.

            Synopsis:
            (ret, piState) = GetDDGInsertionDelay()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Insertion delay state returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - Insertion delay not supported.
                DRV_ERROR_ACK - Unable to communicate with system.
            piState - current state of the insertion delay option (0 - Normal, 1 - Ultra Fast).

            C++ Equiv:
            unsigned int GetDDGInsertionDelay(int * piState);

            See Also:
            GetCapabilities SetDDGInsertionDelay SetDDGIntelligate

        """
        cpiState = c_int()
        ret = self.dll.GetDDGInsertionDelay(byref(cpiState))
        return (ret, cpiState.value)

    def GetDDGIntelligate(self):
        """
            Description:
            This function gets the current state of intelligate.

            Synopsis:
            (ret, piState) = GetDDGIntelligate()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - intelligate state returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - intelligate not supported.
                DRV_ERROR_ACK - Unable to communicate with system.
            piState - current state of the intelligate option (0 - Off, 1 - On).

            C++ Equiv:
            unsigned int GetDDGIntelligate(int * piState);

            See Also:
            GetCapabilities SetDDGIntelligate SetDDGInsertionDelay

        """
        cpiState = c_int()
        ret = self.dll.GetDDGIntelligate(byref(cpiState))
        return (ret, cpiState.value)

    def GetDDGIOC(self):
        """
            Description:
            This function gets the current state of the integrate on chip (IOC) option.

            Synopsis:
            (ret, state) = GetDDGIOC()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IOC state returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - state has invalid memory address.
            state - current state of the IOC option (0 - Off, 1 - On).

            C++ Equiv:
            unsigned int GetDDGIOC(int * state);

            See Also:
            GetCapabilities SetDDGIOC SetDDGIOCFrequency

        """
        cstate = c_int()
        ret = self.dll.GetDDGIOC(byref(cstate))
        return (ret, cstate.value)

    def GetDDGIOCFrequency(self):
        """
            Description:
            This function can be used to return the actual IOC frequency that will be triggered. It should only be called once all the conditions of the experiment have been defined.

            Synopsis:
            (ret, frequency) = GetDDGIOCFrequency()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ERROR_ACK - Unable to communicate with card
            frequency - the frequency of integrate on chip pulses triggered within the fire pulse.

            C++ Equiv:
            unsigned int GetDDGIOCFrequency(double * frequency);

            See Also:
            GetCapabilities SetDDGIOCFrequency SetDDGIOCNumber GetDDGIOCNumber GetDDGIOCPulses SetDDGIOC SetDDGIOCFrequency

        """
        cfrequency = c_double()
        ret = self.dll.GetDDGIOCFrequency(byref(cfrequency))
        return (ret, cfrequency.value)

    def GetDDGIOCNumber(self):
        """
            Description:
            This function can be used to return the actual number of pulses that will be triggered. It should only be called once all the conditions of the experiment have been defined.

            Synopsis:
            (ret, numberPulses) = GetDDGIOCNumber()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ERROR_ACK - Unable to communicate with card
            numberPulses - the number of integrate on chip pulses triggered within the fire pulse.

            C++ Equiv:
            unsigned int GetDDGIOCNumber(unsigned long * numberPulses);

            See Also:
            GetCapabilities SetDDGIOCFrequency GetDDGIOCFrequency SetDDGIOCNumber GetDDGIOCPulses SetDDGIOC SetDDGIOCFrequency

        """
        cnumberPulses = c_ulong()
        ret = self.dll.GetDDGIOCNumber(byref(cnumberPulses))
        return (ret, cnumberPulses.value)

    def GetDDGIOCNumberRequested(self):
        """
            Description:
            This function can be used to return the number of pulses that were requested by the user.

            Synopsis:
            (ret, pulses) = GetDDGIOCNumberRequested()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - pulses has invalid memory address.
            pulses - the number of integrate on chip pulses requested.

            C++ Equiv:
            unsigned int GetDDGIOCNumberRequested(at_u32 * pulses);

            See Also:
            GetCapabilities SetDDGIOCNumber SetDDGIOC SetDDGIOCFrequency

        """
        cpulses = c_uint()
        ret = self.dll.GetDDGIOCNumberRequested(byref(cpulses))
        return (ret, cpulses.value)

    def GetDDGIOCPeriod(self):
        """
            Description:
            This function can be used to return the actual IOC period that will be triggered. It should only be called once all the conditions of the experiment have been defined.

            Synopsis:
            (ret, period) = GetDDGIOCPeriod()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IOC period returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - period has invalid memory address.
            period - the period of integrate on chip pulses triggered within the fire pulse.

            C++ Equiv:
            unsigned int GetDDGIOCPeriod(at_u64 * period);

            See Also:
            GetCapabilities SetDDGIOC SetDDGIOCPeriod SetDDGIOCFrequency

        """
        cperiod = c_ulonglong()
        ret = self.dll.GetDDGIOCPeriod(byref(cperiod))
        return (ret, cperiod.value)

    def GetDDGIOCPulses(self):
        """
            Description:
            This function can be used to calculate the number of pulses that will be triggered with the given exposure time, readout mode, acquisition mode and integrate on chip frequency. It should only be called once all the conditions of the experiment have been defined.

            Synopsis:
            (ret, pulses) = GetDDGIOCPulses()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
            pulses - the number of integrate on chip pulses triggered within the fire pulse.

            C++ Equiv:
            unsigned int GetDDGIOCPulses(int * pulses);

            See Also:
            GetCapabilities SetDDGIOCFrequency GetDDGIOCFrequency SetDDGIOCNumber GetDDGIOCNumber SetDDGIOC SetDDGIOCFrequency

        """
        cpulses = c_int()
        ret = self.dll.GetDDGIOCPulses(byref(cpulses))
        return (ret, cpulses.value)

    def GetDDGIOCTrigger(self):
        """
            Description:
            function can be used to retrieve the active IOC trigger.
            at_u32* trigger: active IOC trigger (0 - Fire pulse, 1 - External trigger).
            at_u32 int
            DRV_SUCCESS
            DRV_NOT_INITIALIZED
            DRV_NOT_SUPPORTED
            DRV_ACQUIRING
            DRV_ERROR_ACK
            DRV_P1INVALID
            IOC trigger returned.
            System not initialized.
            IOC not supported.
            Acquisition in progress.
            Unable to communicate with system.
            Invalid trigger.
            See also
            GetCapabilities SetDDGIOC SetDDGIOCFrequency SetDDGIOCTrigger SetTriggerMode
            GetDDGLiteControlByte
            GetDDGLiteControlByte int WINAPI GetDDGLiteControlByte(AT_DDGLiteChannelId channel, unsigned char * control)
            Description
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, trigger) = GetDDGIOCTrigger()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            trigger -

            C++ Equiv:
            unsigned int GetDDGIOCTrigger(at_u32 * trigger);

        """
        ctrigger = c_uint()
        ret = self.dll.GetDDGIOCTrigger(byref(ctrigger))
        return (ret, ctrigger.value)

    def GetDDGLiteControlByte(self, channel):
        """
            Description:
            THIS FUNCTION IS RESERVED

            Synopsis:
            (ret, control) = GetDDGLiteControlByte(channel)

            Inputs:
            channel -

            Outputs:
            ret - Function Return Code
            control -

            C++ Equiv:
            unsigned int GetDDGLiteControlByte(AT_DDGLiteChannelId channel, unsigned char * control);

        """
        cchannel = (channel)
        ccontrol = c_ubyte()
        ret = self.dll.GetDDGLiteControlByte(cchannel, byref(ccontrol))
        return (ret, ccontrol.value)

    def GetDDGLiteGlobalControlByte(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, control) = GetDDGLiteGlobalControlByte()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            control -

            C++ Equiv:
            unsigned int GetDDGLiteGlobalControlByte(unsigned char * control);

        """
        ccontrol = c_ubyte()
        ret = self.dll.GetDDGLiteGlobalControlByte(byref(ccontrol))
        return (ret, ccontrol.value)

    def GetDDGLiteInitialDelay(self, channel):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, fDelay) = GetDDGLiteInitialDelay(channel)

            Inputs:
            channel -

            Outputs:
            ret - Function Return Code
            fDelay -

            C++ Equiv:
            unsigned int GetDDGLiteInitialDelay(AT_DDGLiteChannelId channel, float * fDelay);

        """
        cchannel = (channel)
        cfDelay = c_float()
        ret = self.dll.GetDDGLiteInitialDelay(cchannel, byref(cfDelay))
        return (ret, cfDelay.value)

    def GetDDGLiteInterPulseDelay(self, channel):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, fDelay) = GetDDGLiteInterPulseDelay(channel)

            Inputs:
            channel -

            Outputs:
            ret - Function Return Code
            fDelay -

            C++ Equiv:
            unsigned int GetDDGLiteInterPulseDelay(AT_DDGLiteChannelId channel, float * fDelay);

        """
        cchannel = (channel)
        cfDelay = c_float()
        ret = self.dll.GetDDGLiteInterPulseDelay(cchannel, byref(cfDelay))
        return (ret, cfDelay.value)

    def GetDDGLitePulsesPerExposure(self, channel):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, ui32Pulses) = GetDDGLitePulsesPerExposure(channel)

            Inputs:
            channel -

            Outputs:
            ret - Function Return Code
            ui32Pulses -

            C++ Equiv:
            unsigned int GetDDGLitePulsesPerExposure(AT_DDGLiteChannelId channel, at_u32 * ui32Pulses);

        """
        cchannel = (channel)
        cui32Pulses = c_uint()
        ret = self.dll.GetDDGLitePulsesPerExposure(
            cchannel, byref(cui32Pulses))
        return (ret, cui32Pulses.value)

    def GetDDGLitePulseWidth(self, channel):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, fWidth) = GetDDGLitePulseWidth(channel)

            Inputs:
            channel -

            Outputs:
            ret - Function Return Code
            fWidth -

            C++ Equiv:
            unsigned int GetDDGLitePulseWidth(AT_DDGLiteChannelId channel, float * fWidth);

        """
        cchannel = (channel)
        cfWidth = c_float()
        ret = self.dll.GetDDGLitePulseWidth(cchannel, byref(cfWidth))
        return (ret, cfWidth.value)

    def GetDDGOpticalWidthEnabled(self):
        """
            Description:
            This function can be used to check whether optical gate widths are being used.

            Synopsis:
            (ret, puiEnabled) = GetDDGOpticalWidthEnabled()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Optical gate width not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
            puiEnabled - optical gate width option (0 - Off, 1 - On).

            C++ Equiv:
            unsigned int GetDDGOpticalWidthEnabled(at_u32 * puiEnabled);

            See Also:
            GetCapabilities GetDDGTTLGateWidth

        """
        cpuiEnabled = c_uint()
        ret = self.dll.GetDDGOpticalWidthEnabled(byref(cpuiEnabled))
        return (ret, cpuiEnabled.value)

    def GetDDGPulse(self, wid, resolution):
        """
            Description:
            This function attempts to find a laser pulse in a user-defined region with a given resolution. The values returned will provide an estimation of the location of the pulse.

            Synopsis:
            (ret, Delay, Width) = GetDDGPulse(wid, resolution)

            Inputs:
            wid - the time in picoseconds of the region to be searched.
            resolution - the minimum gate pulse used to locate the laser.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Location returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
            Delay - the approximate start of the laser pulse.
            Width - the pulse width, which encapsulated the laser pulse.

            C++ Equiv:
            unsigned int GetDDGPulse(double wid, double resolution, double * Delay, double * Width);

            Note: Available in iStar.

        """
        cwid = c_double(wid)
        cresolution = c_double(resolution)
        cDelay = c_double()
        cWidth = c_double()
        ret = self.dll.GetDDGPulse(
            cwid, cresolution, byref(cDelay), byref(cWidth))
        return (ret, cDelay.value, cWidth.value)

    def GetDDGStepCoefficients(self, mode):
        """
            Description:
            This function will return the coefficients for a particular gate step mode.

            Synopsis:
            (ret, p1, p2) = GetDDGStepCoefficients(mode)

            Inputs:
            mode - the gate step mode.:
                0 - constant.
                1 - exponential.
                2 - logarithmic.
                3 - linear.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step coefficients returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Gate step not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - Gate step mode invalid.
                DRV_P2_INVALID - p1 has invalid memory address.
                DRV_P3_INVALID - p2 has invalid memory address.
            p1 - First coefficient
            p2 - Second coefficient

            C++ Equiv:
            unsigned int GetDDGStepCoefficients(at_u32 mode, double * p1, double * p2);

            See Also:
            StartAcquisition SetDDGStepMode SetDDGStepCoefficients

        """
        cmode = c_uint(mode)
        cp1 = c_double()
        cp2 = c_double()
        ret = self.dll.GetDDGStepCoefficients(cmode, byref(cp1), byref(cp2))
        return (ret, cp1.value, cp2.value)

    def GetDDGStepMode(self):
        """
            Description:
            This function will return the current gate step mode.

            Synopsis:
            (ret, mode) = GetDDGStepMode()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Gate step not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - mode has invalid memory address.
            mode - the gate step mode.:
                0 - constant.
                1 - exponential.
                2 - logarithmic.
                3 - linear.
                100 - off.

            C++ Equiv:
            unsigned int GetDDGStepMode(at_u32 * mode);

            See Also:
            StartAcquisition SetDDGStepMode SetDDGStepCoefficients GetDDGStepCoefficients

        """
        cmode = c_uint()
        ret = self.dll.GetDDGStepMode(byref(cmode))
        return (ret, cmode.value)

    def GetDDGTTLGateWidth(self, opticalWidth):
        """
            Description:
            This function can be used to get the TTL gate width which corresponds to a particular optical gate width.

            Synopsis:
            (ret, ttlWidth) = GetDDGTTLGateWidth(opticalWidth)

            Inputs:
            opticalWidth - optical gate width in picoseconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Optical gate width not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P2_INVALID - Unable to communicate with system.
            ttlWidth - TTL gate width in picoseconds.

            C++ Equiv:
            unsigned int GetDDGTTLGateWidth(at_u64 opticalWidth, at_u64 * ttlWidth);

            See Also:
            GetCapabilities SetDDGOpticalWidthEnabled SetDDGGateStep

        """
        copticalWidth = c_ulonglong(opticalWidth)
        cttlWidth = c_ulonglong()
        ret = self.dll.GetDDGTTLGateWidth(copticalWidth, byref(cttlWidth))
        return (ret, cttlWidth.value)

    def GetDDGWidthStepCoefficients(self, mode):
        """
            Description:
            This function will return the coefficients for a particular gate width step mode.

            Synopsis:
            (ret, p1, p2) = GetDDGWidthStepCoefficients(mode)

            Inputs:
            mode - the gate step mode.:
                0 - constant.
                1 - exponential.
                2 - logarithmic.
                3 - linear.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step coefficients returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Gate step not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - Gate step mode invalid.
                DRV_P2_INVALID - p1 has invalid memory address.
                DRV_P3_INVALID - p2 has invalid memory address.
            p1 - The first coefficient.
            p2 - The second coefficient.

            C++ Equiv:
            unsigned int GetDDGWidthStepCoefficients(at_u32 mode, double * p1, double * p2);

            See Also:
            SetDDGWidthStepCoefficients SetDDGWidthStepMode GetDDGWidthStepMode

        """
        cmode = c_uint(mode)
        cp1 = c_double()
        cp2 = c_double()
        ret = self.dll.GetDDGWidthStepCoefficients(
            cmode, byref(cp1), byref(cp2))
        return (ret, cp1.value, cp2.value)

    def GetDDGWidthStepMode(self):
        """
            Description:
            This function will return the current gate width step mode.

            Synopsis:
            (ret, mode) = GetDDGWidthStepMode()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Gate step not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - mode has invalid memory address.
            mode - the gate step mode.:
                0 - constant.
                1 - exponential.
                2 - logarithmic.
                3 - linear.
                100 - off.

            C++ Equiv:
            unsigned int GetDDGWidthStepMode(at_u32 * mode);

            See Also:
            SetDDGWidthStepCoefficients SetDDGWidthStepMode GetDDGWidthStepCoefficients StartAcquisition

        """
        cmode = c_uint()
        ret = self.dll.GetDDGWidthStepMode(byref(cmode))
        return (ret, cmode.value)

    def GetDetector(self):
        """
            Description:
            This function returns the size of the detector in pixels. The horizontal axis is taken to be the axis parallel to the readout register.

            Synopsis:
            (ret, xpixels, ypixels) = GetDetector()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Detector size returned.
                DRV_NOT_INITIALIZED - System not initialized.
            xpixels - number of horizontal pixels.
            ypixels - number of vertical pixels.

            C++ Equiv:
            unsigned int GetDetector(int * xpixels, int * ypixels);

        """
        cxpixels = c_int()
        cypixels = c_int()
        ret = self.dll.GetDetector(byref(cxpixels), byref(cypixels))
        return (ret, cxpixels.value, cypixels.value)

    def GetDICameraInfo(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, info) = GetDICameraInfo()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            info -

            C++ Equiv:
            unsigned int GetDICameraInfo(void * info);

        """
        cinfo = c_void()
        ret = self.dll.GetDICameraInfo(byref(cinfo))
        return (ret, cinfo.value)

    def GetDualExposureTimes(self):
        """
            Description:
            This function will return the current valid acquisition timing information for dual exposure mode.  This mode is only available for certain sensors in run till abort mode, external trigger, full image.

            Synopsis:
            (ret, exposure1, exposure2) = GetDualExposureTimes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized. .
                DRV_NOT_SUPPORTED - Dual exposure mode not supported on this camera.
                DRV_NOT_AVAILABLE - Dual exposure mode not configured correctly.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - exposure1 has invalid memory address.
                DRV_P2INVALID - exposure2 has invalid memory address.
            exposure1 - valid exposure time in seconds for each odd numbered frame.
            exposure2 - valid exposure time in seconds for each even numbered frame.

            C++ Equiv:
            unsigned int GetDualExposureTimes(float * exposure1, float * exposure2);

            See Also:
            GetCapabilities SetDualExposureMode SetDualExposureTimes

        """
        cexposure1 = c_float()
        cexposure2 = c_float()
        ret = self.dll.GetDualExposureTimes(
            byref(cexposure1), byref(cexposure2))
        return (ret, cexposure1.value, cexposure2.value)

    def GetEMAdvanced(self):
        """
            Description:
            Returns the current Advanced gain setting.

            Synopsis:
            (ret, state) = GetEMAdvanced()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Advanced state returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - state has invalid memory address.
            state - current EM advanced gain setting

            C++ Equiv:
            unsigned int GetEMAdvanced(int * state);

        """
        cstate = c_int()
        ret = self.dll.GetEMAdvanced(byref(cstate))
        return (ret, cstate.value)

    def GetEMCCDGain(self):
        """
            Description:
            Returns the current gain setting. The meaning of the value returned depends on the EM Gain mode.

            Synopsis:
            (ret, gain) = GetEMCCDGain()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gain returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
            gain - current EM gain setting

            C++ Equiv:
            unsigned int GetEMCCDGain(int * gain);

        """
        cgain = c_int()
        ret = self.dll.GetEMCCDGain(byref(cgain))
        return (ret, cgain.value)

    def GetEMGainRange(self):
        """
            Description:
            Returns the minimum and maximum values of the current selected EM Gain mode and temperature of the sensor.

            Synopsis:
            (ret, low, high) = GetEMGainRange()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gain range returned.
                DRV_NOT_INITIALIZED - System not initialized.
            low - lowest gain setting
            high - highest gain setting

            C++ Equiv:
            unsigned int GetEMGainRange(int * low, int * high);

        """
        clow = c_int()
        chigh = c_int()
        ret = self.dll.GetEMGainRange(byref(clow), byref(chigh))
        return (ret, clow.value, chigh.value)

    def GetESDEventStatus(self):
        """
            Description:
            This function will return if a USB reset has occurred.

            Synopsis:
            (ret, camStatus) = GetESDEventStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Status returned
                DRV_NOT_INITIALIZED - System not initialized
            camStatus - The USB communications status:
                0 - Normal
                1 - Reset occurred

            C++ Equiv:
            unsigned int GetESDEventStatus(DWORD * camStatus);

            See Also:
            SetDriverEvent GetCameraEventStatus

        """
        ccamStatus = ()
        ret = self.dll.GetESDEventStatus(byref(ccamStatus))
        return (ret, ccamStatus.value)

    def GetExternalTriggerTermination(self):
        """
            Description:
            This function can be used to get the current external trigger termination mode.

            Synopsis:
            (ret, puiTermination) = GetExternalTriggerTermination()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Termination returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Trigger termination not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
            puiTermination - trigger termination option.:
                0 - 50 ohm.
                1 - hi-Z.

            C++ Equiv:
            unsigned int GetExternalTriggerTermination(at_u32 * puiTermination);

            See Also:
            GetCapabilities SetExternalTriggerTermination

        """
        cpuiTermination = c_uint()
        ret = self.dll.GetExternalTriggerTermination(byref(cpuiTermination))
        return (ret, cpuiTermination.value)

    def GetFastestRecommendedVSSpeed(self):
        """
            Description:
            As your Andor SDK system may be capable of operating at more than one vertical shift speed this function will return the fastest recommended speed available.  The very high readout speeds, may require an increase in the amplitude of the Vertical Clock Voltage using SetVSAmplitudeSetVSAmplitude.  This function returns the fastest speed which does not require the Vertical Clock Voltage to be adjusted.  The values returned are the vertical shift speed index and the actual speed in microseconds per pixel shift.

            Synopsis:
            (ret, index, speed) = GetFastestRecommendedVSSpeed()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            index - index of the fastest recommended vertical shift speed
            speed - speed in microseconds per pixel shift.

            C++ Equiv:
            unsigned int GetFastestRecommendedVSSpeed(int * index, float * speed);

            See Also:
            GetVSSpeed GetNumberVSSpeeds SetVSSpeed

        """
        cindex = c_int()
        cspeed = c_float()
        ret = self.dll.GetFastestRecommendedVSSpeed(
            byref(cindex), byref(cspeed))
        return (ret, cindex.value, round(cspeed.value, 6))

    def GetFIFOUsage(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, FIFOusage) = GetFIFOUsage()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            FIFOusage -

            C++ Equiv:
            unsigned int GetFIFOUsage(int * FIFOusage);

        """
        cFIFOusage = c_int()
        ret = self.dll.GetFIFOUsage(byref(cFIFOusage))
        return (ret, cFIFOusage.value)

    def GetFilterMode(self):
        """
            Description:
            This function returns the current state of the cosmic ray filtering mode.

            Synopsis:
            (ret, mode) = GetFilterMode()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Filter mode returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            mode - current state of filter:
                0 - OFF
                2 - ON

            C++ Equiv:
            unsigned int GetFilterMode(int * mode);

            See Also:
            SetFilterMode

        """
        cmode = c_int()
        ret = self.dll.GetFilterMode(byref(cmode))
        return (ret, cmode.value)

    def GetFKExposureTime(self):
        """
            Description:
            This function will return the current "valid" exposure time for a fast kinetics acquisition. This function should be used after all the acquisitions settings have been set, i.e. SetFastKineticsSetFastKinetics and SetFKVShiftSpeedSetFKVShiftSpeed. The value returned is the actual time used in subsequent acquisitions.

            Synopsis:
            (ret, time) = GetFKExposureTime()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timing information returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_INVALID_MODE - Fast kinetics is not available.
            time - valid exposure time in seconds

            C++ Equiv:
            unsigned int GetFKExposureTime(float * time);

            See Also:
            SetFastKinetics SetFKVShiftSpeed

        """
        ctime = c_float()
        ret = self.dll.GetFKExposureTime(byref(ctime))
        return (ret, ctime.value)

    def GetFKVShiftSpeed(self, index):
        """
            Description:
            Deprecated see Note:
            As your Andor SDK system is capable of operating at more than one fast kinetics vertical shift speed this function will return the actual speeds available. The value returned is in microseconds per pixel shift.

            Synopsis:
            (ret, speed) = GetFKVShiftSpeed(index)

            Inputs:
            index - speed required:
                0 - to GetNumberFKVShiftSpeedsGetNumberFKVShiftSpeeds()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
            speed - speed in micro-seconds per pixel shift

            C++ Equiv:
            unsigned int GetFKVShiftSpeed(int index, int * speed); // deprecated

            See Also:
            GetNumberFKVShiftSpeeds SetFKVShiftSpeed

            Note: Deprecated by GetFKVShiftSpeedFGetNumberHSSpeeds

        """
        cindex = c_int(index)
        cspeed = c_int()
        ret = self.dll.GetFKVShiftSpeed(cindex, byref(cspeed))
        return (ret, cspeed.value)

    def GetFKVShiftSpeedF(self, index):
        """
            Description:
            As your Andor system is capable of operating at more than one fast kinetics vertical shift speed this function will return the actual speeds available. The value returned is in microseconds per pixel shift.

            Synopsis:
            (ret, speed) = GetFKVShiftSpeedF(index)

            Inputs:
            index - speed required:
                0 - to GetNumberFKVShiftSpeedsGetNumberFKVShiftSpeeds()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
            speed - speed in micro-seconds per pixel shift

            C++ Equiv:
            unsigned int GetFKVShiftSpeedF(int index, float * speed);

            See Also:
            GetNumberFKVShiftSpeeds SetFKVShiftSpeed

            Note: Only available if camera is Classic or iStar.

        """
        cindex = c_int(index)
        cspeed = c_float()
        ret = self.dll.GetFKVShiftSpeedF(cindex, byref(cspeed))
        return (ret, cspeed.value)

    def GetFrontEndStatus(self):
        """
            Description:
            This function will return if the Front End cooler has overheated.

            Synopsis:
            (ret, piFlag) = GetFrontEndStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Front End cooler not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with card.
            piFlag - The status of the front end cooler:
                0 - Normal
                1 - Tripped

            C++ Equiv:
            unsigned int GetFrontEndStatus(int * piFlag);

            See Also:
            SetFrontEndEvent

        """
        cpiFlag = c_int()
        ret = self.dll.GetFrontEndStatus(byref(cpiFlag))
        return (ret, cpiFlag.value)

    def GetGateMode(self):
        """
            Description:
            Allows the user to get the current photocathode gating mode.

            Synopsis:
            (ret, piGatemode) = GetGateMode()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gating mode accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - gatemode has invalid memory address.
            piGatemode - the gate mode.:
                0 - Fire ANDed with the Gate input.
                1 - Gating controlled from Fire pulse only.
                2 - Gating controlled from SMB Gate input only.
                3 - Gating ON continuously.
                4 - Gating OFF continuously.
                5 - Gate using DDG

            C++ Equiv:
            unsigned int GetGateMode(int * piGatemode);

            See Also:
            GetCapabilities SetGateMode

        """
        cpiGatemode = c_int()
        ret = self.dll.GetGateMode(byref(cpiGatemode))
        return (ret, cpiGatemode.value)

    def GetHardwareVersion(self):
        """
            Description:
            This function returns the Hardware version information.

            Synopsis:
            (ret, PCB, Decode, dummy1, dummy2, CameraFirmwareVersion, CameraFirmwareBuild) = GetHardwareVersion()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Version information returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
            PCB - Plug-in card version
            Decode - Flex 10K file version
            dummy1 -
            dummy2 -
            CameraFirmwareVersion - Version number of camera firmware
            CameraFirmwareBuild - Build number of camera firmware

            C++ Equiv:
            unsigned int GetHardwareVersion(unsigned int * PCB, unsigned int * Decode, unsigned int * dummy1, unsigned int * dummy2, unsigned int * CameraFirmwareVersion, unsigned int * CameraFirmwareBuild);

        """
        cPCB = c_uint()
        cDecode = c_uint()
        cdummy1 = c_uint()
        cdummy2 = c_uint()
        cCameraFirmwareVersion = c_uint()
        cCameraFirmwareBuild = c_uint()
        ret = self.dll.GetHardwareVersion(byref(cPCB), byref(cDecode), byref(cdummy1), byref(
            cdummy2), byref(cCameraFirmwareVersion), byref(cCameraFirmwareBuild))
        return (ret, cPCB.value, cDecode.value, cdummy1.value, cdummy2.value, cCameraFirmwareVersion.value, cCameraFirmwareBuild.value)

    def GetHeadModel(self):
        """
            Description:
            This function will retrieve the type of CCD attached to your system.

            Synopsis:
            (ret, name) = GetHeadModel()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Name returned.
                DRV_NOT_INITIALIZED - System not initialized.
            name - A user allocated array of characters for storage of the Head Model. This should be declared as size MAX_PATH.

            C++ Equiv:
            unsigned int GetHeadModel(char * name);

        """
        cname = create_string_buffer(MAX_PATH)
        ret = self.dll.GetHeadModel(cname)
        return (ret, cname.value)

    def GetHorizontalSpeed(self, index):
        """
            Description:
            Deprecated see Note:
            As your Andor system is capable of operating at more than one horizontal shift speed this function will return the actual speeds available. The value returned is in microseconds per pixel shift.

            Synopsis:
            (ret, speed) = GetHorizontalSpeed(index)

            Inputs:
            index - speed required, 0 to NumberSpeeds-1, where NumberSpeeds is the parameter returned by GetNumberHorizontalSpeedsGetNumberHorizontalSpeeds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
            speed - speed in micro-seconds per pixel shift

            C++ Equiv:
            unsigned int GetHorizontalSpeed(int index, int * speed); // deprecated

            See Also:
            GetNumberHorizontalSpeeds SetHorizontalSpeed

            Note: Deprecated by GetHSSpeedGetNumberHSSpeeds

        """
        cindex = c_int(index)
        cspeed = c_int()
        ret = self.dll.GetHorizontalSpeed(cindex, byref(cspeed))
        return (ret, cspeed.value)

    def GetHSSpeed(self, channel, typ, index):
        """
            Description:
            As your Andor system is capable of operating at more than one horizontal shift speed this function will return the actual speeds available.
            The value returned is in MHz.

            Synopsis:
            (ret, speed) = GetHSSpeed(channel, typ, index)

            Inputs:
            channel - the AD channel.
            typ - output amplification.:
                0 - electron multiplication/Conventional(clara).
                1 - conventional/Extended NIR Mode(clara).
            index - speed required Valid values: 0 to NumberSpeeds-1, where NumberSpeeds is value returned in first parameter after a call to GetNumberHSSpeeds().

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid channel.
                DRV_P2INVALID - Invalid horizontal read mode
                DRV_P3INVALID - Invalid index
            speed - speed in in MHz.

            C++ Equiv:
            unsigned int GetHSSpeed(int channel, int typ, int index, float * speed);

            See Also:
            GetNumberHSSpeeds SetHSSpeed

            Note: The speed is returned in microseconds per pixel shift for iStar and Classic systems.

        """
        cchannel = c_int(channel)
        ctyp = c_int(typ)
        cindex = c_int(index)
        cspeed = c_float()
        ret = self.dll.GetHSSpeed(cchannel, ctyp, cindex, byref(cspeed))
        return (ret, round(cspeed.value, 6))

    def GetHVflag(self):
        """
            Description:
            This function will retrieve the High Voltage flag from your USB iStar intensifier. A 0 value indicates that the high voltage is abnormal.

            Synopsis:
            (ret, bFlag) = GetHVflag()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - HV flag returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_AVAILABLE - Not a USB iStar.
            bFlag - pointer to High Voltage flag.

            C++ Equiv:
            unsigned int GetHVflag(int * bFlag);

            Note: Available only on USB iStar.

        """
        cbFlag = c_int()
        ret = self.dll.GetHVflag(byref(cbFlag))
        return (ret, cbFlag.value)

    def GetID(self, devNum):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, id) = GetID(devNum)

            Inputs:
            devNum -

            Outputs:
            ret - Function Return Code
            id -

            C++ Equiv:
            unsigned int GetID(int devNum, int * id);

        """
        cdevNum = c_int(devNum)
        cid = c_int()
        ret = self.dll.GetID(cdevNum, byref(cid))
        return (ret, cid.value)

    def GetImageFlip(self):
        """
            Description:
            This function will obtain whether the acquired data output is flipped in either the horizontal or vertical direction.

            Synopsis:
            (ret, iHFlip, iVFlip) = GetImageFlip()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - HFlip parameter invalid.
                DRV_P2INVALID - VFlip parameter invalid
            iHFlip - Gets horizontal flipping.
            iVFlip - Gets vertical flipping.:
                1 - Flipping Enabled
                0 - Flipping Disabled

            C++ Equiv:
            unsigned int GetImageFlip(int * iHFlip, int * iVFlip);

            See Also:
            SetImageRotate SetImageFlip

        """
        ciHFlip = c_int()
        ciVFlip = c_int()
        ret = self.dll.GetImageFlip(byref(ciHFlip), byref(ciVFlip))
        return (ret, ciHFlip.value, ciVFlip.value)

    def GetImageRotate(self):
        """
            Description:
            This function will obtain whether the acquired data output is rotated in any direction.

            Synopsis:
            (ret, iRotate) = GetImageRotate()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Rotate parameter invalid.
            iRotate - Rotation setting:
                0 - - No rotation
                1 - - Rotate 90 degrees clockwise
                2 - - Rotate 90 degrees anti-clockwise

            C++ Equiv:
            unsigned int GetImageRotate(int * iRotate);

            See Also:
            SetImageFlip SetImageRotate SetReadMode

        """
        ciRotate = c_int()
        ret = self.dll.GetImageRotate(byref(ciRotate))
        return (ret, ciRotate.value)

    def GetImages(self, first, last, size):
        """
            Description:
            This function will update the data array with the specified series of images from the circular buffer.
            If the specified series is out of range (i.e. the images have been overwritten or have not yet been acquired then an error will be returned.)

            Synopsis:
            (ret, arr, validfirst, validlast) = GetImages(first, last, size)

            Inputs:
            first - index of first image in buffer to retrieve.
            last - index of last image in buffer to retrieve.
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Images have been copied into array.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_GENERAL_ERRORS - The series is out of range.
                DRV_P3INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.
            validfirst - index of the first valid image.
            validlast - index of the last valid image.

            C++ Equiv:
            unsigned int GetImages(long first, long last, at_32 * arr, long size, long * validfirst, long * validlast);

            See Also:
            GetImages16 GetNumberNewImages

        """
        cfirst = c_int(first)
        clast = c_int(last)
        carr = (c_int * size)()
        csize = c_int(size)
        cvalidfirst = c_int()
        cvalidlast = c_int()
        ret = self.dll.GetImages(cfirst, clast, carr, csize, byref(
            cvalidfirst), byref(cvalidlast))
        return (ret, np.ctypeslib.as_array(carr), cvalidfirst.value, cvalidlast.value)

    def GetImages16(self, first, last, size):
        """
            Description:
            16-bit version of the GetImages function.

            Synopsis:
            (ret, arr, validfirst, validlast) = GetImages16(first, last, size)

            Inputs:
            first - index of first image in buffer to retrieve.
            last - index of last image in buffer to retrieve.
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Images have been copied into array.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_GENERAL_ERRORS - The series is out of range.
                DRV_P3INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.
            validfirst - index of the first valid image.
            validlast - index of the last valid image.

            C++ Equiv:
            unsigned int GetImages16(long first, long last, WORD * arr, long size, long * validfirst, long * validlast);

            See Also:
            GetImages GetNumberNewImages

        """
        cfirst = c_int(first)
        clast = c_int(last)
        carr = (c_short * size)()
        csize = c_int(size)
        cvalidfirst = c_int()
        cvalidlast = c_int()
        ret = self.dll.GetImages16(
            cfirst, clast, carr, csize, byref(cvalidfirst), byref(cvalidlast))
        return (ret, np.ctypeslib.as_array(carr), cvalidfirst.value, cvalidlast.value)

    def GetImagesPerDMA(self):
        """
            Description:
            This function will return the maximum number of images that can be transferred during a single DMA transaction.

            Synopsis:
            (ret, images) = GetImagesPerDMA()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of images per DMA returned.
            images - The maximum number of images that can be transferred during a single DMA transaction

            C++ Equiv:
            unsigned int GetImagesPerDMA(unsigned long * images);

        """
        cimages = c_ulong()
        ret = self.dll.GetImagesPerDMA(byref(cimages))
        return (ret, cimages.value)

    def GetIODirection(self, index):
        """
            Description:
            Available in some systems are a number of IOs that can be configured to be inputs or outputs. This function gets the current state of a particular IO.

            Synopsis:
            (ret, iDirection) = GetIODirection(index)

            Inputs:
            index - IO index. Valid values: 0 to GetNumberIO() - 1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IO direction returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Invalid parameter.
                DRV_NOT_AVAILABLE - Feature not available.
            iDirection - current direction for this index.:
                0 - 0 Output
                1 - 1 Input

            C++ Equiv:
            unsigned int GetIODirection(int index, int * iDirection);

            See Also:
            GetNumberIO GetIOLevel SetIODirection SetIOLevel

        """
        cindex = c_int(index)
        ciDirection = c_int()
        ret = self.dll.GetIODirection(cindex, byref(ciDirection))
        return (ret, ciDirection.value)

    def GetIOLevel(self, index):
        """
            Description:
            Available in some systems are a number of IOs that can be configured to be inputs or outputs. This function gets the current state of a particular IO.

            Synopsis:
            (ret, iLevel) = GetIOLevel(index)

            Inputs:
            index - IO index:
                0 - toGetNumberIO() - 1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IO level returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Invalid parameter.
                DRV_NOT_AVAILABLE - Feature not available.
            iLevel - current level for this index.:
                0 - 0 Low
                1 - 1 High

            C++ Equiv:
            unsigned int GetIOLevel(int index, int * iLevel);

            See Also:
            GetNumberIO GetIODirection SetIODirection SetIOLevel

        """
        cindex = c_int(index)
        ciLevel = c_int()
        ret = self.dll.GetIOLevel(cindex, byref(ciLevel))
        return (ret, ciLevel.value)

    def GetIRIGData(self, index):
        """
            Description:
            This function retrieves the IRIG data for the requested frame. The buffer will be populated with 128bits of information.
            · The least significant 100 bits contains the IRIG time frame as received from the external IRIG device.
                This will be the most recent full time frame available when the start exposure event occurred.
            · The next 8 bits indicate the bit position in the current IRIG time frame when the start exposure event occurred.
            · The final 20 bits indicates how many 10nS clock periods have occurred since the start of the current bit period and the start exposure event.

            Synopsis:
            (ret, irigData) = GetIRIGData(index)

            Inputs:
            index - frame for which IRIG data is required

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IRIG data successfully retrieved
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_SUPPORTED - Feature not supported on this camera
                DRV_P1INVALID - Buffer is invalid
                DRV_MSTIMINGS_ERROR - Requested frame isn’t valid
            irigData - buffer that will be populated with the IRIG data, must be at least 16 bytes

            C++ Equiv:
            unsigned int GetIRIGData(unsigned char * irigData, unsigned int index);

            See Also:
            GetCapabilities

            Note: This function retrieves the IRIG data for the requested frame. The buffer will be populated with 128bits of information.
            · The least significant 100 bits contains the IRIG time frame as received from the external IRIG device.
            This will be the most recent full time frame available when the start exposure event occurred.
            · The next 8 bits indicate the bit position in the current IRIG time frame when the start exposure event occurred.
            · The final 20 bits indicates how many 10nS clock periods have occurred since the start of the current bit period and the start exposure event.

        """
        cirigData = c_ubyte()
        cindex = c_uint(index)
        ret = self.dll.GetIRIGData(byref(cirigData), cindex)
        return (ret, cirigData.value)

    def GetIRQ(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, IRQ) = GetIRQ()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            IRQ -

            C++ Equiv:
            unsigned int GetIRQ(int * IRQ);

        """
        cIRQ = c_int()
        ret = self.dll.GetIRQ(byref(cIRQ))
        return (ret, cIRQ.value)

    def GetKeepCleanTime(self):
        """
            Description:
            This function will return the time to perform a keep clean cycle.
            This function should be used after all the acquisitions settings have been set,
            e.g. SetExposureTimeSetExposureTime, SetKineticCycleTimeSetKineticCycleTime and SetReadModeSetReadMode etc.
            The value returned is the actual times used in subsequent acquisitions.

            Synopsis:
            (ret, KeepCleanTime) = GetKeepCleanTime()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timing information returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_CODES - Error communicating with camera.
            KeepCleanTime - valid readout time in seconds

            C++ Equiv:
            unsigned int GetKeepCleanTime(float * KeepCleanTime);

            See Also:
            GetAcquisitionTimings GetReadOutTime

            Note: Available on iDus, iXon, Luca & Newton.

        """
        cKeepCleanTime = c_float()
        ret = self.dll.GetKeepCleanTime(byref(cKeepCleanTime))
        return (ret, cKeepCleanTime.value)

    def GetMaximumBinning(self, ReadMode, HorzVert):
        """
            Description:
            This function will return the maximum binning allowable in either the vertical or horizontal dimension for a particular readout mode.

            Synopsis:
            (ret, MaxBinning) = GetMaximumBinning(ReadMode, HorzVert)

            Inputs:
            ReadMode - The readout mode for which to retrieve the maximum binning (see SetReadMode for possible values).
            HorzVert - 0 to retrieve horizontal binning limit, 1 to retreive limit in the vertical.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Maximum Binning returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - Invalid Readmode
                DRV_P2INVALID - HorzVert not equal to 0 or 1
                DRV_P3INVALID - Invalid MaxBinning address (i.e. NULL)
            MaxBinning - Will contain the Maximum binning value on return.

            C++ Equiv:
            unsigned int GetMaximumBinning(int ReadMode, int HorzVert, int * MaxBinning);

            See Also:
            GetMinimumImageLength SetReadMode

        """
        cReadMode = c_int(ReadMode)
        cHorzVert = c_int(HorzVert)
        cMaxBinning = c_int()
        ret = self.dll.GetMaximumBinning(
            cReadMode, cHorzVert, byref(cMaxBinning))
        return (ret, cMaxBinning.value)

    def GetMaximumExposure(self):
        """
            Description:
            This function will return the maximum Exposure Time in seconds that is settable by the SetExposureTime function.

            Synopsis:
            (ret, MaxExp) = GetMaximumExposure()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Maximum Exposure returned.
                DRV_P1INVALID - Invalid MaxExp value (i.e. NULL)
            MaxExp - Will contain the Maximum exposure value on return.

            C++ Equiv:
            unsigned int GetMaximumExposure(float * MaxExp);

            See Also:
            SetExposureTime

        """
        cMaxExp = c_float()
        ret = self.dll.GetMaximumExposure(byref(cMaxExp))
        return (ret, cMaxExp.value)

    def GetMaximumNumberRingExposureTimes(self):
        """
            Description:
            This function will return the maximum number of exposures that can be configured in the SetRingExposureTimes SDK function.

            Synopsis:
            (ret, number) = GetMaximumNumberRingExposureTimes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Success
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - Invalid number value (ie NULL)
                DRV_NOTAVAILABLE - System does not support this option
            number - Will contain the maximum number of exposures on return.

            C++ Equiv:
            unsigned int GetMaximumNumberRingExposureTimes(int * number);

            See Also:
            GetCapabilities GetNumberRingExposureTimes GetAdjustedRingExposureTimes GetRingExposureRange IsTriggerModeAvailable SetRingExposureTimes

        """
        cnumber = c_int()
        ret = self.dll.GetMaximumNumberRingExposureTimes(byref(cnumber))
        return (ret, cnumber.value)

    def GetMCPGain(self):
        """
            Description:
            This function will retrieve the set value for the MCP Gain.

            Synopsis:
            (ret, gain) = GetMCPGain()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Table returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - Invalid pointer (i.e. NULL)
                DRV_NOT_AVAILABLE - Not a USB iStar
            gain - Returned gain value.

            C++ Equiv:
            unsigned int GetMCPGain(int * gain);

            See Also:
            SetMCPGain

            Note: Available only on USB iStar.

        """
        cgain = c_int()
        ret = self.dll.GetMCPGain(byref(cgain))
        return (ret, cgain.value)

    def GetMCPGainRange(self):
        """
            Description:
            Returns the minimum and maximum values of the SetMCPGain function.

            Synopsis:
            (ret, iLow, iHigh) = GetMCPGainRange()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gain range returned.
                DRV_NOT_INITIALIZED - System not initialized.
            iLow - lowest gain setting
            iHigh - highest gain setting

            C++ Equiv:
            unsigned int GetMCPGainRange(int * iLow, int * iHigh);

            See Also:
            SetMCPGain

            Note: Available only iStar.

        """
        ciLow = c_int()
        ciHigh = c_int()
        ret = self.dll.GetMCPGainRange(byref(ciLow), byref(ciHigh))
        return (ret, ciLow.value, ciHigh.value)

    def GetMCPGainTable(self, iNum):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, piGain, pfPhotoepc) = GetMCPGainTable(iNum)

            Inputs:
            iNum -

            Outputs:
            ret - Function Return Code
            piGain -
            pfPhotoepc -

            C++ Equiv:
            unsigned int GetMCPGainTable(int iNum, int * piGain, float * pfPhotoepc);

        """
        ciNum = c_int(iNum)
        cpiGain = c_int()
        cpfPhotoepc = c_float()
        ret = self.dll.GetMCPGainTable(
            ciNum, byref(cpiGain), byref(cpfPhotoepc))
        return (ret, cpiGain.value, cpfPhotoepc.value)

    def GetMCPVoltage(self):
        """
            Description:
            This function will retrieve the current Micro Channel Plate voltage.

            Synopsis:
            (ret, iVoltage) = GetMCPVoltage()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Voltage returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_AVAILABLE - Not a USB iStar.
                DRV_GENERAL_ERRORS - EEPROM not valid
            iVoltage - Will contain voltage on return. The unit is in Volts and should be between the range 600 - 1100 Volts.

            C++ Equiv:
            unsigned int GetMCPVoltage(int * iVoltage);

            See Also:
            GetMCPGain

            Note: Available only on USB iStar.

        """
        ciVoltage = c_int()
        ret = self.dll.GetMCPVoltage(byref(ciVoltage))
        return (ret, ciVoltage.value)

    def GetMetaDataInfo(self, index):
        """
            Description:
            This function will return the time of the initial frame and the time in milliseconds of further frames from this point.

            Synopsis:
            (ret, TimeOfStart, pfTimeFromStart) = GetMetaDataInfo(index)

            Inputs:
            index - frame for which time is required.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_MSTIMINGS_ERROR - Invalid timing request
            TimeOfStart - Structure with start time details.
            pfTimeFromStart - time in milliseconds for a particular frame from time of start.

            C++ Equiv:
            unsigned int GetMetaDataInfo(SYSTEMTIME * TimeOfStart, float * pfTimeFromStart, int index);

            See Also:
            SetMetaData

        """
        cTimeOfStart = SYSTEMTIME()
        cpfTimeFromStart = c_float()
        cindex = c_int(index)
        ret = self.dll.GetMetaDataInfo(
            byref(cTimeOfStart), byref(cpfTimeFromStart), cindex)
        return (ret, cTimeOfStart, cpfTimeFromStart.value)

    def GetMinimumImageLength(self):
        """
            Description:
            This function will return the minimum number of pixels that can be read out from the chip at each exposure. This minimum value arises due the way in which the chip is read out and will limit the possible sub image dimensions and binning sizes that can be applied.

            Synopsis:
            (ret, MinImageLength) = GetMinimumImageLength()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Minimum Number of Pixels returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - Invalid MinImageLength value (i.e. NULL)
            MinImageLength - Will contain the minimum number of super pixels on return.

            C++ Equiv:
            unsigned int GetMinimumImageLength(int * MinImageLength);

            See Also:
            SetImage

        """
        cMinImageLength = c_int()
        ret = self.dll.GetMinimumImageLength(byref(cMinImageLength))
        return (ret, cMinImageLength.value)

    def GetMinimumNumberInSeries(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, number) = GetMinimumNumberInSeries()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            number -

            C++ Equiv:
            unsigned int GetMinimumNumberInSeries(int * number);

        """
        cnumber = c_int()
        ret = self.dll.GetMinimumNumberInSeries(byref(cnumber))
        return (ret, cnumber.value)

    def GetMostRecentColorImage16(self, size, algorithm):
        """
            Description:
            For colour sensors only.
            Color version of the GetMostRecentImage16 function. The CCD is sensitive to Cyan, Yellow, Magenta and Green (CYMG). The Red, Green and Blue (RGB) are calculated and Data is stored in 3 planes/images, one for each basic color.

            Synopsis:
            (ret, red, green, blue) = GetMostRecentColorImage16(size, algorithm)

            Inputs:
            size - total number of pixels.
            algorithm - algorithm used to extract the RGB from the original CYMG CCD.:
                0 - 0 basic algorithm combining Cyan, Yellow and Magenta.
                1 - 1 algorithm combining Cyan, Yellow, Magenta and Green.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Image RGB has been copied into arrays.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Arrays size is incorrect.
                DRV_P2INVALID - Invalid algorithm.
                DRV_P3INVALID - Invalid red pointer (i.e. NULL)..
                DRV_P4INVALID - Invalid green pointer (i.e. NULL)..
                DRV_P5INVALID - Invalid bluepointer (i.e. NULL)..
                DRV_NO_NEW_DATA - There is no new data yet.
            red - pointer to red data storage allocated by the user.
            green - pointer to red data storage allocated by the user.
            blue - pointer to red data storage allocated by the user.

            C++ Equiv:
            unsigned int GetMostRecentColorImage16(unsigned long size, int algorithm, WORD * red, WORD * green, WORD * blue);

            See Also:
            GetMostRecentImage16 DemosaicImage WhiteBalance

        """
        csize = c_ulong(size)
        calgorithm = c_int(algorithm)
        cred = (c_short * size)()
        cgreen = (c_short * size)()
        cblue = (c_short * size)()
        ret = self.dll.GetMostRecentColorImage16(
            csize, calgorithm, cred, cgreen, cblue)
        return (ret, cred, cgreen, cblue)

    def GetMostRecentImage(self, size):
        """
            Description:
            This function will update the data array with the most recently acquired image in any acquisition mode.
            The data are returned as long integers (32-bit signed integers).
            The "array" must be exactly the same size as the complete image.

            Synopsis:
            (ret, arr) = GetMostRecentImage(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Image has been copied into array.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetMostRecentImage(at_32 * arr, unsigned long size);

            See Also:
            GetMostRecentImage16 GetOldestImage GetOldestImage16 GetImages

        """
        carr = (c_int * size)()
        csize = c_ulong(size)
        ret = self.dll.GetMostRecentImage(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetMostRecentImage16(self, size):
        """
            Description:
            16-bit version of the GetMostRecentImageGetMostRecentImage function.

            Synopsis:
            (ret, arr) = GetMostRecentImage16(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Image has been copied into array.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetMostRecentImage16(WORD * arr, long size);

            See Also:
            GetMostRecentImage GetOldestImage16 GetOldestImage GetImages

        """
        carr = (c_short * size)()
        csize = c_int(size)
        ret = self.dll.GetMostRecentImage16(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetMSTimingsData(self, inoOfImages):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, TimeOfStart, pfDifferences) = GetMSTimingsData(inoOfImages)

            Inputs:
            inoOfImages -

            Outputs:
            ret - Function Return Code
            TimeOfStart -
            pfDifferences -

            C++ Equiv:
            unsigned int GetMSTimingsData(SYSTEMTIME * TimeOfStart, float * pfDifferences, int inoOfImages);

        """
        cTimeOfStart = SYSTEMTIME()
        cpfDifferences = c_float()
        cinoOfImages = c_int(inoOfImages)
        ret = self.dll.GetMSTimingsData(
            byref(cTimeOfStart), byref(cpfDifferences), cinoOfImages)
        return (ret, cTimeOfStart, cpfDifferences.value)

    def GetMSTimingsEnabled(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = GetMSTimingsEnabled()

            Inputs:
            None

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int GetMSTimingsEnabled(void);

        """
        ret = self.dll.GetMSTimingsEnabled()
        return (ret)

    def GetNewData(self, size):
        """
            Description:
            Deprecated see Note:
            This function will update the data array to hold data acquired so far. The data are returned as long integers (32-bit signed integers). The array must be large enough to hold the complete data set. When used in conjunction with the SetDriverEventSetDriverEvent and GetAcquisitonProgressGetAcquisitionProgress functions, the data from each scan in a kinetic series can be processed while the acquisition is taking place.

            Synopsis:
            (ret, arr) = GetNewData(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data copied.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetNewData(at_32 * arr, long size); // deprecated

            See Also:
            SetDriverEvent GetAcquisitionProgress SetAcquisitionMode SetAcGetNewData8 GetNewData16

            Note: Deprecated by the following functions:
                * GetImages
                * GetMostRecentImage
                * GetOldestImage

        """
        carr = c_int()
        csize = c_int(size)
        ret = self.dll.GetNewData(byref(carr), csize)
        return (ret, carr.value)

    def GetNewData16(self, size):
        """
            Description:
            Deprecated see Note:
            16-bit version of the GetNewDataGetNewData function.

            Synopsis:
            (ret, arr) = GetNewData16(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data copied.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetNewData16(WORD * arr, long size); // deprecated

            Note: Deprecated by the following functions:
                * GetImages
                * GetMostRecentImage
                * GetOldestImage

        """
        carr = c_short()
        csize = c_int(size)
        ret = self.dll.GetNewData16(byref(carr), csize)
        return (ret, carr.value)

    def GetNewData8(self, size):
        """
            Description:
            Deprecated see Note:
            8-bit version of the GetNewDataGetNewData function. This function will return the data in the lower 8 bits of the acquired data.

            Synopsis:
            (ret, arr) = GetNewData8(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data copied.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetNewData8(unsigned char * arr, long size);

            Note: Deprecated by the following functions:
                * GetImages
                * GetMostRecentImage
                * GetOldestImage

        """
        carr = (c_ubyte * size)()
        csize = c_int(size)
        ret = self.dll.GetNewData8(carr, csize)
        return (ret, carr)

    def GetNewFloatData(self, size):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, arr) = GetNewFloatData(size)

            Inputs:
            size -

            Outputs:
            ret - Function Return Code
            arr -

            C++ Equiv:
            unsigned int GetNewFloatData(float * arr, long size);

        """
        carr = c_float()
        csize = c_int(size)
        ret = self.dll.GetNewFloatData(byref(carr), csize)
        return (ret, carr.value)

    def GetNumberADChannels(self):
        """
            Description:
            As your Andor SDK system may be capable of operating with more than one A-D converter, this function will tell you the number available.

            Synopsis:
            (ret, channels) = GetNumberADChannels()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of channels returned.
            channels - number of allowed channels

            C++ Equiv:
            unsigned int GetNumberADChannels(int * channels);

            See Also:
            SetADChannel

        """
        cchannels = c_int()
        ret = self.dll.GetNumberADChannels(byref(cchannels))
        return (ret, cchannels.value)

    def GetNumberAmp(self):
        """
            Description:
            As your Andor SDK system may be capable of operating with more than one output amplifier, this function will tell you the number available.

            Synopsis:
            (ret, amp) = GetNumberAmp()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of output amplifiers returned.
            amp - number of allowed channels

            C++ Equiv:
            unsigned int GetNumberAmp(int * amp);

            See Also:
            SetOutputAmplifier

        """
        camp = c_int()
        ret = self.dll.GetNumberAmp(byref(camp))
        return (ret, camp.value)

    def GetNumberAvailableImages(self):
        """
            Description:
            This function will return information on the number of available images in the circular buffer. This information can be used with GetImages to retrieve a series of images. If any images are overwritten in the circular buffer they no longer can be retrieved and the information returned will treat overwritten images as not available.

            Synopsis:
            (ret, first, last) = GetNumberAvailableImages()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of acquired images returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ERROR_ACK - Unable to communicate with card
                DRV_NO_NEW_DATA - There is no new data yet
            first - returns the index of the first available image in the circular buffer.
            last - returns the index of the last available image in the circular buffer.

            C++ Equiv:
            unsigned int GetNumberAvailableImages(at_32 * first, at_32 * last);

            See Also:
            GetImages GetImages16 GetNumberNewImages

        """
        cfirst = c_int()
        clast = c_int()
        ret = self.dll.GetNumberAvailableImages(byref(cfirst), byref(clast))
        return (ret, cfirst.value, clast.value)

    def GetNumberDDGExternalOutputs(self):
        """
            Description:
            This function gets the number of available external outputs.

            Synopsis:
            (ret, puiCount) = GetNumberDDGExternalOutputs()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
            puiCount - number of available external outputs.

            C++ Equiv:
            unsigned int GetNumberDDGExternalOutputs(at_u32 * puiCount);

            See Also:
            GetCapabilities SetDDGExternalOutputEnabled SetDDGGateStep

            Note: Available on USB iStar.

        """
        cpuiCount = c_uint()
        ret = self.dll.GetNumberDDGExternalOutputs(byref(cpuiCount))
        return (ret, cpuiCount.value)

    def GetNumberDevices(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, numDevs) = GetNumberDevices()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            numDevs -

            C++ Equiv:
            unsigned int GetNumberDevices(int * numDevs);

        """
        cnumDevs = c_int()
        ret = self.dll.GetNumberDevices(byref(cnumDevs))
        return (ret, cnumDevs.value)

    def GetNumberFKVShiftSpeeds(self):
        """
            Description:
            As your Andor SDK system is capable of operating at more than one fast kinetics vertical shift speed this function will return the actual number of speeds available.

            Synopsis:
            (ret, number) = GetNumberFKVShiftSpeeds()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of speeds returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            number - number of allowed speeds

            C++ Equiv:
            unsigned int GetNumberFKVShiftSpeeds(int * number);

            See Also:
            GetFKVShiftSpeedF SetFKVShiftSpeed

            Note: Only available if camera is Classic or iStar.

        """
        cnumber = c_int()
        ret = self.dll.GetNumberFKVShiftSpeeds(byref(cnumber))
        return (ret, cnumber.value)

    def GetNumberHorizontalSpeeds(self):
        """
            Description:
            Deprecated see Note:
            As your Andor SDK system is capable of operating at more than one horizontal shift speed this function will return the actual number of speeds available.

            Synopsis:
            (ret, number) = GetNumberHorizontalSpeeds()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of speeds returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            number - number of allowed horizontal speeds

            C++ Equiv:
            unsigned int GetNumberHorizontalSpeeds(int * number); // deprecated

            See Also:
            GetHorizontalSpeed SetHorizontalSpeed

            Note: Deprecated by GetNumberHSSpeedsGetNumberHSSpeeds

        """
        cnumber = c_int()
        ret = self.dll.GetNumberHorizontalSpeeds(byref(cnumber))
        return (ret, cnumber.value)

    def GetNumberHSSpeeds(self, channel, typ):
        """
            Description:
            As your Andor SDK system is capable of operating at more than one horizontal shift speed this function will return the actual number of speeds available.

            Synopsis:
            (ret, speeds) = GetNumberHSSpeeds(channel, typ)

            Inputs:
            channel - the AD channel.
            typ - output amplification.:
                0 - electron multiplication.
                1 - conventional.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of speeds returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid channel.
                DRV_P2INVALID - Invalid horizontal read mode
            speeds - number of allowed horizontal speeds

            C++ Equiv:
            unsigned int GetNumberHSSpeeds(int channel, int typ, int * speeds);

            See Also:
            GetHSSpeed SetHSSpeed GetNumberADChannel

        """
        cchannel = c_int(channel)
        ctyp = c_int(typ)
        cspeeds = c_int()
        ret = self.dll.GetNumberHSSpeeds(cchannel, ctyp, byref(cspeeds))
        return (ret, cspeeds.value)

    def GetNumberIO(self):
        """
            Description:
            Available in some systems are a number of IOs that can be configured to be inputs or outputs. This function gets the number of these IOs available. The functions GetIODirection, GetIOLevel, SetIODirection and SetIOLevel can be used to specify the configuration.

            Synopsis:
            (ret, iNumber) = GetNumberIO()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of  IOs returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid parameter.
                DRV_NOT_AVAILABLE - Feature not available.
            iNumber - number of allowed IOs

            C++ Equiv:
            unsigned int GetNumberIO(int * iNumber);

            See Also:
            GetIOLevel GetIODirection SetIODirection SetIOLevel

        """
        ciNumber = c_int()
        ret = self.dll.GetNumberIO(byref(ciNumber))
        return (ret, ciNumber.value)

    def GetNumberNewImages(self):
        """
            Description:
            This function will return information on the number of new images (i.e. images which have not yet been retrieved) in the circular buffer. This information can be used with GetImages to retrieve a series of the latest images. If any images are overwritten in the circular buffer they can no longer be retrieved and the information returned will treat overwritten images as having been retrieved.

            Synopsis:
            (ret, first, last) = GetNumberNewImages()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of acquired images returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_NO_NEW_DATA - There is no new data yet.
            first - returns the index of the first available image in the circular buffer.
            last - returns the index of the last available image in the circular buffer.

            C++ Equiv:
            unsigned int GetNumberNewImages(long * first, long * last);

            See Also:
            GetImages GetImages16 GetNumberAvailableImages

            Note: This index will increment as soon as a single accumulation has been completed within the current acquisition.


        """
        cfirst = c_int()
        clast = c_int()
        ret = self.dll.GetNumberNewImages(byref(cfirst), byref(clast))
        return (ret, cfirst.value, clast.value)

    def GetNumberPhotonCountingDivisions(self):
        """
            Description:
            Available in some systems is photon counting mode. This function gets the number of photon counting divisions available. The functions SetPhotonCounting and SetPhotonCountingThreshold can be used to specify which of these divisions is to be used.

            Synopsis:
            (ret, noOfDivisions) = GetNumberPhotonCountingDivisions()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of photon counting divisions returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid parameter.
                DRV_NOT_AVAILABLE - Photon Counting not available
            noOfDivisions - number of allowed photon counting divisions

            C++ Equiv:
            unsigned int GetNumberPhotonCountingDivisions(at_u32 * noOfDivisions);

            See Also:
            SetPhotonCounting IsPreAmpGainAvailable SetPhotonCountingThresholdGetPreAmpGain GetCapabilities

        """
        cnoOfDivisions = c_uint()
        ret = self.dll.GetNumberPhotonCountingDivisions(byref(cnoOfDivisions))
        return (ret, cnoOfDivisions.value)

    def GetNumberPreAmpGains(self):
        """
            Description:
            Available in some systems are a number of pre amp gains that can be applied to the data as it is read out. This function gets the number of these pre amp gains available. The functions GetPreAmpGain and SetPreAmpGain can be used to specify which of these gains is to be used.

            Synopsis:
            (ret, noGains) = GetNumberPreAmpGains()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of pre amp gains returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            noGains - number of allowed pre amp gains

            C++ Equiv:
            unsigned int GetNumberPreAmpGains(int * noGains);

            See Also:
            IsPreAmpGainAvailable GetPreAmpGain SetPreAmpGain GetCapabilities

        """
        cnoGains = c_int()
        ret = self.dll.GetNumberPreAmpGains(byref(cnoGains))
        return (ret, cnoGains.value)

    def GetNumberRingExposureTimes(self):
        """
            Description:
            Gets the number of exposures in the ring at this moment.

            Synopsis:
            (ret, ipnumTimes) = GetNumberRingExposureTimes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Success
                DRV_NOT_INITIALIZED - System not initialized
            ipnumTimes - Numberof exposure times.

            C++ Equiv:
            unsigned int GetNumberRingExposureTimes(int * ipnumTimes);

            See Also:
            SetRingExposureTimes

        """
        cipnumTimes = c_int()
        ret = self.dll.GetNumberRingExposureTimes(byref(cipnumTimes))
        return (ret, cipnumTimes.value)

    def GetNumberVerticalSpeeds(self):
        """
            Description:
            Deprecated see Note:
            As your Andor system may be capable of operating at more than one vertical shift speed this function will return the actual number of speeds available.

            Synopsis:
            (ret, number) = GetNumberVerticalSpeeds()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of speeds returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            number - number of allowed vertical speeds

            C++ Equiv:
            unsigned int GetNumberVerticalSpeeds(int * number); // deprecated

            See Also:
            GetVerticalSpeed SetVerticalSpeed

            Note: Deprecated by GetNumberVSSpeedsGetNumberVSSpeeds

        """
        cnumber = c_int()
        ret = self.dll.GetNumberVerticalSpeeds(byref(cnumber))
        return (ret, cnumber.value)

    def GetNumberVSAmplitudes(self):
        """
            Description:
            This function will normally return the number of vertical clock voltage amplitues that the camera has.

            Synopsis:
            (ret, number) = GetNumberVSAmplitudes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters
                DRV_NOT_INITIALIZED - int* speeds: number of allowed vertical speeds
                DRV_NOT_AVAILABLE - Return
                Number returned - Return int
                System not initialized - DRV_SUCCESS
                Your system does not support this feature - DRV_NOT_INITIALIZED
                GetNumberVSSpeeds - DRV_ACQUIRING
                GetNumberVSSpeeds int WINAPI GetNumberVSSpeeds(int* speeds) - Number of speeds returned.
                Description - System not initialized.
                As your Andor system may be capable of operating at more than one vertical shift speed this function will return the actual number of speeds available. - Acquisition in progress.
            number - Number of vertical clock voltages.

            C++ Equiv:
            unsigned int GetNumberVSAmplitudes(int * number);

            See Also:
            GetVSSpeed SetVSSpeed GetFastestRecommendedVSSpeed

        """
        cnumber = c_int()
        ret = self.dll.GetNumberVSAmplitudes(byref(cnumber))
        return (ret, cnumber.value)

    def GetNumberVSSpeeds(self):
        """
            Description:


            Synopsis:
            (ret, speeds) = GetNumberVSSpeeds()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            speeds -

            C++ Equiv:
            unsigned int GetNumberVSSpeeds(int * speeds);

        """
        cspeeds = c_int()
        ret = self.dll.GetNumberVSSpeeds(byref(cspeeds))
        return (ret, cspeeds.value)

    def GetOldestImage(self, size):
        """
            Description:
            This function will update the data array with the oldest image in the circular buffer. Once the oldest image has been retrieved it no longer is available. The data are returned as long integers (32-bit signed integers). The "array" must be exactly the same size as the full image.

            Synopsis:
            (ret, arr) = GetOldestImage(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Image has been copied into array.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetOldestImage(at_32 * arr, unsigned long size);

            See Also:
            GetOldestImage16 GetMostRecentImage GetMostRecentImage16

        """
        carr = (c_int * size)()
        csize = c_ulong(size)
        ret = self.dll.GetOldestImage(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetOldestImage16(self, size):
        """
            Description:
            16-bit version of the GetOldestImageGetOldestImage function.

            Synopsis:
            (ret, arr) = GetOldestImage16(size)

            Inputs:
            size - total number of pixels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Image has been copied into array.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Array size is incorrect.
                DRV_NO_NEW_DATA - There is no new data yet.
            arr - pointer to data storage allocated by the user.

            C++ Equiv:
            unsigned int GetOldestImage16(WORD * arr, unsigned long size);

            See Also:
            GetOldestImage GetMostRecentImage16 GetMostRecentImage

        """
        carr = (c_short * size)()
        csize = c_ulong(size)
        ret = self.dll.GetOldestImage16(carr, csize)
        return (ret, np.ctypeslib.as_array(carr))

    def GetPhosphorStatus(self):
        """
            Description:
            This function will return if the phosphor has saturated.

            Synopsis:
            (ret, flag) = GetPhosphorStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Phosphor status not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
            flag - The status of the phosphor:
                0 - Saturated
                1 - Normal

            C++ Equiv:
            unsigned int GetPhosphorStatus(int * flag);

        """
        cflag = c_int()
        ret = self.dll.GetPhosphorStatus(byref(cflag))
        return (ret, cflag.value)

    def GetPhysicalDMAAddress(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, Address1, Address2) = GetPhysicalDMAAddress()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            Address1 -
            Address2 -

            C++ Equiv:
            unsigned int GetPhysicalDMAAddress(unsigned long * Address1, unsigned long * Address2);

        """
        cAddress1 = c_ulong()
        cAddress2 = c_ulong()
        ret = self.dll.GetPhysicalDMAAddress(
            byref(cAddress1), byref(cAddress2))
        return (ret, cAddress1.value, cAddress2.value)

    def GetPixelSize(self):
        """
            Description:
            This function returns the dimension of the pixels in the detector in microns.

            Synopsis:
            (ret, xSize, ySize) = GetPixelSize()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Pixel size returned.
            xSize - width of pixel.
            ySize - height of pixel.

            C++ Equiv:
            unsigned int GetPixelSize(float * xSize, float * ySize);

        """
        cxSize = c_float()
        cySize = c_float()
        ret = self.dll.GetPixelSize(byref(cxSize), byref(cySize))
        return (ret, cxSize.value, cySize.value)

    def GetPreAmpGain(self, index):
        """
            Description:
            For those systems that provide a number of pre amp gains to apply to the data as it is read out; this function retrieves the amount of gain that is stored for a particular index. The number of gains available can be obtained by calling the GetNumberPreAmpGainsGetNumberPreAmpGains function and a specific Gain can be selected using the function SetPreAmpGainSetPreAmpGain.

            Synopsis:
            (ret, gain) = GetPreAmpGain(index)

            Inputs:
            index - gain index:
                0 - to GetNumberPreAmpGainsGetNumberPreAmpGains()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gain returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
            gain - gain factor for this index.

            C++ Equiv:
            unsigned int GetPreAmpGain(int index, float * gain);

            See Also:
            IsPreAmpGainAvailable GetNumberPreAmpGains SetPreAmpGain GetCapabilities

        """
        cindex = c_int(index)
        cgain = c_float()
        ret = self.dll.GetPreAmpGain(cindex, byref(cgain))
        return (ret, cgain.value)

    def GetPreAmpGainText(self, index, length):
        """
            Description:
            This function will return a string with a pre amp gain description. The pre amp gain is selected using the index. The SDK has a string associated with each of its pre amp gains. The maximum number of characters needed to store the pre amp gain descriptions is 30. The user has to specify the number of characters they wish to have returned to them from this function.

            Synopsis:
            (ret, name) = GetPreAmpGainText(index, length)

            Inputs:
            index - gain index 0 to GetNumberPreAmpGainsGetNumberPreAmpGains()-1
            length - The length of the user allocated character array.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Description returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Array size is incorrect
                DRV_NOT_SUPPORTED - Function not supported with this camera
            name - nameA user allocated array of characters for storage of the description.

            C++ Equiv:
            unsigned int GetPreAmpGainText(int index, char * name, int length);

            See Also:
            IsPreAmpGainAvailable GetNumberPreAmpGains SetPreAmpGain GetCapabilities

        """
        cindex = c_int(index)
        cname = create_string_buffer(length)
        clength = c_int(length)
        ret = self.dll.GetPreAmpGainText(cindex, cname, clength)
        return (ret, cname)

    def GetQE(self, sensor, wavelength, mode):
        """
            Description:
            Returns the percentage QE for a particular head model at a user specified wavelengthSetPreAmpGain.

            Synopsis:
            (ret, QE) = GetQE(sensor, wavelength, mode)

            Inputs:
            sensor - head model
            wavelength - wavelength at which QE is required
            mode - Clara mode (Normal (0) or Extended NIR (1)).  0 for all other systems

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - QE returned.
                DRV_NOT_INITIALIZED - System not initialized.
            QE - requested QE

            C++ Equiv:
            unsigned int GetQE(char * sensor, float wavelength, int mode, float * QE);

            See Also:
            GetHeadModel IsPreAmpGainAvailable SetPreAmpGain GetCapabilities

        """
        csensor = sensor
        cwavelength = c_float(wavelength)
        cmode = c_int(mode)
        cQE = c_float()
        ret = self.dll.GetQE(csensor, cwavelength, cmode, byref(cQE))
        return (ret, cQE.value)

    def GetReadOutTime(self):
        """
            Description:
            This function will return the time to readout data from a sensor. This function should be used after all the acquisitions settings have been set, e.g. SetExposureTimeSetExposureTime, SetKineticCycleTimeSetKineticCycleTime and SetReadModeSetReadMode etc. The value returned is the actual times used in subsequent acquisitions.

            Synopsis:
            (ret, ReadOutTime) = GetReadOutTime()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timing information returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_CODES - Error communicating with camera.
            ReadOutTime - valid readout time in seconds

            C++ Equiv:
            unsigned int GetReadOutTime(float * ReadOutTime);

            See Also:
            GetAcquisitionTimings GetKeepCleanTime

            Note: Available on iDus, iXon, Luca & Newton.

        """
        cReadOutTime = c_float()
        ret = self.dll.GetReadOutTime(byref(cReadOutTime))
        return (ret, cReadOutTime.value)

    def GetRegisterDump(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, mode) = GetRegisterDump()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            mode -

            C++ Equiv:
            unsigned int GetRegisterDump(int * mode);

        """
        cmode = c_int()
        ret = self.dll.GetRegisterDump(byref(cmode))
        return (ret, cmode.value)

    def GetRelativeImageTimes(self, first, last, size):
        """
            Description:
            This function will return an array of the start times in nanoseconds of a user defined number of frames relative to the initial frame.

            Synopsis:
            (ret, arr) = GetRelativeImageTimes(first, last, size)

            Inputs:
            first - Index of first frame in array.
            last - Index of last frame in array.
            size - number of frames for which start time is required.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_MSTIMINGS_ERROR - Invalid timing request
            arr - array of times in nanoseconds for each frame from time of start.

            C++ Equiv:
            unsigned int GetRelativeImageTimes(int first, int last, at_u64 * arr, int size);

            See Also:
            GetCapabilities SetMetaData

        """
        cfirst = c_int(first)
        clast = c_int(last)
        carr = (c_ulonglong * size)()
        csize = c_int(size)
        ret = self.dll.GetRelativeImageTimes(cfirst, clast, carr, csize)
        
        arr = np.ndarray((size, ), 'f', carr, order='C')
        return (ret, arr)

    def GetRingExposureRange(self):
        """
            Description:
            With the Ring Of Exposure feature there may be a case when not all exposures can be met. The ring of exposure feature will guarantee that the highest exposure will be met but this may mean that the lower exposures may not be. If the lower exposures are too low they will be increased to the lowest value possible. This function will return these upper and lower values.

            Synopsis:
            (ret, fpMin, fpMax) = GetRingExposureRange()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Min and max returned
                DRV_NOT_INITIALIZED - System not initialize
                DRV_INVALID_MODE - Trigger mode is not available
            fpMin - Minimum exposure
            fpMax - Maximum exposure.

            C++ Equiv:
            unsigned int GetRingExposureRange(float * fpMin, float * fpMax);

            See Also:
            GetCapabilities GetNumberRingExposureTimes IsTriggerModeAvailable SetRingExposureTimes

        """
        cfpMin = c_float()
        cfpMax = c_float()
        ret = self.dll.GetRingExposureRange(byref(cfpMin), byref(cfpMax))
        return (ret, cfpMin.value, cfpMax.value)

    def GetSDK3Handle(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, Handle) = GetSDK3Handle()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            Handle -

            C++ Equiv:
            unsigned int GetSDK3Handle(int * Handle);

        """
        cHandle = c_int()
        ret = self.dll.GetSDK3Handle(byref(cHandle))
        return (ret, cHandle.value)

    def GetSensitivity(self, channel, horzShift, amplifier, pa):
        """
            Description:
            This function returns the sensitivity for a particular speed.

            Synopsis:
            (ret, sensitivity) = GetSensitivity(channel, horzShift, amplifier, pa)

            Inputs:
            channel - AD channel index.
            horzShift - Type of output amplifier.
            amplifier - Channel speed index.
            pa - PreAmp gain index.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Sensitivity returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid channel.
                DRV_P2INVALID - Invalid amplifier.
                DRV_P3INVALID - Invalid speed index.
                DRV_P4INVALID - Invalid gain.
            sensitivity - requested sensitivity.

            C++ Equiv:
            unsigned int GetSensitivity(int channel, int horzShift, int amplifier, int pa, float * sensitivity);

            See Also:
            GetCapabilities

            Note: Available only on iXon+ and Clara.

        """
        cchannel = c_int(channel)
        chorzShift = c_int(horzShift)
        camplifier = c_int(amplifier)
        cpa = c_int(pa)
        csensitivity = c_float()
        ret = self.dll.GetSensitivity(
            cchannel, chorzShift, camplifier, cpa, byref(csensitivity))
        return (ret, csensitivity.value)

    def GetShutterMinTimes(self):
        """
            Description:
            This function will return the minimum opening and closing times in milliseconds for the shutter on the current camera.

            Synopsis:
            (ret, minclosingtime, minopeningtime) = GetShutterMinTimes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Minimum times successfully returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Parameter is NULL.
                DRV_P2INVALID - Parameter is NULL
            minclosingtime - returns the minimum closing time in milliseconds that the shutter of the camera supports.
            minopeningtime - returns the minimum opening time in milliseconds that the shutter of the camera supports.

            C++ Equiv:
            unsigned int GetShutterMinTimes(int * minclosingtime, int * minopeningtime);

        """
        cminclosingtime = c_int()
        cminopeningtime = c_int()
        ret = self.dll.GetShutterMinTimes(
            byref(cminclosingtime), byref(cminopeningtime))
        return (ret, cminclosingtime.value, cminopeningtime.value)

    def GetSizeOfCircularBuffer(self):
        """
            Description:
            This function will return the maximum number of images the circular buffer can store based on the current acquisition settings.

            Synopsis:
            (ret, index) = GetSizeOfCircularBuffer()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Maximum number of images returned.
                DRV_NOT_INITIALIZED - System not initialized.
            index - returns the maximum number of images the circular buffer can store.

            C++ Equiv:
            unsigned int GetSizeOfCircularBuffer(long * index);

        """
        cindex = c_int()
        ret = self.dll.GetSizeOfCircularBuffer(byref(cindex))
        return (ret, cindex.value)

    def GetSlotBusDeviceFunction(self):
        """
            Description:
            THIS FUNCTION IS RESERVED

            Synopsis:
            (ret, dwslot, dwBus, dwDevice, dwFunction) = GetSlotBusDeviceFunction()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            dwslot -
            dwBus -
            dwDevice -
            dwFunction -

            C++ Equiv:
            unsigned int GetSlotBusDeviceFunction(DWORD * dwslot, DWORD * dwBus, DWORD * dwDevice, DWORD * dwFunction);

        """
        cdwslot = ()
        cdwBus = ()
        cdwDevice = ()
        cdwFunction = ()
        ret = self.dll.GetSlotBusDeviceFunction(
            byref(cdwslot), byref(cdwBus), byref(cdwDevice), byref(cdwFunction))
        return (ret, cdwslot.value, cdwBus.value, cdwDevice.value, cdwFunction.value)

    def GetSoftwareVersion(self):
        """
            Description:
            This function returns the Software version information for the microprocessor code and the driver.

            Synopsis:
            (ret, eprom, coffile, vxdrev, vxdver, dllrev, dllver) = GetSoftwareVersion()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Version information returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
            eprom - EPROM version
            coffile - COF file version
            vxdrev - Driver revision number
            vxdver - Driver version number
            dllrev - DLL revision number
            dllver - DLL version number

            C++ Equiv:
            unsigned int GetSoftwareVersion(unsigned int * eprom, unsigned int * coffile, unsigned int * vxdrev, unsigned int * vxdver, unsigned int * dllrev, unsigned int * dllver);

        """
        ceprom = c_uint()
        ccoffile = c_uint()
        cvxdrev = c_uint()
        cvxdver = c_uint()
        cdllrev = c_uint()
        cdllver = c_uint()
        ret = self.dll.GetSoftwareVersion(byref(ceprom), byref(ccoffile), byref(
            cvxdrev), byref(cvxdver), byref(cdllrev), byref(cdllver))
        return (ret, ceprom.value, ccoffile.value, cvxdrev.value, cvxdver.value, cdllrev.value, cdllver.value)

    def GetSpoolProgress(self):
        """
            Description:
            Deprecated see Note:
            This function will return information on the progress of the current spool operation. The value returned is the number of images that have been saved to disk during the current kinetic scan.

            Synopsis:
            (ret, index) = GetSpoolProgress()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Spool progress returned.
                DRV_NOT_INITIALIZED - System not initialized.
            index - returns the number of files saved to disk in the current kinetic scan.

            C++ Equiv:
            unsigned int GetSpoolProgress(long * index); // deprecated

            See Also:
            SetSpool

            Note: Deprecated by GetTotalNumberImagesAcquiredGetNumberHSSpeeds

        """
        cindex = c_int()
        ret = self.dll.GetSpoolProgress(byref(cindex))
        return (ret, cindex.value)

    def GetStartUpTime(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, time) = GetStartUpTime()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            time -

            C++ Equiv:
            unsigned int GetStartUpTime(float * time);

        """
        ctime = c_float()
        ret = self.dll.GetStartUpTime(byref(ctime))
        return (ret, ctime.value)

    def GetStatus(self):
        """
            Description:
            This function will return the current status of the Andor SDK system. This function should be called before an acquisition is started to ensure that it is IDLE and during an acquisition to monitor the process.

            Synopsis:
            (ret, status) = GetStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Status returned
                DRV_NOT_INITIALIZED - System not initialized
            status - current status:
                DRV_IDLE - waiting on instructions.
                DRV_TEMPCYCLE - Executing temperature cycle.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ACCUM_TIME_NOT_MET - Unable to meet Accumulate cycle time.
                DRV_KINETIC_TIME_NOT_MET - Unable to meet Kinetic cycle time.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_ACQ_BUFFER - Computer unable to read the data via the ISA slot at the required rate.
                DRV_SPOOLERROR - Overflow of the spool buffer.

            C++ Equiv:
            unsigned int GetStatus(int * status);

            See Also:
            SetTemperature StartAcquisition

            Note: If the status is one of the following:

        """
        cstatus = c_int()
        ret = self.dll.GetStatus(byref(cstatus))
        return (ret, cstatus.value)

    def GetTECStatus(self):
        """
            Description:
            This function will return if the TEC has overheated.

            Synopsis:
            (ret, piFlag) = GetTECStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - TEC status not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with card.
            piFlag - The status of the TEC:
                0 - Normal
                1 - Tripped

            C++ Equiv:
            unsigned int GetTECStatus(int * piFlag);

            See Also:
            SetTECEvent

        """
        cpiFlag = c_int()
        ret = self.dll.GetTECStatus(byref(cpiFlag))
        return (ret, cpiFlag.value)

    def GetTemperature(self):
        """
            Description:
            This function returns the temperature of the detector to the nearest degree. It also gives the status of cooling process.

            Synopsis:
            (ret, temperature) = GetTemperature()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_TEMP_OFF - Temperature is OFF.
                DRV_TEMP_STABILIZED - Temperature has stabilized at set point.
                DRV_TEMP_NOT_REACHED - Temperature has not reached set point.
                DRV_TEMP_DRIFT - Temperature had stabilized but has since drifted
                DRV_TEMP_NOT_STABILIZED - Temperature reached but not stabilized
            temperature - temperature of the detector

            C++ Equiv:
            unsigned int GetTemperature(int * temperature);

            See Also:
            GetTemperatureF SetTemperature CoolerON CoolerOFF GetTemperatureRange

        """
        ctemperature = c_int()
        ret = self.dll.GetTemperature(byref(ctemperature))
        return (ret, ctemperature.value)

    def GetTemperatureF(self):
        """
            Description:
            This function returns the temperature in degrees of the detector. It also gives the status of cooling process.

            Synopsis:
            (ret, temperature) = GetTemperatureF()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_TEMP_OFF - Temperature is OFF.
                DRV_TEMP_STABILIZED - Temperature has stabilized at set point.
                DRV_TEMP_NOT_REACHED - Temperature has not reached set point.
                DRV_TEMP_DRIFT - Temperature had stabilised but has since drifted
                DRV_TEMP_NOT_STABILIZED - Temperature reached but not stabilized
            temperature - temperature of the detector

            C++ Equiv:
            unsigned int GetTemperatureF(float * temperature);

            See Also:
            GetTemperature SetTemperature CoolerON CoolerOFF GetTemperatureRange

        """
        ctemperature = c_float()
        ret = self.dll.GetTemperatureF(byref(ctemperature))
        return (ret, ctemperature.value)

    def GetTemperaturePrecision(self):
        """
            Description:
            This function returns the number of decimal places to which the sensor temperature can be returned.

            Synopsis:
            (ret, precision) = GetTemperaturePrecision()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Temperature precision returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
            precision - number of decimal places

            C++ Equiv:
            unsigned int GetTemperaturePrecision(int * precision);

            See Also:
            GetTemperature GetTemperatureF SetTemperature CoolerON CoolerOFF

            Note: This function returns the number of decimal places to which the sensor temperature can be returned.

        """
        cprecision = c_int()
        ret = self.dll.GetTemperaturePrecision(byref(cprecision))
        return (ret, cprecision.value)

    def GetTemperatureRange(self):
        """
            Description:
            This function returns the valid range of temperatures in centigrade to which the detector can be cooled.

            Synopsis:
            (ret, mintemp, maxtemp) = GetTemperatureRange()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Temperature range returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
            mintemp - minimum temperature
            maxtemp - maximum temperature

            C++ Equiv:
            unsigned int GetTemperatureRange(int * mintemp, int * maxtemp);

            See Also:
            GetTemperature GetTemperatureF SetTemperature CoolerON CoolerOFF

        """
        cmintemp = c_int()
        cmaxtemp = c_int()
        ret = self.dll.GetTemperatureRange(byref(cmintemp), byref(cmaxtemp))
        return (ret, cmintemp.value, cmaxtemp.value)

    def GetTemperatureStatus(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, SensorTemp, TargetTemp, AmbientTemp, CoolerVolts) = GetTemperatureStatus()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            SensorTemp -
            TargetTemp -
            AmbientTemp -
            CoolerVolts -

            C++ Equiv:
            unsigned int GetTemperatureStatus(float * SensorTemp, float * TargetTemp, float * AmbientTemp, float * CoolerVolts);

        """
        cSensorTemp = c_float()
        cTargetTemp = c_float()
        cAmbientTemp = c_float()
        cCoolerVolts = c_float()
        ret = self.dll.GetTemperatureStatus(byref(cSensorTemp), byref(
            cTargetTemp), byref(cAmbientTemp), byref(cCoolerVolts))
        return (ret, cSensorTemp.value, cTargetTemp.value, cAmbientTemp.value, cCoolerVolts.value)

    def GetTotalNumberImagesAcquired(self):
        """
            Description:
            This function will return the total number of images acquired since the current acquisition started. If the camera is idle the value returned is the number of images acquired during the last acquisition.

            Synopsis:
            (ret, index) = GetTotalNumberImagesAcquired()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Number of acquired images returned.
                DRV_NOT_INITIALIZED - System not initialized.
            index - returns the total number of images acquired since the acquisition started.

            C++ Equiv:
            unsigned int GetTotalNumberImagesAcquired(long * index);

        """
        cindex = c_int()
        ret = self.dll.GetTotalNumberImagesAcquired(byref(cindex))
        return (ret, cindex.value)

    def GetTriggerLevelRange(self):
        """
            Description:
            This function returns the valid range of triggers in volts which the system can use.

            Synopsis:
            (ret, minimum, maximum) = GetTriggerLevelRange()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Levels returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Trigger levels not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - minimum has invalid memory address.
            minimum - minimum trigger voltage
            maximum - maximum trigger voltage

            C++ Equiv:
            unsigned int GetTriggerLevelRange(float * minimum, float * maximum);

            See Also:
            GetCapabilities SetTriggerLevel

        """
        cminimum = c_float()
        cmaximum = c_float()
        ret = self.dll.GetTriggerLevelRange(byref(cminimum), byref(cmaximum))
        return (ret, cminimum.value, cmaximum.value)

    def GetUSBDeviceDetails(self):
        """
            Description:
            This function returns details for the active USB system

            Synopsis:
            (ret, VendorID, ProductID, FirmwareVersion, SpecificationNumber) = GetUSBDeviceDetails()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - USB details returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_SUPPORTED - Not a USB system
            VendorID - USB camera vendor ID
            ProductID - USB camera product ID
            FirmwareVersion - USB camera firmware version
            SpecificationNumber - USB camera specification number

            C++ Equiv:
            unsigned int GetUSBDeviceDetails(WORD * VendorID, WORD * ProductID, WORD * FirmwareVersion, WORD * SpecificationNumber);

            See Also:
            GetCapabilities

        """
        cVendorID = c_short()
        cProductID = c_short()
        cFirmwareVersion = c_short()
        cSpecificationNumber = c_short()
        ret = self.dll.GetUSBDeviceDetails(byref(cVendorID), byref(
            cProductID), byref(cFirmwareVersion), byref(cSpecificationNumber))
        return (ret, cVendorID.value, cProductID.value, cFirmwareVersion.value, cSpecificationNumber.value)

    def GetVersionInfo(self, arr, ui32BufferLen):
        """
            Description:
            This function retrieves version information about different aspects of the Andor system.
            The information is copied into a passed string buffer.
            Currently, the version of the SDK and the Device Driver (USB or PCI) is supported.

            Synopsis:
            (ret, szVersionInfo) = GetVersionInfo(arr, ui32BufferLen)

            Inputs:
            arr - :
                AT_SDKVersion - requests the SDK version information
                AT_DeviceDriverVersion - requests the device driver version
            ui32BufferLen - The size of the passed character array,

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS     Information returned - Information returned
                DRV_NOT_INITIALIZED System not initialized - System not initialized
                DRV_P1INVALID - Invalid information type requested
                DRV_P2INVALID - Storage array pointer is NULL
                DRV_P3INVALID - Size of the storage array is zero
            szVersionInfo - A user allocated array of characters for storage of the information

            C++ Equiv:
            unsigned int GetVersionInfo(AT_VersionInfoId arr, char * szVersionInfo, at_u32 ui32BufferLen);

            See Also:
            GetHeadModel GetCameraSerialNumber GetCameraInformation GetCapabilities

        """
        carr = (arr)
        cszVersionInfo = create_string_buffer(ui32BufferLen)
        cui32BufferLen = c_uint(ui32BufferLen)
        ret = self.dll.GetVersionInfo(carr, cszVersionInfo, cui32BufferLen)
        return (ret, cszVersionInfo)

    def GetVerticalSpeed(self, index):
        """
            Description:
            Deprecated see Note:
            As your Andor system may be capable of operating at more than one vertical shift speed this function will return the actual speeds available. The value returned is in microseconds per pixel shift.

            Synopsis:
            (ret, speed) = GetVerticalSpeed(index)

            Inputs:
            index - speed required:
                0 - to GetNumberVerticalSpeedsGetNumberVerticalSpeeds()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING DRV_P1INVALID - Acquisition in progress.
            speed - speed in microseconds per pixel shift.

            C++ Equiv:
            unsigned int GetVerticalSpeed(int index, int * speed); // deprecated

            See Also:
            GetNumberVerticalSpeeds SetVerticalSpeed

            Note: Deprecated by GetVSSpeedGetVSSpeed.

        """
        cindex = c_int(index)
        cspeed = c_int()
        ret = self.dll.GetVerticalSpeed(cindex, byref(cspeed))
        return (ret, cspeed.value)

    def GetVirtualDMAAddress(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, Address1, Address2) = GetVirtualDMAAddress()

            Inputs:
            None

            Outputs:
            ret - Function Return Code
            Address1 -
            Address2 -

            C++ Equiv:
            unsigned int GetVirtualDMAAddress(void * Address1, void * Address2);

        """
        cAddress1 = c_void()
        cAddress2 = c_void()
        ret = self.dll.GetVirtualDMAAddress(byref(cAddress1), byref(cAddress2))
        return (ret, cAddress1.value, cAddress2.value)

    def GetVSAmplitudeFromString(self, text):
        """
            Description:
            This Function is used to get the index of the Vertical Clock Amplitude that corresponds to the string passed in.

            Synopsis:
            (ret, index) = GetVSAmplitudeFromString(text)

            Inputs:
            text - String to test "Normal" , "+1" , "+2" , "+3" , "+4"

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Vertical Clock Amplitude string Index returned
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid text.
                DRV_P2INVALID - Invalid index pointer.
            index - Returns the Index of the VSAmplitude that matches string passed in

            C++ Equiv:
            unsigned int GetVSAmplitudeFromString(char * text, int * index);

            See Also:
            GetVSAmplitudeString GetVSAmplitudeValue

        """
        ctext = text
        cindex = c_int()
        ret = self.dll.GetVSAmplitudeFromString(ctext, byref(cindex))
        return (ret, cindex.value)

    def GetVSAmplitudeString(self, index):
        """
            Description:
            This Function is used to get the Vertical Clock Amplitude string that corresponds to the index passed in.

            Synopsis:
            (ret, text) = GetVSAmplitudeString(index)

            Inputs:
            index - Index of VS amplitude required:
                0 - to GetNumberVSAmplitudes()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Vertical Clock Amplitude string returned
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Invalid text pointer.
            text - Returns string value of the VS Amplitude found at the index supplied

            C++ Equiv:
            unsigned int GetVSAmplitudeString(int index, char * text);

            See Also:
            GetVSAmplitudeFromString GetVSAmplitudeValue

        """
        cindex = c_int(index)
        ctext = create_string_buffer(64)
        ret = self.dll.GetVSAmplitudeString(cindex, ctext)
        return (ret, ctext)

    def GetVSAmplitudeValue(self, index):
        """
            Description:
            This Function is used to get the value of the Vertical Clock Amplitude found at the index passed in.

            Synopsis:
            (ret, value) = GetVSAmplitudeValue(index)

            Inputs:
            index - Index of VS amplitude required:
                0 - to GetNumberVSAmplitudes()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Vertical Clock Amplitude value returned
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Invalid value pointer.
            value - Returns Value of Vertical Clock Amplitude that matches index passed in

            C++ Equiv:
            unsigned int GetVSAmplitudeValue(int index, int * value);

            See Also:
            GetVSAmplitudeFromString GetVSAmplitudeString

        """
        cindex = c_int(index)
        cvalue = c_int()
        ret = self.dll.GetVSAmplitudeValue(cindex, byref(cvalue))
        return (ret, cvalue.value)

    def GetVSSpeed(self, index):
        """
            Description:
            As your Andor SDK system may be capable of operating at more than one vertical shift speed this function will return the actual speeds available. The value returned is in microseconds.

            Synopsis:
            (ret, speed) = GetVSSpeed(index)

            Inputs:
            index - speed required:
                0 - to GetNumberVSSpeeds()-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Speed returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
            speed - speed in microseconds per pixel shift.

            C++ Equiv:
            unsigned int GetVSSpeed(int index, float * speed);

            See Also:
            GetNumberVSSpeeds SetVSSpeed GetFastestRecommendedVSSpeed

        """
        cindex = c_int(index)
        cspeed = c_float()
        ret = self.dll.GetVSSpeed(cindex, byref(cspeed))
        return (ret, round(cspeed.value, 6))

    def GPIBReceive(self, id, address, size):
        """
            Description:
            This function reads data from a device until a byte is received with the EOI line asserted or until size bytes have been read.

            Synopsis:
            (ret, text) = GPIBReceive(id, address, size)

            Inputs:
            id - The interface board number:
                short - address: Address of device to send data
            address - The address to send the data to
            size - Number of characters to read

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data received.
                DRV_P3INVALID - Invalid pointer (e.g. NULL).  .Other errors may be returned by the GPIB device. Consult the help documentation supplied with these devices.
            text - The data to be sent

            C++ Equiv:
            unsigned int GPIBReceive(int id, short address, char * text, int size);

            See Also:
            GPIBSend

        """
        cid = c_int(id)
        caddress = c_short(address)
        ctext = create_string_buffer(size)
        csize = c_int(size)
        ret = self.dll.GPIBReceive(cid, caddress, ctext, csize)
        return (ret, ctext)

    def GPIBSend(self, id, address, text):
        """
            Description:
            This function initializes the GPIB by sending interface clear. Then the device described by address is put in a listen-active state. Finally the string of characters, text, is sent to the device with a newline character and with the EOI line asserted after the final character.

            Synopsis:
            ret = GPIBSend(id, address, text)

            Inputs:
            id - The interface board number:
                short - address: Address of device to send data
            address - The GPIB address to send data to
            text - The data to send

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data sent.
                DRV_P3INVALID - Invalid pointer (e.g. NULL). The GPIB device may return other errors. Consult the help documentation supplied with these devices.

            C++ Equiv:
            unsigned int GPIBSend(int id, short address, char * text);

            See Also:
            GPIBReceive

        """
        cid = c_int(id)
        caddress = c_short(address)
        ctext = text
        ret = self.dll.GPIBSend(cid, caddress, ctext)
        return (ret)

    def I2CBurstRead(self, i2cAddress, nBytes):
        """
            Description:
            This function will read a specified number of bytes from a chosen device attached to the I2C data bus.

            Synopsis:
            (ret, data) = I2CBurstRead(i2cAddress, nBytes)

            Inputs:
            i2cAddress - The address of the device to read from.
            nBytes - The number of bytes to read from the device.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Read successful.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_INIERROR - Unable to load DETECTOR.INI.
                DRV_COFERROR - Unable to load *.COF.
                DRV_FLEXERROR - Unable to load *.RBF.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_I2CDEVNOTFOUND - Could not find the specified device.
                DRV_I2CTIMEOUT - Timed out reading from device.
                DRV_UNKNOWN_FUNC - Unknown function, incorrect cof file.
            data - The data read from the device.

            C++ Equiv:
            unsigned int I2CBurstRead(BYTE i2cAddress, long nBytes, BYTE * data);

            See Also:
            I2CBurstWrite I2CRead I2CWrite I2CReset

        """
        ci2cAddress = c_ubyte(i2cAddress)
        cnBytes = c_int(nBytes)
        cdata = (c_ubyte * nBytes)()
        ret = self.dll.I2CBurstRead(ci2cAddress, cnBytes, cdata)
        return (ret, np.ctypeslib.as_array(cdata))


    def I2CBurstWrite(self, i2cAddress, nBytes, data):
        """
            Description:
            This function will write a specified number of bytes to a chosen device attached to the I2C data bus.

            Synopsis:
            ret = I2CBurstWrite(i2cAddress, nBytes, data)

            Inputs:
            i2cAddress - The address of the device to write to.
            nBytes - The number of bytes to write to the device.
            data - The data to write to the device.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Write successful.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_INIERROR - Unable to load DETECTOR.INI.
                DRV_COFERROR - Unable to load *.COF.
                DRV_FLEXERROR - Unable to load *.RBF.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_I2CDEVNOTFOUND - Could not find the specified device.
                DRV_I2CTIMEOUT - Timed out reading from device.
                DRV_UNKNOWN_FUNC - Unknown function, incorrect cof file.

            C++ Equiv:
            unsigned int I2CBurstWrite(BYTE i2cAddress, long nBytes, BYTE * data);

            See Also:
            I2CBurstRead I2CRead I2CWrite I2CReset

        """
        ci2cAddress = c_ubyte(i2cAddress)
        cnBytes = c_int(nBytes)
        cdata = (c_ubyte * nBytes)(data)
        ret = self.dll.I2CBurstWrite(ci2cAddress, cnBytes, cdata)
        return (ret)

    def I2CRead(self, deviceID, intAddress):
        """
            Description:
            This function will read a single byte from the chosen device.

            Synopsis:
            (ret, pdata) = I2CRead(deviceID, intAddress)

            Inputs:
            deviceID - The device to read from.
            intAddress - The internal address of the device to be read from.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Read successful.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_INIERROR - Unable to load DETECTOR.INI.
                DRV_COFERROR - Unable to load *.COF.
                DRV_FLEXERROR - Unable to load *.RBF.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_I2CDEVNOTFOUND - Could not find the specified device.
                DRV_I2CTIMEOUT - Timed out reading from device.
                DRV_UNKNOWN_FUNC - Unknown function, incorrect cof file.
            pdata - The byte read from the device.

            C++ Equiv:
            unsigned int I2CRead(BYTE deviceID, BYTE intAddress, BYTE * pdata);

            See Also:
            I2CBurstWrite I2CBurstRead I2CWrite I2CReset

        """
        cdeviceID = c_ubyte(deviceID)
        cintAddress = c_ubyte(intAddress)
        cpdata = c_ubyte()
        ret = self.dll.I2CRead(cdeviceID, cintAddress, byref(cpdata))
        return (ret, cpdata.value)

    def I2CReset(self):
        """
            Description:
            This function will reset the I2C data bus.

            Synopsis:
            ret = I2CReset()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Reset successful.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_INIERROR - Unable to load DETECTOR.INI.
                DRV_COFERROR - Unable to load *.COF.
                DRV_FLEXERROR - Unable to load *.RBF.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_I2CTIMEOUT - Timed out reading from device.
                DRV_UNKNOWN_FUNC - Unknown function, incorrect cof file.

            C++ Equiv:
            unsigned int I2CReset(void);

            See Also:
            I2CBurstWrite I2CBurstRead I2CWrite

        """
        ret = self.dll.I2CReset()
        return (ret)

    def I2CWrite(self, deviceID, intAddress, data):
        """
            Description:
            This function will write a single byte to the chosen device.

            Synopsis:
            ret = I2CWrite(deviceID, intAddress, data)

            Inputs:
            deviceID - The device to write to.
            intAddress - The internal address of the device to write to.
            data - The byte to be written to the device.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Write successful.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_INIERROR - Unable to load DETECTOR.INI.
                DRV_COFERROR - Unable to load *.COF.
                DRV_FLEXERROR - Unable to load *.RBF.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_I2CDEVNOTFOUND - Could not find the specified device.
                DRV_I2CTIMEOUT - Timed out reading from device.
                DRV_UNKNOWN_FUNC - Unknown function, incorrect cof file.

            C++ Equiv:
            unsigned int I2CWrite(BYTE deviceID, BYTE intAddress, BYTE data);

            See Also:
            I2CBurstWrite I2CBurstRead I2CRead I2CReset

        """
        cdeviceID = c_ubyte(deviceID)
        cintAddress = c_ubyte(intAddress)
        cdata = c_ubyte(data)
        ret = self.dll.I2CWrite(cdeviceID, cintAddress, cdata)
        return (ret)

    def IdAndorDll(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = IdAndorDll()

            Inputs:
            None

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int IdAndorDll(void);

        """
        ret = self.dll.IdAndorDll()
        return (ret)

    def InAuxPort(self, port):
        """
            Description:
            This function returns the state of the TTL Auxiliary Input Port on the Andor plug-in card.

            Synopsis:
            (ret, state) = InAuxPort(port)

            Inputs:
            port - Number of AUX in port on Andor card.  Valid Values: 1 to 4

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - AUX read.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid port id.
            state - current state of port:
                0 - OFF/LOW
                all others - ON/HIGH

            C++ Equiv:
            unsigned int InAuxPort(int port, int * state);

            See Also:
            OutAuxPort

        """
        cport = c_int(port)
        cstate = c_int()
        ret = self.dll.InAuxPort(cport, byref(cstate))
        return (ret, cstate.value)

    def Initialize(self, dir):
        """
            Description:
            This function will initialize the Andor SDK System.
            As part of the initialization procedure on some cameras (i.e. Classic, iStar and earlier iXion)
            the DLL will need access to a DETECTOR.INI which contains information relating to the detector head, number pixels, readout speeds etc.
            If your system has multiple cameras then see the section Controlling multiple cameras

            Synopsis:
            ret = Initialize(dir)

            Inputs:
            dir - Path to the directory containing the files

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS DRV_VXDNOTINSTALLED - Initialization successful.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_INIERROR - Unable to load DETECTOR.INI.
                DRV_COFERROR - Unable to load *.COF.
                DRV_FLEXERROR - Unable to load *.RBF.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_ERROR_FILELOAD - Unable to load *.COF or *.RBF files.
                DRV_ERROR_PAGELOCK - Unable to acquire lock on requested memory.
                DRV_USBERROR - Unable to detect USB device or not USB2.0.
                DRV_ERROR_NOCAMERA - No camera found

            C++ Equiv:
            unsigned int Initialize(char * dir);

            See Also:
            GetAvailableCameras SetCurrentCamera GetCurrentCamera

        """
        cdir = dir.encode('utf-8')
        ret = self.dll.Initialize(cdir)
        return (ret)

    def InitializeDevice(self, dir):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = InitializeDevice(dir)

            Inputs:
            dir -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int InitializeDevice(char * dir);

        """
        cdir = dir.encode('utf-8')
        ret = self.dll.InitializeDevice(cdir)
        return (ret)

    def IsAmplifierAvailable(self, iamp):
        """
            Description:
            This function checks if the hardware and current settings permit the use of the specified amplifier.

            Synopsis:
            ret = IsAmplifierAvailable(iamp)

            Inputs:
            iamp - amplifier to check.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Amplifier available
                DRV_NOT_INITIALIZED - System not initialized
                DRV_INVALID_AMPLIFIER - Not a valid amplifier

            C++ Equiv:
            unsigned int IsAmplifierAvailable(int iamp);

            See Also:
            SetHSSpeed

        """
        ciamp = c_int(iamp)
        ret = self.dll.IsAmplifierAvailable(ciamp)
        return (ret)

    def IsCoolerOn(self):
        """
            Description:
            This function checks the status of the cooler.

            Synopsis:
            (ret, iCoolerStatus) = IsCoolerOn()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Status returned.
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - Parameter is NULL
            iCoolerStatus - iCoolerStatus0: Cooler is OFF.:
                1 - 1 Cooler is ON.

            C++ Equiv:
            unsigned int IsCoolerOn(int * iCoolerStatus);

            See Also:
            CoolerON CoolerOFF

        """
        ciCoolerStatus = c_int()
        ret = self.dll.IsCoolerOn(byref(ciCoolerStatus))
        return (ret, ciCoolerStatus.value)

    def IsCountConvertModeAvailable(self, mode):
        """
            Description:
            This function checks if the hardware and current settings permit the use of the specified Count Convert mode.

            Synopsis:
            ret = IsCountConvertModeAvailable(mode)

            Inputs:
            mode - Count Convert mode to be checked

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Count Convert mode available.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Count Convert not supported on this camera
                DRV_INVALID_COUNTCONVERT_MODE - Count Convert mode not available with current acquisition settings

            C++ Equiv:
            unsigned int IsCountConvertModeAvailable(int mode);

            See Also:
            GetCapabilities SetCountConvertMode SetCountConvertWavelength

        """
        cmode = c_int(mode)
        ret = self.dll.IsCountConvertModeAvailable(cmode)
        return (ret)

    def IsInternalMechanicalShutter(self):
        """
            Description:
            This function checks if an iXon camera has a mechanical shutter installed.


            Synopsis:
            (ret, internalShutter) = IsInternalMechanicalShutter()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Internal shutter state returned
                DRV_NOT_AVAILABLE - Not an iXon Camera
                DRV_P1INVALID - Parameter is NULL
            internalShutter - Status of internal shutter:
                0 - Mechanical shutter not installed.
                1 - Mechanical shutter installed.

            C++ Equiv:
            unsigned int IsInternalMechanicalShutter(int * internalShutter);

            Note: Available only on iXon

        """
        cinternalShutter = c_int()
        ret = self.dll.IsInternalMechanicalShutter(byref(cinternalShutter))
        return (ret, cinternalShutter.value)

    def IsPreAmpGainAvailable(self, channel, amplifier, index, pa):
        """
            Description:
            This function checks that the AD channel exists, and that the amplifier, speed and gain are available for the AD channel.

            Synopsis:
            (ret, status) = IsPreAmpGainAvailable(channel, amplifier, index, pa)

            Inputs:
            channel - AD channel index.
            amplifier - Type of output amplifier.
            index - Channel speed index.
            pa - PreAmpGain index.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - PreAmpGain status returned.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid channel.
                DRV_P2INVALID - Invalid amplifier.
                DRV_P3INVALID - Invalid speed index.
                DRV_P4INVALID - Invalid gain.
            status - PreAmpGain Status:
                0 - PreAmpGain not available.
                1 - PreAmpGain available.

            C++ Equiv:
            unsigned int IsPreAmpGainAvailable(int channel, int amplifier, int index, int pa, int * status);

            See Also:
            GetNumberPreAmpGains GetPreAmpGain SetPreAmpGain

            Note: Available only on iXon.

        """
        cchannel = c_int(channel)
        camplifier = c_int(amplifier)
        cindex = c_int(index)
        cpa = c_int(pa)
        cstatus = c_int()
        ret = self.dll.IsPreAmpGainAvailable(
            cchannel, camplifier, cindex, cpa, byref(cstatus))
        return (ret, cstatus.value)

    def IsReadoutFlippedByAmplifier(self, amplifier):
        """
            Description:
            On cameras with multiple amplifiers the frame readout may be flipped.  This function can be used to determine if this is the case.

            Synopsis:
            (ret, flipped) = IsReadoutFlippedByAmplifier(amplifier)

            Inputs:
            amplifier - amplifier to check

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Flipped status returned
                DRV_NOT_INITIALIZED - System not initialized
                DRV_INVALID_AMPLIFIER - Not a valid amplifier
            flipped -  0: Frame reads left to right.  1: Frame reads right to left.

            C++ Equiv:
            unsigned int IsReadoutFlippedByAmplifier(int amplifier, int * flipped);

            See Also:
            GetNumberAmp

            Note: On cameras with multiple amplifiers the frame readout may be flipped.  This function can be used to determine if this is the case.

        """
        camplifier = c_int(amplifier)
        cflipped = c_int()
        ret = self.dll.IsReadoutFlippedByAmplifier(camplifier, byref(cflipped))
        return (ret, cflipped.value)

    def IsTriggerModeAvailable(self, iTriggerMode):
        """
            Description:
            This function checks if the hardware and current settings permit the use of the specified trigger mode.

            Synopsis:
            ret = IsTriggerModeAvailable(iTriggerMode)

            Inputs:
            iTriggerMode - Trigger mode to check.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Trigger mode available
                DRV_NOT_INITIALIZED - System not initialize
                DRV_INVALID_MODE - Not a valid mode

            C++ Equiv:
            unsigned int IsTriggerModeAvailable(int iTriggerMode);

            See Also:
            SetTriggerMode

        """
        ciTriggerMode = c_int(iTriggerMode)
        ret = self.dll.IsTriggerModeAvailable(ciTriggerMode)
        return (ret)

    def Merge(self, nOrder, nPoint, nPixel, fit, hbin):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, arr, coeff, output, start, step_Renamed) = Merge(nOrder, nPoint, nPixel, fit, hbin)

            Inputs:
            nOrder -
            nPoint -
            nPixel -
            fit -
            hbin -

            Outputs:
            ret - Function Return Code
            arr -
            coeff -
            output -
            start -
            step_Renamed -

            C++ Equiv:
            unsigned int Merge(const at_32 * arr, long nOrder, long nPoint, long nPixel, float * coeff, long fit, long hbin, at_32 * output, float * start, float * step_Renamed);

        """
        carr = c_int()
        cnOrder = c_int(nOrder)
        cnPoint = c_int(nPoint)
        cnPixel = c_int(nPixel)
        ccoeff = c_float()
        cfit = c_int(fit)
        chbin = c_int(hbin)
        coutput = c_int()
        cstart = c_float()
        cstep_Renamed = c_float()
        ret = self.dll.Merge(byref(carr), cnOrder, cnPoint, cnPixel, byref(
            ccoeff), cfit, chbin, byref(coutput), byref(cstart), byref(cstep_Renamed))
        return (ret, carr.value, ccoeff.value, coutput.value, cstart.value, cstep_Renamed.value)

    def OA_AddMode(self, uiModeNameLen, pcModeDescription, uiModeDescriptionLen):
        """
            Description:
            This function will add a mode name and description to memory.  Note that this will not add the mode to file, a subsequent call to OA_WriteToFile must be made.

            Synopsis:
            (ret, pcModeName) = OA_AddMode(uiModeNameLen, pcModeDescription, uiModeDescriptionLen)

            Inputs:
            uiModeNameLen - Mode name string length.
            pcModeDescription - A description of the user defined mode.
            uiModeDescriptionLen - Mode Description string length.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS            DRV_P1INVALID           DRV_P3INVALID DRV_OA_INVALID_STRING_LENGTH - All parameters accepted                                Null mode name.                                        Null mode description.                                            One or more parameters have an invalid length, i.e. > 255.
                DRV_OA_INVALID_NAMING - Mode and description have the same name, this is not valid.
                DRV_OA_MODE_BUFFER_FULL        DRV_OA_INVALID_CHARS_IN_NAME - Number of modes exceeds limit.             Mode name and/or description contain invalid characters.
                DRV_OA_MODE_ALREADY_EXISTS - Mode name already exists in the file.
                DRV_OA_INVALID_CHARS_IN_NAME - Invalid charcters in Mode Name or Mode Description
            pcModeName - A name for the mode to be defined.

            C++ Equiv:
            unsigned int OA_AddMode(char * pcModeName, int uiModeNameLen, char * pcModeDescription, int uiModeDescriptionLen);

            See Also:
            OA_DeleteMode OA_WriteToFile

        """
        cpcModeName = create_string_buffer(uiModeNameLen)
        cuiModeNameLen = c_int(uiModeNameLen)
        cpcModeDescription = pcModeDescription
        cuiModeDescriptionLen = c_int(uiModeDescriptionLen)
        ret = self.dll.OA_AddMode(
            cpcModeName, cuiModeNameLen, cpcModeDescription, cuiModeDescriptionLen)
        return (ret, cpcModeName)

    def OA_DeleteMode(self, pcModeName, uiModeNameLen):
        """
            Description:
            This function will remove a mode from memory.  To permanently remove a mode from file, call OA_WriteToFile after OA_DeleteMode.  The Preset file will not be affected.

            Synopsis:
            ret = OA_DeleteMode(pcModeName, uiModeNameLen)

            Inputs:
            pcModeName - The name of the mode to be removed.
            uiModeNameLen - Mode name string length.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - DRV_OA_MODE_DOES_NOT_EXIST
                DRV_P1INVALID - All parameters accepted                                Null mode name.
                DRV_OA_INVALID_STRING_LENGTH - The mode name parameter has an invalid  length, i.e. > 256.

            C++ Equiv:
            unsigned int OA_DeleteMode(const char * pcModeName, int uiModeNameLen);

            See Also:
            OA_AddMode OA_WriteToFile

        """
        cpcModeName = pcModeName
        cuiModeNameLen = c_int(uiModeNameLen)
        ret = self.dll.OA_DeleteMode(cpcModeName, cuiModeNameLen)
        return (ret)

    def OA_EnableMode(self, pcModeName):
        """
            Description:
            This function will set all the parameters associated with the specified mode to be used for all subsequent acquisitions.  The mode specified by the user must be in either the Preset file or the User defined file.

            Synopsis:
            ret = OA_EnableMode(pcModeName)

            Inputs:
            pcModeName - The mode to be used for all subsequent acquisitions.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted
                DRV_P1INVALID - Null mode name.
                DRV_OA_MODE_DOES_NOT_EXIST - Mode name does not exist.
                DRV_OA_CAMERA_NOT_SUPPORTED - Camera not supported.

            C++ Equiv:
            unsigned int OA_EnableMode(const char * pcModeName);

            See Also:
            OA_AddMode

        """
        cpcModeName = pcModeName
        ret = self.dll.OA_EnableMode(cpcModeName)
        return (ret)

    def OA_GetFloat(self, pcModeName, pcModeParam):
        """
            Description:
            This function is used to get the values for floating point type acquisition parameters.
            Values are retrieved from memory for the specified mode name.

            Synopsis:
            (ret, fFloatValue) = OA_GetFloat(pcModeName, pcModeParam)

            Inputs:
            pcModeName - The name of the mode for which an acquisition parameter will be retrieved.
            pcModeParam - The name of the acquisition parameter for which a value will be retrieved.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted
                DRV_P1INVALID - Null mode parameter.
                DRV_P2INVALID - Null mode parameter.
                DRV_P3INVALID - Null float value.
            fFloatValue - The value of the acquisition parameter.

            C++ Equiv:
            unsigned int OA_GetFloat(const char * pcModeName, const char * pcModeParam, float * fFloatValue);

            See Also:
            OA_SetFloat

        """
        cpcModeName = pcModeName
        cpcModeParam = pcModeParam
        cfFloatValue = c_float()
        ret = self.dll.OA_GetFloat(
            cpcModeName, cpcModeParam, byref(cfFloatValue))
        return (ret, cfFloatValue.value)

    def OA_GetInt(self, pcModeName, pcModeParam):
        """
            Description:
            This function is used to get the values for integer type acquisition parameters. Values  are retrieved from memory for the specified mode name.


            Synopsis:
            (ret, iintValue) = OA_GetInt(pcModeName, pcModeParam)

            Inputs:
            pcModeName - The name of the mode for which an acquisition parameter
            pcModeParam - The name of the acquisition parameter for which a value

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null mode parameter.
                DRV_P3INVALID - Null integer value.
            iintValue - The buffer to return the value of the acquisition.

            C++ Equiv:
            unsigned int OA_GetInt(const char * pcModeName, const char * pcModeParam, int * iintValue);

            See Also:
            OA_SetInt

        """
        cpcModeName = pcModeName
        cpcModeParam = pcModeParam
        ciintValue = c_int()
        ret = self.dll.OA_GetInt(cpcModeName, cpcModeParam, byref(ciintValue))
        return (ret, ciintValue.value)

    def OA_GetModeAcqParams(self, pcModeName):
        """
            Description:
            This function will return all acquisition parameters associated with the specified mode.  The mode specified by the user must be in either the Preset file or the User defined file.  The user must allocate enough memory for all of the acquisition parameters.

            Synopsis:
            (ret, pcListOfParams) = OA_GetModeAcqParams(pcModeName)

            Inputs:
            pcModeName - The mode for which all acquisition parameters must be returned.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null mode parameter.
                DRV_OA_NO_USER_DATA - No data for selected mode.
            pcListOfParams - A user allocated array of characters for storage of the acquisition parameters.  Parameters will be delimited by a ','.

            C++ Equiv:
            unsigned int OA_GetModeAcqParams(const char * pcModeName, char * pcListOfParams);

            See Also:
            OA_GetNumberOfAcqParams

        """
        cpcModeName = pcModeName
        cpcListOfParams = create_string_buffer(MAX_PATH)
        ret = self.dll.OA_GetModeAcqParams(cpcModeName, cpcListOfParams)
        return (ret, cpcListOfParams)

    def OA_GetNumberOfAcqParams(self, pcModeName):
        """
            Description:
            This function will return the parameters associated with a specified mode.  The mode must be present in either the Preset file or the User defined file.

            Synopsis:
            (ret, puiNumberOfParams) = OA_GetNumberOfAcqParams(pcModeName)

            Inputs:
            pcModeName - The mode to search for a list of acquisition parameters.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null number of parameters.
                DRV_OA_NULL_ERROR - Invalid pointer.
            puiNumberOfParams - The number of acquisition parameters for the specified mode.

            C++ Equiv:
            unsigned int OA_GetNumberOfAcqParams(const char * pcModeName, unsigned int * puiNumberOfParams);

            See Also:
            OA_GetModeAcqParams

        """
        cpcModeName = pcModeName
        cpuiNumberOfParams = c_uint()
        ret = self.dll.OA_GetNumberOfAcqParams(
            cpcModeName, byref(cpuiNumberOfParams))
        return (ret, cpuiNumberOfParams.value)

    def OA_GetNumberOfPreSetModes(self):
        """
            Description:
            This function will return the number of modes defined in the Preset file.  The Preset file must exist.

            Synopsis:
            (ret, puiNumberOfModes) = OA_GetNumberOfPreSetModes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null number of modes.
                DRV_OA_NULL_ERROR - Invalid pointer.
                DRV_OA_BUFFER_FULL - Number of modes exceeds limit.
            puiNumberOfModes - The number of modes in the Andor file.

            C++ Equiv:
            unsigned int OA_GetNumberOfPreSetModes(unsigned int * puiNumberOfModes);

            See Also:
            OA_GetPreSetModeNames

        """
        cpuiNumberOfModes = c_uint()
        ret = self.dll.OA_GetNumberOfPreSetModes(byref(cpuiNumberOfModes))
        return (ret, cpuiNumberOfModes.value)

    def OA_GetNumberOfUserModes(self):
        """
            Description:
            This function will return the number of modes defined in the User file.  The user defined file must exist.

            Synopsis:
            (ret, puiNumberOfModes) = OA_GetNumberOfUserModes()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null number of modes.
                DRV_OA_NULL_ERROR - Invalid pointer.
                DRV_OA_BUFFER_FULL - Number of modes exceeds limit.
            puiNumberOfModes - The number of modes in the user file.

            C++ Equiv:
            unsigned int OA_GetNumberOfUserModes(unsigned int * puiNumberOfModes);

            See Also:
            OA_GetUserModeNames

        """
        cpuiNumberOfModes = c_uint()
        ret = self.dll.OA_GetNumberOfUserModes(byref(cpuiNumberOfModes))
        return (ret, cpuiNumberOfModes.value)

    def OA_GetPreSetModeNames(self):
        """
            Description:
            This function will return the available mode names from the Preset file.  The mode and the Preset file must exist.  The user must allocate enough memory for all of the acquisition parameters.

            Synopsis:
            (ret, pcListOfModes) = OA_GetPreSetModeNames()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null list of modes.
                DRV_OA_NULL_ERROR - Invalid pointer.
            pcListOfModes - A user allocated array of characters for storage of the mode names.  Mode names will be delimited by a ','.

            C++ Equiv:
            unsigned int OA_GetPreSetModeNames(char * pcListOfModes);

            See Also:
            OA_GetNumberOfPreSetModes

        """
        cpcListOfModes = create_string_buffer(MAX_PATH)
        ret = self.dll.OA_GetPreSetModeNames(cpcListOfModes)
        return (ret, cpcListOfModes)

    def OA_GetString(self, pcModeName, pcModeParam, uiStringLen):
        """
            Description:
            This function is used to get the values for string type acquisition parameters.  Values
            are retrieved from memory for the specified mode name.

            Synopsis:
            (ret, pcStringValue) = OA_GetString(pcModeName, pcModeParam, uiStringLen)

            Inputs:
            pcModeName - The name of the mode for which an acquisition parameter  will be retrieved.
            pcModeParam - The name of the acquisition parameter for which a value will be retrieved.
            uiStringLen - The length of the buffer.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null mode parameter.
                DRV_P3INVALID - Null string value.
                DRV_P4INVALID - Invalid string length
            pcStringValue - The buffer to return the value of the acquisition parameter.

            C++ Equiv:
            unsigned int OA_GetString(const char * pcModeName, const char * pcModeParam, char * pcStringValue, const int uiStringLen);

            See Also:
            OA_SetString

        """
        cpcModeName = pcModeName
        cpcModeParam = pcModeParam
        cpcStringValue = create_string_buffer(uiStringLen)
        cuiStringLen = c_int(uiStringLen)
        ret = self.dll.OA_GetString(
            cpcModeName, cpcModeParam, cpcStringValue, cuiStringLen)
        return (ret, cpcStringValue)

    def OA_GetUserModeNames(self):
        """
            Description:
            This function will return the available mode names from a User defined file.  The mode and the User defined file must exist.  The user must allocate enough memory for all of the acquisition parameters.

            Synopsis:
            (ret, pcListOfModes) = OA_GetUserModeNames()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null list of modes.
                DRV_OA_NULL_ERROR - Invalid pointer.
            pcListOfModes - A user allocated array of characters for storage of the mode names.  Mode names will be delimited by a ','.

            C++ Equiv:
            unsigned int OA_GetUserModeNames(char * pcListOfModes);

            See Also:
            OA_GetNumberOfUserModes

        """
        cpcListOfModes = create_string_buffer(MAX_PATH)
        ret = self.dll.OA_GetUserModeNames(cpcListOfModes)
        return (ret, cpcListOfModes)

    def OA_Initialize(self, pcFilename, uiFileNameLen):
        """
            Description:
            This function will Initialize the OptAcquire settings from a Preset file and a User defined file if it exists.

            Synopsis:
            ret = OA_Initialize(pcFilename, uiFileNameLen)

            Inputs:
            pcFilename - The name of a user xml file.  If the file exists then data will be read from the file.  If the file does not exist the file name may be used when the user calls WriteToFile().
            uiFileNameLen - The length of the filename.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null filename.
                DRV_OA_CAMERA_NOT_SUPPORTED - Camera not supported.
                DRV_OA_GET_CAMERA_ERROR - Unable to retrieve information about the
                DRV_OA_INVALID_STRING_LENGTH - model of the Camera.
                DRV_OA_ANDOR_FILE_NOT_LOADED - The parameter has an invalid length, i.e. > 255.
                DRV_OA_USER_FILE_NOT_LOADED - Preset Andor file failed to load.
                DRV_OA_FILE_ACCESS_ERROR - Supplied User file failed to load.
                DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED - Failed to determine status of file.

            C++ Equiv:
            unsigned int OA_Initialize(const char * pcFilename, int uiFileNameLen);

            See Also:
            OA_WriteToFile

        """
        cpcFilename = pcFilename
        cuiFileNameLen = c_int(uiFileNameLen)
        ret = self.dll.OA_Initialize(cpcFilename, cuiFileNameLen)
        return (ret)

    def OA_SetFloat(self, pcModeName, pcModeParam, fFloatValue):
        """
            Description:
            This function is used to set values for floating point type acquisition parameters where
            the new values are stored in memory.  To commit changes to file call WriteToFile().

            Synopsis:
            ret = OA_SetFloat(pcModeName, pcModeParam, fFloatValue)

            Inputs:
            pcModeName - The name of the mode for which an acquisition parameter will be edited.
            pcModeParam - The name of the acquisition parameter to be edited.
            fFloatValue - The value to assign to the acquisition parameter.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null mode parameter.
                DRV_OA_INVALID_STRING_LENGTH - One or more of the string parameters has an invalid length, i.e. > 255.
                DRV_OA_MODE_DOES_NOT_EXIST - The Mode does not exist.

            C++ Equiv:
            unsigned int OA_SetFloat(const char * pcModeName, const char * pcModeParam, const float fFloatValue);

            See Also:
            OA_GetFloat OA_EnableMode OA_WriteToFile

        """
        cpcModeName = pcModeName
        cpcModeParam = pcModeParam
        cfFloatValue = c_float(fFloatValue)
        ret = self.dll.OA_SetFloat(cpcModeName, cpcModeParam, cfFloatValue)
        return (ret)

    def OA_SetInt(self, pcModeName, pcModeParam, iintValue):
        """
            Description:
            This function is used to set values for integer type acquisition parameters where the
            new values are stored in memory.  To commit changes to file call WriteToFile().

            Synopsis:
            ret = OA_SetInt(pcModeName, pcModeParam, iintValue)

            Inputs:
            pcModeName - The name of the mode for which an acquisition parameter will be edited.
            pcModeParam - The name of the acquisition parameter to be edited.
            iintValue - The value to assign to the acquisition parameter.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null mode parameter.
                DRV_OA_INVALID_STRING_LENGTH - One or more of the string parameters has an invalid length, i.e. > 255.
                DRV_OA_MODE_DOES_NOT_EXIST - The Mode does not exist.

            C++ Equiv:
            unsigned int OA_SetInt(const char * pcModeName, const char * pcModeParam, const int iintValue);

            See Also:
            OA_GetInt OA_EnableMode OA_WriteToFile

        """
        cpcModeName = pcModeName
        cpcModeParam = pcModeParam
        ciintValue = c_int(iintValue)
        ret = self.dll.OA_SetInt(cpcModeName, cpcModeParam, ciintValue)
        return (ret)

    def OA_SetString(self, pcModeName, pcModeParam, pcStringValue, uiStringLen):
        """
            Description:
            This function is used to set values for string type acquisition parameters where the
            new values are stored in memory.  To commit changes to file call WriteToFile().

            Synopsis:
            ret = OA_SetString(pcModeName, pcModeParam, pcStringValue, uiStringLen)

            Inputs:
            pcModeName - The name of the mode for which an acquisition parameter is to be edited.
            pcModeParam - The name of the acquisition parameter to be edited.
            pcStringValue - The value to assign to the acquisition parameter.
            uiStringLen - The length of the input string.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null mode name.
                DRV_P2INVALID - Null mode parameter.
                DRV_P3INVALID - Null string value.
                DRV_P4INVALID - Invalid string length
                DRV_OA_INVALID_STRING_LENGTH - One or more of the string parameters has an invalid length, i.e. > 255.
                DRV_OA_MODE_DOES_NOT_EXIST - The Mode does not exist.

            C++ Equiv:
            unsigned int OA_SetString(const char * pcModeName, const char * pcModeParam, char * pcStringValue, const int uiStringLen);

            See Also:
            OA_GetString OA_EnableMode OA_WriteToFile

        """
        cpcModeName = pcModeName
        cpcModeParam = pcModeParam
        cpcStringValue = pcStringValue
        cuiStringLen = c_int(uiStringLen)
        ret = self.dll.OA_SetString(
            cpcModeName, cpcModeParam, cpcStringValue, cuiStringLen)
        return (ret)

    def OA_WriteToFile(self, pcFileName, uiFileNameLen):
        """
            Description:
            This function will write a User defined list of modes to the User file.  The Preset file will not be affected.

            Synopsis:
            ret = OA_WriteToFile(pcFileName, uiFileNameLen)

            Inputs:
            pcFileName - The name of the file to be written to.
            uiFileNameLen - File name string length.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_P1INVALID - Null filename
                DRV_OA_INVALID_STRING_LENGTH - One or more of the string parameters has an invalid length, i.e. > 255.
                DRV_OA_INVALID_FILE - Data cannot be written to the Preset Andor file.
                DRV_ERROR_FILESAVE - Failed to save data to file.
                DRV_OA_FILE_HAS_BEEN_MODIFIED - File to be written to has been modified since last write, local copy of file may not be the same.
                DRV_OA_INVALID_CHARS_IN_NAME - File name contains invalid characters.

            C++ Equiv:
            unsigned int OA_WriteToFile(const char * pcFileName, int uiFileNameLen);

            See Also:
            OA_AddMode OA_DeleteMode

        """
        cpcFileName = pcFileName
        cuiFileNameLen = c_int(uiFileNameLen)
        ret = self.dll.OA_WriteToFile(cpcFileName, cuiFileNameLen)
        return (ret)

    def OutAuxPort(self, port, state):
        """
            Description:
            This function sets the TTL Auxiliary Output port (P) on the Andor plug-in card to either ON/HIGH or OFF/LOW.

            Synopsis:
            ret = OutAuxPort(port, state)

            Inputs:
            port - Number of AUX out port on Andor card:
                1 - to 4
            state - state to put port in:
                0 - OFF/LOW
                all - others    ON/HIGH

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - AUX port set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid port id.

            C++ Equiv:
            unsigned int OutAuxPort(int port, int state);

            See Also:
            InAuxPort

        """
        cport = c_int(port)
        cstate = c_int(state)
        ret = self.dll.OutAuxPort(cport, cstate)
        return (ret)

    def PostProcessCountConvert(self, pInputImage, iOutputBufferSize, iNumImages, iBaseline, iMode, iEmGain, fQE, fSensitivity, iHeight, iWidth):
        """
            Description:
            This function will convert the input image data to either Photons or Electrons based on the mode selected by the user.  The input data should be in counts.

            Synopsis:
            (ret, pOutputImage) = PostProcessCountConvert(pInputImage, iOutputBufferSize, iNumImages, iBaseline, iMode, iEmGain, fQE, fSensitivity, iHeight, iWidth)

            Inputs:
            pInputImage - The input image data to be processed.:
                at32 - * pOutputImage:  The output buffer to return the processed image.
            iOutputBufferSize - The size of the output buffer.:
                data - data
            iNumImages - The number of images if a kinetic series is supplied as the input
            iBaseline - The baseline associated with the image.:
                1 - - Convert to Electrons
                2 - - Convert to Photons
            iMode - The mode to use to process the data.
            iEmGain - The gain level of the input image.
            fQE - The Quantum Efficiency of the sensor.
            fSensitivity - The Sensitivity value used to acquire the image.
            iHeight - The height of the image.
            iWidth - The width of the image.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition prepared.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Number of images less than zero.
                DRV_P5INVALID - Baseline less than zero.
                DRV_P6INVALID - Invalid count convert mode.
                DRV_P7INVALID - EMGain less than zero.
                DRV_P8INVALID DRV_P9INVALID - QE less than zero.
                DRV_P10INVALID - Sensitivity less than zero.
                DRV_P11INVALID - Height less than zero.
                DRV_ERROR_BUFFSIZE - Width less than zero.
            pOutputImage - The output buffer to return the processed image.

            C++ Equiv:
            unsigned int PostProcessCountConvert(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iNumImages, int iBaseline, int iMode, int iEmGain, float fQE, float fSensitivity, int iHeight, int iWidth);

        """
        cpInputImage = c_int(pInputImage)
        cpOutputImage = c_int()
        ciOutputBufferSize = c_int(iOutputBufferSize)
        ciNumImages = c_int(iNumImages)
        ciBaseline = c_int(iBaseline)
        ciMode = c_int(iMode)
        ciEmGain = c_int(iEmGain)
        cfQE = c_float(fQE)
        cfSensitivity = c_float(fSensitivity)
        ciHeight = c_int(iHeight)
        ciWidth = c_int(iWidth)
        ret = self.dll.PostProcessCountConvert(byref(cpInputImage), byref(
            cpOutputImage), ciOutputBufferSize, ciNumImages, ciBaseline, ciMode, ciEmGain, cfQE, cfSensitivity, ciHeight, ciWidth)
        return (ret, cpOutputImage.value)

    def PostProcessDataAveraging(self, pInputImage, iOutputBufferSize, iNumImages, iAveragingFilterMode, iHeight, iWidth, iFrameCount, iAveragingFactor):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, pOutputImage) = PostProcessDataAveraging(pInputImage, iOutputBufferSize, iNumImages, iAveragingFilterMode, iHeight, iWidth, iFrameCount, iAveragingFactor)

            Inputs:
            pInputImage -
            iOutputBufferSize -
            iNumImages -
            iAveragingFilterMode -
            iHeight -
            iWidth -
            iFrameCount -
            iAveragingFactor -

            Outputs:
            ret - Function Return Code
            pOutputImage -

            C++ Equiv:
            unsigned int PostProcessDataAveraging(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iNumImages, int iAveragingFilterMode, int iHeight, int iWidth, int iFrameCount, int iAveragingFactor);

        """
        cpInputImage = c_int(pInputImage)
        cpOutputImage = c_int()
        ciOutputBufferSize = c_int(iOutputBufferSize)
        ciNumImages = c_int(iNumImages)
        ciAveragingFilterMode = c_int(iAveragingFilterMode)
        ciHeight = c_int(iHeight)
        ciWidth = c_int(iWidth)
        ciFrameCount = c_int(iFrameCount)
        ciAveragingFactor = c_int(iAveragingFactor)
        ret = self.dll.PostProcessDataAveraging(byref(cpInputImage), byref(
            cpOutputImage), ciOutputBufferSize, ciNumImages, ciAveragingFilterMode, ciHeight, ciWidth, ciFrameCount, ciAveragingFactor)
        return (ret, cpOutputImage.value)

    def PostProcessNoiseFilter(self, pInputImage, iOutputBufferSize, iBaseline, iMode, fThreshold, iHeight, iWidth):
        """
            Description:
            This function will apply a filter to the input image and return the processed image in the output buffer.  The filter applied is chosen by the user by setting Mode to a permitted value.

            Synopsis:
            (ret, pOutputImage) = PostProcessNoiseFilter(pInputImage, iOutputBufferSize, iBaseline, iMode, fThreshold, iHeight, iWidth)

            Inputs:
            pInputImage - The input image data to be processed.:
                at32 - * pOutputImage:  The output buffer to return the processed image.
            iOutputBufferSize - The baseline associated with the image.
            iBaseline - The mode to use to process the data.:
                1 - Use Median Filter.
                2 - Use Level Above Filter.
                3 - Use interquartile Range Filter.
                4 - Use Noise Threshold Filter.
            iMode - This is the Threshold multiplier for the Median, interquartile:
                and - Noise Threshold filters.  For the Level Above filter this is
                Threshold - count above the baseline.
            fThreshold - The height of the image.
            iHeight - The width of the image.
            iWidth -

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition prepared.
                DRV_NOT_SUPPORTED DRV_NOT_INITIALIZED - Camera does not support Noise filter processing. System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Baseline less than zero.
                DRV_P5INVALID - Invalid Filter mode.
                DRV_P6INVALID - Threshold value not valid for selected mode.
                DRV_P7INVALID - Height less than zero.
                DRV_P8INVALID DRV_ERROR_BUFFSIZE - Width less than zero.
            pOutputImage - The size of the output buffer.

            C++ Equiv:
            unsigned int PostProcessNoiseFilter(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iBaseline, int iMode, float fThreshold, int iHeight, int iWidth);

        """
        cpInputImage = c_int(pInputImage)
        cpOutputImage = c_int()
        ciOutputBufferSize = c_int(iOutputBufferSize)
        ciBaseline = c_int(iBaseline)
        ciMode = c_int(iMode)
        cfThreshold = c_float(fThreshold)
        ciHeight = c_int(iHeight)
        ciWidth = c_int(iWidth)
        ret = self.dll.PostProcessNoiseFilter(byref(cpInputImage), byref(
            cpOutputImage), ciOutputBufferSize, ciBaseline, ciMode, cfThreshold, ciHeight, ciWidth)
        return (ret, cpOutputImage.value)

    def PostProcessPhotonCounting(self, pInputImage, iOutputBufferSize, iNumImages, iNumframes, iNumberOfThresholds, iHeight, iWidth):
        """
            Description:
            This function will convert the input image data to photons and return the processed image in the output buffer.

            Synopsis:
            (ret, pOutputImage, pfThreshold) = PostProcessPhotonCounting(pInputImage, iOutputBufferSize, iNumImages, iNumframes, iNumberOfThresholds, iHeight, iWidth)

            Inputs:
            pInputImage - The input image data to be processed.:
                at32 - * pOutputImage:  The output buffer to return the processed image.
            iOutputBufferSize - The number of images if a kinetic series is supplied as the input:
                data - data
            iNumImages - The number of frames per output image.
            iNumframes - The number of thresholds provided by the user.
            iNumberOfThresholds - The Thresholds used to define a photon.
            iHeight - The width of the image.
            iWidth -

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS DRV_NOT_INITIALIZED - Acquisition prepared.
                DRV_ACQUIRING - System not initialized.
                DRV_P1INVALID - Acquisition in progress.
                DRV_P2INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Invalid pointer (i.e. NULL).
                DRV_P5INVALID - Number of images less than zero.
                DRV_P6INVALID - Invalid Number of Frames requested.
                DRV_P7INVALID - Invalid number of thresholds.
                DRV_P8INVALID - Invalid pointer (i.e. NULL).
                DRV_P9INVALID - Height less than zero.
                DRV_ERROR_BUFFSIZE - Width less than zero.
            pOutputImage - The size of the output buffer.
            pfThreshold - The height of the image.

            C++ Equiv:
            unsigned int PostProcessPhotonCounting(at_32 * pInputImage, at_32 * pOutputImage, int iOutputBufferSize, int iNumImages, int iNumframes, int iNumberOfThresholds, float * pfThreshold, int iHeight, int iWidth);

        """
        cpInputImage = c_int(pInputImage)
        cpOutputImage = c_int()
        ciOutputBufferSize = c_int(iOutputBufferSize)
        ciNumImages = c_int(iNumImages)
        ciNumframes = c_int(iNumframes)
        ciNumberOfThresholds = c_int(iNumberOfThresholds)
        cpfThreshold = c_float()
        ciHeight = c_int(iHeight)
        ciWidth = c_int(iWidth)
        ret = self.dll.PostProcessPhotonCounting(byref(cpInputImage), byref(
            cpOutputImage), ciOutputBufferSize, ciNumImages, ciNumframes, ciNumberOfThresholds, byref(cpfThreshold), ciHeight, ciWidth)
        return (ret, cpOutputImage.value, cpfThreshold.value)

    def PrepareAcquisition(self):
        """
            Description:
            This function reads the current acquisition setup and allocates and configures any memory that will be used during the acquisition. The function call is not required as it will be called automatically by the StartAcquisition function if it has not already been called externally.
            However for long kinetic series acquisitions the time to allocate and configure any memory can be quite long which can result in a long delay between calling StartAcquisition and the acquisition actually commencing. For iDus, there is an additional delay caused by the camera being set-up with any new acquisition parameters. Calling PrepareAcquisition first will reduce this delay in the StartAcquisition call.

            Synopsis:
            ret = PrepareAcquisition()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition prepared.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_INIERROR - Error reading DETECTOR.INI.
                DRV_ACQERROR - Acquisition settings invalid.
                DRV_ERROR_PAGELOCK - Unable to allocate memory.
                DRV_INVALID_FILTER - Filter not available for current acquisition.
                DRV_IOCERROR - integrate On Chip setup error.
                DRV_BINNING_ERROR - Range not multiple of horizontal binning.
                DRV_SPOOLSETUPERROR - Error with spool settings.

            C++ Equiv:
            unsigned int PrepareAcquisition(void);

            See Also:
            StartAcquisition FreeInternalMemory

        """
        ret = self.dll.PrepareAcquisition()
        return (ret)

    def SaveAsBmp(self, path, palette, ymin, ymax):
        """
            Description:
            This function saves the last acquisition as a bitmap file, which can be loaded into an imaging package. The palette parameter specifies the location of a .PAL file, which describes the colors to use in the bitmap. This file consists of 256 lines of ASCII text; each line containing three numbers separated by spaces indicating the red, green and blue component of the respective color value.
            The ymin and ymax parameters indicate which data values will map to the first and last colors in the palette:
            * All data values below or equal to ymin will be colored with the first color.
            * All values above or equal to ymax will be colored with the last color
            * All other palette colors will be scaled across values between these limits.

            Synopsis:
            ret = SaveAsBmp(path, palette, ymin, ymax)

            Inputs:
            path - The filename of the bitmap.
            palette - The filename of a palette file (.PAL) for applying color to the bitmap.
            ymin - Min data value that palette will be scaled across. If ymin = 0 and ymax = 0 the palette will scale across the full range of values.
            ymax - Max data value that palette will be scaled across. If ymin = 0 and ymax = 0 the palette will scale across the full range of values.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved as bitmap.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_ERROR_PAGELOCK - File too large to be generated in memory.

            C++ Equiv:
            unsigned int SaveAsBmp(const char * path, const char * palette, long ymin, long ymax);

            See Also:
            SaveAsSif SaveAsEDF SaveAsFITS SaveAsRaw SaveAsSPC SaveAsTiff

            Note: If the last acquisition was in Kinetic Series mode, each image will be saved in a separate Bitmap file. The filename specified will have an index number appended to it, indicating the position in the series.

        """
        cpath = c_char_p(path.encode('utf-8'))
        cpalette = c_char_p(palette.encode('utf-8'))
        cymin = c_int(ymin)
        cymax = c_int(ymax)
        ret = self.dll.SaveAsBmp(cpath, cpalette, cymin, cymax)
        return (ret)

    def SaveAsCalibratedSif(self, path, data_type, unit, coeff, rayleighWave):
        """
            Description:
            This function will save the data from the last acquisition into a file. User text can be added to sif files using the SaveAsCommentedSif and SetSifComment functions.

            Synopsis:
            ret = SaveAsCalibratedSif(path, data_type, unit, coeff, rayleighWave)

            Inputs:
            path - pointer to a filename specified by the user
            data_type - the label that will be applied to the x-axis when displayed
            unit - unit to be used for x-axis when displayed
            coeff - the 4 calibration constants for the x-axis for a third order polynomial of  the form         Cal = coeff[0] + coeff[1]*P + coeff[2]*P*P + coeff[3]*P*P*P  where P is the pixel number, starting from 1.
            rayleighWave - Rayleigh wavelength required for a Raman Shift calibration

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data saved
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING  - Acquisition in progress
                DRV_ERROR_ACK  - Unable to communicate with card
                DRV_P1INVALID - Invalid filename
                DRV_P2INVALID - Invalid data type
                DRV_P3INVALID - Invalid units

            C++ Equiv:
            unsigned int SaveAsCalibratedSif(char * path, int data_type, int unit, float * coeff, float rayleighWave);

            See Also:
            SetSifComment SaveAsCommentedSif SaveAsEDF SaveAsFITS SaveAsRaw SaveAsSPC SaveAsTiff SaveAsBmp

        """
        cpath = c_char_p(path.encode('utf-8'))
        cdata_type = c_int(data_type)
        cunit = c_int(unit)
        ccoeff = c_float * 4
        crayleighWave = c_float(rayleighWave)
        ret = self.dll.SaveAsCalibratedSif(
            cpath, cdata_type, cunit, ccoeff(*coeff), crayleighWave)
        return (ret)

    def SaveAsCommentedSif(self, path, comment):
        """
            Description:
            This function will save the data from the last acquisition into a file. The comment text will be added to the user text portion of the Sif file.

            Synopsis:
            ret = SaveAsCommentedSif(path, comment)

            Inputs:
            path - pointer to a filename specified by the user.
            comment - comment text to add to the sif file

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data saved.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid filename.

            C++ Equiv:
            unsigned int SaveAsCommentedSif(char * path, char * comment);

            See Also:
            SetSifComment SaveAsSif SaveAsEDF SaveAsFITS SaveAsRaw SaveAsSPC SaveAsTiff SaveAsBmp SetSifComment

            Note: The comment used in SIF files created with this function is discarded once the call completes, i.e. future calls to SaveAsSif will not use this comment. To set a persistent comment use the SetSifComment function.

        """
        cpath = c_char_p(path.encode('utf-8'))
        ccomment = comment
        ret = self.dll.SaveAsCommentedSif(cpath, ccomment)
        return (ret)

    def SaveAsEDF(self, szPath, iMode):
        """
            Description:
            This function saves the last acquisition in the European Synchotron Radiation Facility Data Format (*.edf).

            Synopsis:
            ret = SaveAsEDF(szPath, iMode)

            Inputs:
            szPath - the filename to save to.
            iMode - option to save to multiple files.:
                0 - Save to 1 file
                1 - Save kinetic series to multiple files

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_P2INVALID - Invalid mode
                DRV_ERROR_PAGELOCK - File too large to be generated in memory.

            C++ Equiv:
            unsigned int SaveAsEDF(char * szPath, int iMode);

            See Also:
            SaveAsSif SaveAsFITS SaveAsRaw SaveAsSPC SaveAsTiff SaveAsBmp

        """
        cszPath = c_char_p(szPath.encode('utf-8'))
        ciMode = c_int(iMode)
        ret = self.dll.SaveAsEDF(cszPath, ciMode)
        return (ret)

    def SaveAsFITS(self, szFileTitle, typ):
        """
            Description:
            This function saves the last acquisition in the FITS (Flexible Image Transport System) Data Format (*.fits) endorsed by NASA.

            Synopsis:
            ret = SaveAsFITS(szFileTitle, typ)

            Inputs:
            szFileTitle - the filename to save to.
            typ - Data type:
                0 - Unsigned 16
                1 - Unsigned 32
                2 - Signed 16
                3 - Signed 32
                4 - Float

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_P2INVALID - Invalid mode
                DRV_ERROR_PAGELOCK - File too large to be generated in memory.

            C++ Equiv:
            unsigned int SaveAsFITS(char * szFileTitle, int typ);

            See Also:
            SaveAsSif SaveAsEDF SaveAsRaw SaveAsSPC SaveAsTiff SaveAsBmp

        """
        cszFileTitle = c_char_p(szFileTitle.encode('utf-8'))
        ctyp = c_int(typ)
        ret = self.dll.SaveAsFITS(cszFileTitle, ctyp)
        return (ret)

    def SaveAsRaw(self, szFileTitle, typ):
        """
            Description:
            This function saves the last acquisition as a raw data file.

            Synopsis:
            ret = SaveAsRaw(szFileTitle, typ)

            Inputs:
            szFileTitle - the filename to save to.
            typ - Data type:
                1 - Signed 16
                2 - Signed 32
                3 - Float

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_P2INVALID - Invalid mode
                DRV_ERROR_PAGELOCK - File too large to be generated in memory

            C++ Equiv:
            unsigned int SaveAsRaw(char * szFileTitle, int typ);

            See Also:
            SaveAsSif SaveAsEDF SaveAsFITS SaveAsSPC SaveAsTiff SaveAsBmp

        """
        cszFileTitle = c_char_p(szFileTitle.encode('utf-8'))
        ctyp = c_int(typ)
        ret = self.dll.SaveAsRaw(cszFileTitle, ctyp)
        return (ret)

    def SaveAsSif(self, path):
        """
            Description:
            This function will save the data from the last acquisition into a file, which can be read in by the main application.
            User text can be added to sif files using the SaveAsCommentedSif and SetSifComment functions.

            Synopsis:
            ret = SaveAsSif(path)

            Inputs:
            path - Filename specified by the user.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data saved.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid filename.
                DRV_ERROR_PAGELOCK - File too large to be generated in memory.

            C++ Equiv:
            unsigned int SaveAsSif(char * path);

            See Also:
            SaveAsEDF SaveAsFITS SaveAsRaw SaveAsSPC SaveAsTiff SaveAsBmp SetSifComment SaveAsCommentedSif

        """
        cpath = c_char_p(path.encode('utf-8'))
        ret = self.dll.SaveAsSif(cpath)
        return (ret)

    def SaveAsSPC(self, path):
        """
            Description:
            This function saves the last acquisition in the GRAMS .spc file format

            Synopsis:
            ret = SaveAsSPC(path)

            Inputs:
            path - the filename to save too.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_ERROR_PAGELOCK - File too large to be generated in memory.

            C++ Equiv:
            unsigned int SaveAsSPC(char * path);

            See Also:
            SaveAsSif SaveAsEDF SaveAsFITS SaveAsRaw SaveAsTiff SaveAsBmp

        """
        cpath = c_char_p(path.encode('utf-8'))
        ret = self.dll.SaveAsSPC(cpath)
        return (ret)

    def SaveAsTiff(self, path, palette, position, typ):
        """
            Description:
            This function saves the last acquisition as a tiff file, which can be loaded into an imaging package.
            The palette parameter specifies the location of a .PAL file, which describes the colors to use in the tiff.
            This file consists of 256 lines of ASCII text; each line containing three numbers separated by spaces indicating the red, green and blue
            component of the respective color value.
            The parameter position can be changed to export different scans in a kinetic series.
            If the acquisition is any other mode, position should be set to 1.
            The parameter typ can be set to 0, 1 or 2 which correspond to 8-bit, 16-bit and color, respectively

            Synopsis:
            ret = SaveAsTiff(path, palette, position, typ)

            Inputs:
            path - The filename of the tiff.
            palette - The filename of a palette file (.PAL) for applying color to the tiff.
            position - The number in the series, should be 1 for a single scan.
            typ - The type of tiff file to create.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved as tiff.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_P2INVALID - Invalid palette file
                DRV_P3INVALID - position out of range
                DRV_P4INVALID - type not valid
                DRV_ERROR_PAGELOCK - File too large to be generated in memory.

            C++ Equiv:
            unsigned int SaveAsTiff(char * path, char * palette, int position, int typ);

            See Also:
            SaveAsSif SaveAsEDF SaveAsFITS SaveAsRaw SaveAsSPC SaveAsBmp SaveAsTiffEx SaveAsBmp

        """
        cpath = c_char_p(path.encode('utf-8'))
        cpalette = c_char_p(palette.encode('utf-8'))
        cposition = c_int(position)
        ctyp = c_int(typ)
        ret = self.dll.SaveAsTiff(cpath, cpalette, cposition, ctyp)
        return (ret)

    def SaveAsTiffEx(self, path, palette, position, typ, mode):
        """
            Description:
            This function saves the last acquisition as a tiff file, which can be loaded into an imaging package. This is an extended version of the SaveAsTiff function. The palette parameter specifies the location of a .PAL file, which describes the colors to use in the tiff. This file consists of 256 lines of ASCII text; each line containing three numbers separated by spaces indicating the red, green and blue component of the respective color value. The parameter position can be changed to export different scans in a kinetic series. If the acquisition is any other mode, position should be set to 1. The parameter typ can be set to 0, 1 or 2 which correspond to 8-bit, 16-bit and color, respectively. The mode parameter specifies the mode of output. Data can be output scaled from the min and max count values across the entire range of values (mode 0) or can remain unchanged (mode 1).Of course if the count value is higher or lower than the output data range then even in mode 1 data will be scaled.

            Synopsis:
            ret = SaveAsTiffEx(path, palette, position, typ, mode)

            Inputs:
            path - The filename of the tiff.
            palette - The filename of a palette file (.PAL) for applying color to the tiff.
            position - The number in the series, should be 1 for a single scan.
            typ - The type of tiff file to create.
            mode - The output mode

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Data successfully saved as tiff
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Path invalid.
                DRV_P2INVALID - Invalid palette file
                DRV_P3INVALID - position out of range
                DRV_P4INVALID - type not valid
                DRV_P5INVALID - mode not valid
                DRV_ERROR_PAGELOCK - File too large to be generated in memory

            C++ Equiv:
            unsigned int SaveAsTiffEx(char * path, char * palette, int position, int typ, int mode);

            See Also:
            SaveAsSif SaveAsEDF SaveAsFITS SaveAsRaw SaveAsSPC SaveAsTiff SaveAsBmp

        """
        cpath = c_char_p(path.encode('utf-8'))
        cpalette = c_char_p(palette.encode('utf-8'))
        cposition = c_int(position)
        ctyp = c_int(typ)
        cmode = c_int(mode)
        ret = self.dll.SaveAsTiffEx(cpath, cpalette, cposition, ctyp, cmode)
        return (ret)

    def SaveEEPROMToFile(self, FileName):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SaveEEPROMToFile(FileName)

            Inputs:
            FileName -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SaveEEPROMToFile(char * FileName);

        """
        cFileName = c_char_p(FileName.encode('utf-8'))
        ret = self.dll.SaveEEPROMToFile(cFileName)
        return (ret)

    def SaveToClipBoard(self, palette):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SaveToClipBoard(palette)

            Inputs:
            palette -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SaveToClipBoard(char * palette);

        """
        cpalette = c_char_p(palette.encode('utf-8'))
        ret = self.dll.SaveToClipBoard(cpalette)
        return (ret)

    def SelectDevice(self, devNum):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SelectDevice(devNum)

            Inputs:
            devNum -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SelectDevice(int devNum);

        """
        cdevNum = c_int(devNum)
        ret = self.dll.SelectDevice(cdevNum)
        return (ret)

    def SelectSensorPort(self, port):
        """
            Description:
            This function selects which of the available sensor output ports will be used to acquire the image data. This feature is only supported when “SinglePortMode” has been selected (SetSensorPortMode).

            Synopsis:
            ret = SelectSensorPort(port)

            Inputs:
            port - the port selected. Valid values:
                0 Bottom Left  1 Bottom Right  2 Top Left  3 Top Right

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Port successfully selected
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported on this camera or “SinglePortMode” hasn’t been selected
                DRV_P1INVALID - Requested port isn’t valid

            C++ Equiv:
            unsigned int SelectSensorPort(int port);

            See Also:
            SetSensorPortMode GetCapabilities

            Note: This function selects which of the available sensor output ports will be used to acquire the image data. This feature is only supported when “SinglePortMode” has been selected (SetSensorPortMode).

        """
        cport = c_int(port)
        ret = self.dll.SelectSensorPort(cport)
        return (ret)

    def SendSoftwareTrigger(self):
        """
            Description:
            This function sends an event to the camera to take an acquisition when in Software Trigger mode. Not all cameras have this mode available to them. To check if your camera can operate in this mode check the GetCapabilities function for the Trigger Mode AC_TRIGGERMODE_CONTINUOUS. If this mode is physically possible and other settings are suitable (IsTriggerModeAvailable) and the camera is acquiring then this command will take an acquisition.

            Synopsis:
            ret = SendSoftwareTrigger()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Trigger sent
                DRV_NOT_INITIALIZED - System not initialized
                DRV_INVALID_MODE - Not in SoftwareTrigger mode
                DRV_IDLE - Not Acquiring
                DRV_ERROR_CODES - Error communicating with camera
                DRV_ERROR_ACK - Previous acquisition not complete

            C++ Equiv:
            unsigned int SendSoftwareTrigger(void);

            See Also:
            GetCapabilities IsTriggerModeAvailable SetAcquisitionMode SetReadMode SetTriggerMode

            Note: The settings of the camera must be as follows:
                ReadOut mode is full image
                RunMode is Run Till Abort
                TriggerMode is 10




        """
        ret = self.dll.SendSoftwareTrigger()
        return (ret)

    def SetAccumulationCycleTime(self, time):
        """
            Description:
            This function will set the accumulation cycle time to the nearest valid value not less than the given value. The actual cycle time used is obtained by GetAcquisitionTimingsGetAcquisitionTimings. Please refer to SECTION 5 - ACQUISITION MODES for further information.

            Synopsis:
            ret = SetAccumulationCycleTime(time)

            Inputs:
            time - the accumulation cycle time in seconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Cycle time accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Exposure time invalid.

            C++ Equiv:
            unsigned int SetAccumulationCycleTime(float time);

            See Also:
            SetNumberAccumulations GetAcquisitionTimings

        """
        ctime = c_float(time)
        ret = self.dll.SetAccumulationCycleTime(ctime)
        return (ret)

    def SetAcqStatusEvent(self, statusEvent):
        """
            Description:
            This function passes a Win32 Event handle to the driver via which the driver can inform the user software that the camera has started exposing or that the camera has finished exposing. To determine what event has actually occurred call the GetCameraEventStatus funtion. This may give the user software an opportunity to perform other actions that will not affect the readout of the current acquisition. The SetPCIMode function must be called to enable/disable the events from the driver.

            Synopsis:
            ret = SetAcqStatusEvent(statusEvent)

            Inputs:
            statusEvent - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Mode set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_SUPPORTED - Function not supported for operating system

            C++ Equiv:
            unsigned int SetAcqStatusEvent(at_32 statusEvent);

            See Also:
            GetCameraEventStatus SetPCIMode

            Note: This is only available with the CCI23 PCI card.

        """
        cstatusEvent = c_int(statusEvent)
        ret = self.dll.SetAcqStatusEvent(cstatusEvent)
        return (ret)

    def SetAcquisitionMode(self, mode):
        """
            Description:
            This function will set the acquisition mode to be used on the next StartAcquisition.

            Synopsis:
            ret = SetAcquisitionMode(mode)

            Inputs:
            mode - the acquisition mode.:
                1 - Single Scan
                2 - Accumulate
                3 - Kinetics
                4 - Fast Kinetics
                5 - Run till abort

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Acquisition Mode invalid.

            C++ Equiv:
            unsigned int SetAcquisitionMode(int mode);

            See Also:
            StartAcquisition

            Note: In Mode 5 the system uses a Run Till Abort acquisition mode. In Mode 5 only, the camera continually acquires data until the AbortAcquisitionAbortAcquisition function is called. By using the SetDriverEventSetDriverEvent function you will be notified as each acquisition is completed.

        """

        cmode = c_int(mode)
        ret = self.dll.SetAcquisitionMode(cmode)
        return (ret)

    def SetAcquisitionType(self, typ):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetAcquisitionType(typ)

            Inputs:
            typ -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetAcquisitionType(int typ);

        """
        ctyp = c_int(typ)
        ret = self.dll.SetAcquisitionType(ctyp)
        return (ret)

    def SetADChannel(self, channel):
        """
            Description:
            This function will set the AD channel to one of the possible A-Ds of the system. This AD channel will be used for all subsequent operations performed by the system.

            Synopsis:
            ret = SetADChannel(channel)

            Inputs:
            channel - the channel to be used 0 to GetNumberADChannelsGetNumberADChannels-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - AD channel set.
                DRV_P1INVALID - Index is out of range.

            C++ Equiv:
            unsigned int SetADChannel(int channel);

            See Also:
            GetNumberADChannels

        """
        cchannel = c_int(channel)
        ret = self.dll.SetADChannel(cchannel)
        return (ret)

    def SetAdvancedTriggerModeState(self, iState):
        """
            Description:
            This function will set the state for the iCam functionality that some cameras are capable of. There may be some cases where we wish to prevent the software using the new functionality and just do it the way it was previously done.

            Synopsis:
            ret = SetAdvancedTriggerModeState(iState)

            Inputs:
            iState - 0: turn off iCam:
                1 - 1 Enable iCam.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_P1INVALID - state invalid

            C++ Equiv:
            unsigned int SetAdvancedTriggerModeState(int iState);

            See Also:
            iCam

            Note: By default the advanced trigger functionality is enabled.

        """
        ciState = c_int(iState)
        ret = self.dll.SetAdvancedTriggerModeState(ciState)
        return (ret)

    def SetBackground(self, size):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, arr) = SetBackground(size)

            Inputs:
            size -

            Outputs:
            ret - Function Return Code
            arr -

            C++ Equiv:
            unsigned int SetBackground(at_32 * arr, long size);

        """
        carr = c_int()
        csize = c_int(size)
        ret = self.dll.SetBackground(byref(carr), csize)
        return (ret, carr.value)

    def SetBaselineClamp(self, state):
        """
            Description:
            This function turns on and off the baseline clamp functionality. With this feature enabled the baseline level of each scan in a kinetic series will be more consistent across the sequence.

            Synopsis:
            ret = SetBaselineClamp(state)

            Inputs:
            state - Enables/Disables Baseline clamp functionality:
                1 - Enable Baseline Clamp
                0 - Disable Baseline Clamp

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - Baseline Clamp not supported on this camera
                DRV_P1INVALID - State parameter was not zero or one.

            C++ Equiv:
            unsigned int SetBaselineClamp(int state);

        """
        cstate = c_int(state)
        ret = self.dll.SetBaselineClamp(cstate)
        return (ret)

    def SetBaselineOffset(self, offset):
        """
            Description:
            This function allows the user to move the baseline level by the amount selected. For example +100 will add approximately 100 counts to the default baseline value. The value entered should be a multiple of 100 between -1000 and +1000 inclusively.

            Synopsis:
            ret = SetBaselineOffset(offset)

            Inputs:
            offset - Amount to offset baseline by

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_AVAILABLE - Baseline Clamp not available for this camera
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - Offset out of range

            C++ Equiv:
            unsigned int SetBaselineOffset(int offset);

            Note: Only available on iXon range

        """
        coffset = c_int(offset)
        ret = self.dll.SetBaselineOffset(coffset)
        return (ret)

    def SetBitsPerPixel(self, value):
        """
            Description:
            This function will set the size in bits of the dynamic range for the current shift speed

            Synopsis:
            ret = SetBitsPerPixel(value)

            Inputs:
            value - the dynamic range in bits (Typically 16 or 18).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Bit depth set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_SUPPORTED - Variable bit depth not available for this camera
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - Bit depth out of range

            C++ Equiv:
            unsigned int SetBitsPerPixel(int value);

            See Also:
            SetHSSpeed SetADChannel GetCapabilities GetBitsPerPixel

            Note: This function will set the size in bits of the dynamic range for the current shift speed

        """
        cvalue = c_int(value)
        ret = self.dll.SetBitsPerPixel(cvalue)
        return (ret)

    def SetCameraLinkMode(self, mode):
        """
            Description:
            This function allows the user to enable or disable the Camera Link functionality for the camera. Enabling this functionality will start to stream all acquired data through the camera link interface.

            Synopsis:
            ret = SetCameraLinkMode(mode)

            Inputs:
            mode - Enables/Disables Camera Link mode:
                1 - Enable Camera Link
                0 - Disable Camera Link

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Camera Link not supported by this Camera
                DRV_P1INVALID - Mode was not zero or one.

            C++ Equiv:
            unsigned int SetCameraLinkMode(int mode);

            Note: Only available with iXon Ultra.

        """
        cmode = c_int(mode)
        ret = self.dll.SetCameraLinkMode(cmode)
        return (ret)

    def SetCameraStatusEnable(self, Enable):
        """
            Description:
            Use this function to Mask out certain types of acquisition status events. The default is to notify on every type of event but this may cause missed events if different types of event occur very close together. The bits in the mask correspond to the following event types:
            Use0 - Fire pulse down event
            Use1 - Fire pulse up event
            Set the corresponding bit to 0 to disable the event type and 1 to enable the event type.

            Synopsis:
            ret = SetCameraStatusEnable(Enable)

            Inputs:
            Enable - bitmask with bits set for those events about which you wish to be notified.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Mask Set.
                DRV_VXDNOTINSTALLED - Device Driver not installed.

            C++ Equiv:
            unsigned int SetCameraStatusEnable(DWORD Enable);

            See Also:
            SetAcqStatusEvent SetPCIMode

            Note: Only available with PCI systems using the CCI-23 controller card.

                Fire pulse up event not available on USB systems.

        """
        cEnable = (Enable)
        ret = self.dll.SetCameraStatusEnable(cEnable)
        return (ret)

    def SetChargeShifting(self, NumberRows, NumberRepeats):
        """
            Description:
            Use this function in External Charge Shifting trigger mode to configure how many rows to shift and how many times for each frame of data.  The number of repeats must be a multiple of 2.

            Synopsis:
            ret = SetChargeShifting(NumberRows, NumberRepeats)

            Inputs:
            NumberRows - number of rows to shift after each external trigger
            NumberRepeats - number of times to shift rows

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Success
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Trigger mode not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number of rows invalid.
                DRV_P2INVALID - Number of repeats invalid.

            C++ Equiv:
            unsigned int SetChargeShifting(unsigned int NumberRows, unsigned int NumberRepeats);

            See Also:
            SetTriggerMode GetCapabilities

            Note: Only available with certain iKon-M systems.

        """
        cNumberRows = c_uint(NumberRows)
        cNumberRepeats = c_uint(NumberRepeats)
        ret = self.dll.SetChargeShifting(cNumberRows, cNumberRepeats)
        return (ret)

    def SetComplexImage(self, numAreas):
        """
            Description:
            This is a function that allows the setting up of random tracks with more options that the SetRandomTracks function.
            The minimum number of tracks is 1. The maximum number of tracks is the number of vertical pixels.
            There is a further limit to the number of tracks that can be set due to memory constraints in the camera. It is not a fixed number but depends upon the combinations of the tracks. For example, 20 tracks of different heights will take up more memory than 20 tracks of the same height.
            If attempting to set a series of random tracks and the return code equals DRV_RANDOM_TRACK_ERROR, change the makeup of the tracks to have more repeating heights and gaps so less memory is needed.
            Each track must be defined by a group of six integers.
            -The top and bottom positions of the tracks.
            -The left and right positions for the area of interest within each track
            -The horizontal and vertical binning for each track.
            The positions of the tracks are validated to ensure that the tracks are in increasing order.
            The left and right positions for each track must be the same.
            For iXon the range is between 8 and CCD width, inclusive
            For idus the range must be between 257 and CCD width, inclusive.
            Horizontal binning must be an integer between 1 and 64 inclusive, for iXon.
            Horizontal binning is not implementated for iDus and must be set to 1.
            Vertical binning is used in the following way. A track of:
            1 10 1 1024 1 2
            is actually implemented as 5 tracks of height 2. . Note that a vertical binning of 1 will have the effect of vertically binning the entire track; otherwise vertical binning will operate as normal.
            1 2 1 1024 1 1
            3 4 1 1024 1 1
            5 6 1 1024 1 1
            7 8 1 1024 1 1
            9 10 1 1024 1 1

            Synopsis:
            (ret, areas) = SetComplexImage(numAreas)

            Inputs:
            numAreas - int * areas:

            Outputs:
            ret - Function Return Code:
                Unsigned int - DRV_RANDOM_TRACK_ERROR
                DRV_SUCCESS - Success
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number of tracks invalid.
                DRV_P2INVALID - Track positions invalid.
                DRV_ERROR_FILELOAD - Serious internal error
            areas -

            C++ Equiv:
            unsigned int SetComplexImage(int numAreas, int * areas);

            See Also:
            SetRandomTracks

            Note: Only available with iXon+ and USB cameras.

        """
        cnumAreas = c_int(numAreas)
        careas = c_int()
        ret = self.dll.SetComplexImage(cnumAreas, byref(careas))
        return (ret, careas.value)

    def SetCoolerMode(self, mode):
        """
            Description:
            This function determines whether the cooler is switched off when the camera is shut down.

            Synopsis:
            ret = SetCoolerMode(mode)

            Inputs:
            mode - :
                0 - Returns to ambient temperature on ShutDown
                1 - Temperature is maintained on ShutDown

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - State parameter was not zero or one.
                DRV_NOT_SUPPORTED - Camera does not support

            C++ Equiv:
            unsigned int SetCoolerMode(int mode);

            Note: Mode 0 not available on Luca R cameras always cooled to -20C.

        """
        cmode = c_int(mode)
        ret = self.dll.SetCoolerMode(cmode)
        return (ret)

    def SetCountConvertMode(self, Mode):
        """
            Description:
            This function configures the Count Convert mode.

            Synopsis:
            ret = SetCountConvertMode(Mode)

            Inputs:
            Mode - :
                0 - Data in Counts
                1 - Data in Electrons
                2 - Data in Photons

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Count Convert mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - Count Convert not available for this camera
                DRV_NOT_AVAILABLE - Count Convert mode not available with current settings
                DRV_P1INVALID - Mode parameter was out of range.

            C++ Equiv:
            unsigned int SetCountConvertMode(int Mode);

            See Also:
            GetCapabilities SetCountConvertWavelength

            Note: Only available on Clara, iXon 3 and iXon Ultra.

                Modes 1 and 2 are only available when:
                * Baseline Clamp active
                * Isolated crop mode off
                * EM gain must be greater than or equal to 10 and the lowest pre-amp not be selected
                * For Clara systems the extended infra red mode can not be used


        """
        cMode = c_int(Mode)
        ret = self.dll.SetCountConvertMode(cMode)
        return (ret)

    def SetCountConvertWavelength(self, wavelength):
        """
            Description:
            This function configures the wavelength used in Count Convert mode.

            Synopsis:
            ret = SetCountConvertWavelength(wavelength)

            Inputs:
            wavelength - wavelength used to determine QE

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Count Convert wavelength set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_AVAILABLE - Count Convert not available for this camera
                DRV_P1INVALID - Wavelength value was out of range.

            C++ Equiv:
            unsigned int SetCountConvertWavelength(float wavelength);

            See Also:
            GetCapabilities SetCountConvertMode

        """
        cwavelength = c_float(wavelength)
        ret = self.dll.SetCountConvertWavelength(cwavelength)
        return (ret)

    def SetCropMode(self, active, cropHeight, reserved):
        """
            Description:
            This function effectively reduces the height of the CCD by excluding some rows to achieve higher frame rates. This is currently only available on Newton cameras when the selected read mode is Full Vertical Binning. The cropHeight is the number of active rows measured from the bottom of the CCD.
            Note: it is important to ensure that no light falls on the excluded region otherwise the acquired data will be corrupted.

            Synopsis:
            ret = SetCropMode(active, cropHeight, reserved)

            Inputs:
            active - Crop mode active:
                0 - Crop mode is OFF
                1 - Crop mode if ON
            cropHeight - The selected crop height. This value must be between 1 and the CCD:
                height - int reserved: This value should be set to 0.
            reserved - This value should be set to 0

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIAILIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Active parameter is not zero or one.
                DRV_P2INVALID - Cropheight parameter is less than one or greater than the CCD height.
                DRV_P3INVALID - Reserved parameter is not equal to zero.
                DRV_NOT_SUPPORTED - Either the camera is not a Newton or the read mode is not Full Vertical Binning.

            C++ Equiv:
            unsigned int SetCropMode(int active, int cropHeight, int reserved);

            See Also:
            GetDetector SetIsolatedCropMode

            Note: Available on Newton

        """
        cactive = c_int(active)
        ccropHeight = c_int(cropHeight)
        creserved = c_int(reserved)
        ret = self.dll.SetCropMode(cactive, ccropHeight, creserved)
        return (ret)

    def SetCurrentCamera(self, cameraHandle):
        """
            Description:
            When multiple Andor cameras are installed this function allows the user to select which camera is currently active.  Once a camera has been selected the other functions can be called as normal but they will only apply to the selected camera.  If only 1 camera is installed calling this function is not required since that camera will be selected by default.

            Synopsis:
            ret = SetCurrentCamera(cameraHandle)

            Inputs:
            cameraHandle - Selects the active camera

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Camera successfully selected.
                DRV_P1INVALID - Invalid camera handle.

            C++ Equiv:
            unsigned int SetCurrentCamera(long cameraHandle);

            See Also:
            GetCurrentCamera GetAvailableCameras GetCameraHandle

        """
        ccameraHandle = c_int(cameraHandle)
        ret = self.dll.SetCurrentCamera(ccameraHandle)
        return (ret)

    def SetCustomTrackHBin(self, bin):
        """
            Description:
            This function sets the horizontal binning value to be used when the readout mode is set to Random Track.

            Synopsis:
            ret = SetCustomTrackHBin(bin)

            Inputs:
            bin - Binning size.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Binning set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid binning size.

            C++ Equiv:
            unsigned int SetCustomTrackHBin(int bin);

            See Also:
            SetReadMode

            Note: For iDus, it is recommended that you set horizontal binning to 1

        """
        cbin = c_int(bin)
        ret = self.dll.SetCustomTrackHBin(cbin)
        return (ret)

    def SetDACOutput(self, iOption, iResolution, iValue):
        """
            Description:
            Clara offers 2 configurable precision 16-bit DAC outputs.  This function should be used to set the required voltage.

            Synopsis:
            ret = SetDACOutput(iOption, iResolution, iValue)

            Inputs:
            iOption - DAC Output  DAC Pin 1 or 2 (1/2).
            iResolution - resolution of DAC can be set from 2 to 16-bit in steps of 2
            iValue - requested DAC value (for particular resolution)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - DAC Scale option accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_AVAILABLE - Feature not available.
                DRV_P1INVALID - DAC range value invalid.
                DRV_P2INVALID - Resolution unavailable.
                DRV_P3INVALID - Requested value not within DAC range.

            C++ Equiv:
            unsigned int SetDACOutput(int iOption, int iResolution, int iValue);

            See Also:
            SetDACOutputScale

            Note: Only available on Andor Clara

        """
        ciOption = c_int(iOption)
        ciResolution = c_int(iResolution)
        ciValue = c_int(iValue)
        ret = self.dll.SetDACOutput(ciOption, ciResolution, ciValue)
        return (ret)

    def SetDACOutputScale(self, iScale):
        """
            Description:
            Clara offers 2 configurable precision 16-bit DAC outputs.  This function should be used to select the active one.

            Synopsis:
            ret = SetDACOutputScale(iScale)

            Inputs:
            iScale - 5 or 10 volt DAC range (1/2).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - DAC Scale option accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_AVAILABLE - Feature not available
                DRV_P1INVALID - DAC Scale value invalid.

            C++ Equiv:
            unsigned int SetDACOutputScale(int iScale);

            See Also:
            SetDACOutput

            Note: Only available on Andor Clara

        """
        ciScale = c_int(iScale)
        ret = self.dll.SetDACOutputScale(ciScale)
        return (ret)

    def SetDataType(self, typ):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetDataType(typ)

            Inputs:
            typ -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDataType(int typ);

        """
        ctyp = c_int(typ)
        ret = self.dll.SetDataType(ctyp)
        return (ret)

    def SetDDGAddress(self, t0, t1, t2, t3, address):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetDDGAddress(t0, t1, t2, t3, address)

            Inputs:
            t0 -
            t1 -
            t2 -
            t3 -
            address -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGAddress(BYTE t0, BYTE t1, BYTE t2, BYTE t3, BYTE address);

        """
        ct0 = c_ubyte(t0)
        ct1 = c_ubyte(t1)
        ct2 = c_ubyte(t2)
        ct3 = c_ubyte(t3)
        caddress = c_ubyte(address)
        ret = self.dll.SetDDGAddress(ct0, ct1, ct2, ct3, caddress)
        return (ret)

    def SetDDGExternalOutputEnabled(self, uiIndex, uiEnabled):
        """
            Description:
            This function sets the state of a selected external output.

            Synopsis:
            ret = SetDDGExternalOutputEnabled(uiIndex, uiEnabled)

            Inputs:
            uiIndex - index of external output.
            uiEnabled - state of external output (0 - Off,1 - On).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid external output index.

            C++ Equiv:
            unsigned int SetDDGExternalOutputEnabled(at_u32 uiIndex, at_u32 uiEnabled);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled

            Note: Available on USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cuiEnabled = c_uint(uiEnabled)
        ret = self.dll.SetDDGExternalOutputEnabled(cuiIndex, cuiEnabled)
        return (ret)

    def SetDDGExternalOutputPolarity(self, uiIndex, uiPolarity):
        """
            Description:
            This function sets the polarity of a selected external output.

            Synopsis:
            ret = SetDDGExternalOutputPolarity(uiIndex, uiPolarity)

            Inputs:
            uiIndex - index of external output.
            uiPolarity - polarity of external output (0 - Positive,1 - Negative).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Polarity set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid external output index.

            C++ Equiv:
            unsigned int SetDDGExternalOutputPolarity(at_u32 uiIndex, at_u32 uiPolarity);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled GetDDGExternalOutputPolarity

            Note: Available on USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cuiPolarity = c_uint(uiPolarity)
        ret = self.dll.SetDDGExternalOutputPolarity(cuiIndex, cuiPolarity)
        return (ret)

    def SetDDGExternalOutputStepEnabled(self, uiIndex, uiEnabled):
        """
            Description:
            Each external output has the option to track the gate step applied to the gater.  This function can be used to set the state of this option.

            Synopsis:
            ret = SetDDGExternalOutputStepEnabled(uiIndex, uiEnabled)

            Inputs:
            uiIndex - index of external output.
            uiEnabled - state of external output track step (0 - Off,1 - On).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid external output index.

            C++ Equiv:
            unsigned int SetDDGExternalOutputStepEnabled(at_u32 uiIndex, at_u32 uiEnabled);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled GetDDGExternalOutputStepEnabled

            Note: Available on USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cuiEnabled = c_uint(uiEnabled)
        ret = self.dll.SetDDGExternalOutputStepEnabled(cuiIndex, cuiEnabled)
        return (ret)

    def SetDDGExternalOutputTime(self, uiIndex, uiDelay, uiWidth):
        """
            Description:
            This function can be used to set the timings for a particular external output.

            Synopsis:
            ret = SetDDGExternalOutputTime(uiIndex, uiDelay, uiWidth)

            Inputs:
            uiIndex - index of external output.
            uiDelay - external output delay time in picoseconds.
            uiWidth - external output width time in picoseconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - External outputs not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with card.
                DRV_P2INVALID - Invalid external output index.
                DRV_P3INVALID - Invalid delay.

            C++ Equiv:
            unsigned int SetDDGExternalOutputTime(at_u32 uiIndex, at_u64 uiDelay, at_u64 uiWidth);

            See Also:
            GetCapabilities GetDDGExternalOutputEnabled GetDDGExternalOutputTime

            Note: Available in USB iStar.

        """
        cuiIndex = c_uint(uiIndex)
        cuiDelay = c_ulonglong(uiDelay)
        cuiWidth = c_ulonglong(uiWidth)
        ret = self.dll.SetDDGExternalOutputTime(cuiIndex, cuiDelay, cuiWidth)
        return (ret)

    def SetDDGGain(self, gain):
        """
            Description:
            Deprecated for SetMCPGain.

            Synopsis:
            ret = SetDDGGain(gain)

            Inputs:
            gain -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGGain(int gain); // deprecated

        """
        cgain = c_int(gain)
        ret = self.dll.SetDDGGain(cgain)
        return (ret)

    def SetDDGGateStep(self, step):
        """
            Description:
            This function will set a constant value for the gate step in a kinetic series. The lowest available resolution is 25 picoseconds and the maximum permitted value is 25 seconds.

            Synopsis:
            ret = SetDDGGateStep(step)

            Inputs:
            step - gate step in picoseconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Gate step invalid.

            C++ Equiv:
            unsigned int SetDDGGateStep(double step);

            See Also:
            SetDDGTimes SetDDGVariableGateStep

            Note: Available on iStar.

        """
        cstep = c_double(step)
        ret = self.dll.SetDDGGateStep(cstep)
        return (ret)

    def SetDDGGateTime(self, uiDelay, uiWidth):
        """
            Description:
            This function can be used to set the gate timings for a USB iStar.

            Synopsis:
            ret = SetDDGGateTime(uiDelay, uiWidth)

            Inputs:
            uiDelay - gate delay time in picoseconds.
            uiWidth - gate width time in picoseconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - USB iStar not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.
                DRV_P2INVALID - Invalid delay.

            C++ Equiv:
            unsigned int SetDDGGateTime(at_u64 uiDelay, at_u64 uiWidth);

            See Also:
            GetCapabilities GetDDGGateTime

        """
        cuiDelay = c_ulonglong(uiDelay)
        cuiWidth = c_ulonglong(uiWidth)
        ret = self.dll.SetDDGGateTime(cuiDelay, cuiWidth)
        return (ret)

    def SetDDGInsertionDelay(self, state):
        """
            Description:
            This function controls the length of the insertion delay.

            Synopsis:
            ret = SetDDGInsertionDelay(state)

            Inputs:
            state - NORMAL/FAST switch for insertion delay.:
                0 - to set normal insertion delay.
                1 - to set fast insertion delay.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for delay accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with system.

            C++ Equiv:
            unsigned int SetDDGInsertionDelay(int state);

            See Also:
            GetCapabilities SetDDGIntelligate

        """
        cstate = c_int(state)
        ret = self.dll.SetDDGInsertionDelay(cstate)
        return (ret)

    def SetDDGIntelligate(self, state):
        """
            Description:
            This function controls the MCP gating. Not available when the fast insertion delay option is selected.

            Synopsis:
            ret = SetDDGIntelligate(state)

            Inputs:
            state - ON/OFF switch for the MCP gating.:
                0 - to switch MCP gating OFF.
                1 - to switch MCP gating ON.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - intelligate option accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with system.

            C++ Equiv:
            unsigned int SetDDGIntelligate(int state);

            See Also:
            GetCapabilities SetDDGInsertionDelay

        """
        cstate = c_int(state)
        ret = self.dll.SetDDGIntelligate(cstate)
        return (ret)

    def SetDDGIOC(self, state):
        """
            Description:
            This function activates the integrate on chip (IOC) option.

            Synopsis:
            ret = SetDDGIOC(state)

            Inputs:
            state - ON/OFF switch for the IOC option.:
                0 - to switch IOC OFF.
                1 - to switch IOC ON.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IOC option accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ERROR_ACK - Unable to communicate with system.

            C++ Equiv:
            unsigned int SetDDGIOC(int state);

            See Also:
            GetCapabilities SetDDGIOCFrequency GetDDGIOCFrequency SetDDGIOCNumber GetDDGIOCNumber GetDDGIOCPulses

        """
        cstate = c_int(state)
        ret = self.dll.SetDDGIOC(cstate)
        return (ret)

    def SetDDGIOCFrequency(self, frequency):
        """
            Description:
            This function sets the frequency of the integrate on chip option. It should be called once the conditions of the experiment have been setup in order for correct operation. The frequency should be limited to 5000Hz when intelligate is activated to prevent damage to the head and 50000Hz otherwise to prevent the gater from overheating. The recommended order is
            ...
            Experiment setup (exposure time, readout mode, gate parameters, ...)
            ...
            SetDDGIOCFrequency (x)
            SetDDGIOCSetDDGIOC(true)
            GetDDGIOCPulses(y)
            StartAcquisitionStartAcquisition()

            Synopsis:
            ret = SetDDGIOCFrequency(frequency)

            Inputs:
            frequency - frequency of IOC option in Hz.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for frequency accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ERROR_ACK - Unable to communicate with card.

            C++ Equiv:
            unsigned int SetDDGIOCFrequency(double frequency);

            See Also:
            GetDDGIOCFrequency SetDDGIOCNumber GetDDGIOCNumber GetDDGIOCPulses SetDDGIOC

        """
        cfrequency = c_double(frequency)
        ret = self.dll.SetDDGIOCFrequency(cfrequency)
        return (ret)

    def SetDDGIOCNumber(self, numberPulses):
        """
            Description:
            This function allows the user to limit the number of pulses used in the integrate on chip option at a given frequency. It should be called once the conditions of the experiment have been setup in order for correct operation.

            Synopsis:
            ret = SetDDGIOCNumber(numberPulses)

            Inputs:
            numberPulses - the number of integrate on chip pulses triggered within the fire pulse.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for IOC number accepted
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - IOC not supported
                DRV_ERROR_ACK - Unable to communicate with card

            C++ Equiv:
            unsigned int SetDDGIOCNumber(long numberPulses);

            See Also:
            SetDDGIOCFrequency GetDDGIOCFrequency GetDDGIOCNumber GetDDGIOCPulses SetDDGIOC

        """
        cnumberPulses = c_int(numberPulses)
        ret = self.dll.SetDDGIOCNumber(cnumberPulses)
        return (ret)

    def SetDDGIOCPeriod(self, period):
        """
            Description:
            This function can be used to set the IOC period that will be triggered. It should only be called once all the conditions of the experiment have been defined.

            Synopsis:
            ret = SetDDGIOCPeriod(period)

            Inputs:
            period - the period of integrate on chip pulses triggered within the fire pulse.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IOC period set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - Invalid period.

            C++ Equiv:
            unsigned int SetDDGIOCPeriod(at_u64 period);

            See Also:
            GetCapabilities SetDDGIOC SetDDGIOCFrequency GetDDGIOCPeriod

        """
        cperiod = c_ulonglong(period)
        ret = self.dll.SetDDGIOCPeriod(cperiod)
        return (ret)

    def SetDDGIOCTrigger(self, trigger):
        """
            Description:
            This function can be used to select whether to trigger the IOC pulse train with either the rising edge of the fire pulse or an externally supplied trigger.

            Synopsis:
            ret = SetDDGIOCTrigger(trigger)

            Inputs:
            trigger - IOC Trigger Option:
                0 - Fire pulse
                1 - External Trigger

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IOC trigger set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - IOC not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - Invalid trigger.

            C++ Equiv:
            unsigned int SetDDGIOCTrigger(at_u32 trigger);

            See Also:
            GetCapabilities GetDDGIOCTrigger SetDDGIOC SetTriggerMode

        """
        ctrigger = c_uint(trigger)
        ret = self.dll.SetDDGIOCTrigger(ctrigger)
        return (ret)

    def SetDDGLiteControlByte(self, channel, control):
        """
            Description:
            THIS FUNCTION IS RESERVED

            Synopsis:
            ret = SetDDGLiteControlByte(channel, control)

            Inputs:
            channel -
            control -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGLiteControlByte(AT_DDGLiteChannelId channel, char control);

        """
        cchannel = (channel)
        ccontrol = c_char(control)
        ret = self.dll.SetDDGLiteControlByte(cchannel, ccontrol)
        return (ret)

    def SetDDGLiteGlobalControlByte(self, control):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetDDGLiteGlobalControlByte(control)

            Inputs:
            control -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGLiteGlobalControlByte(char control);

        """
        ccontrol = c_char(control)
        ret = self.dll.SetDDGLiteGlobalControlByte(ccontrol)
        return (ret)

    def SetDDGLiteInitialDelay(self, channel, fDelay):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetDDGLiteInitialDelay(channel, fDelay)

            Inputs:
            channel -
            fDelay -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGLiteInitialDelay(AT_DDGLiteChannelId channel, float fDelay);

        """
        cchannel = (channel)
        cfDelay = c_float(fDelay)
        ret = self.dll.SetDDGLiteInitialDelay(cchannel, cfDelay)
        return (ret)

    def SetDDGLiteInterPulseDelay(self, channel, fDelay):
        """
            Description:


            Synopsis:
            ret = SetDDGLiteInterPulseDelay(channel, fDelay)

            Inputs:
            channel -
            fDelay -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGLiteInterPulseDelay(AT_DDGLiteChannelId channel, float fDelay);

        """
        cchannel = (channel)
        cfDelay = c_float(fDelay)
        ret = self.dll.SetDDGLiteInterPulseDelay(cchannel, cfDelay)
        return (ret)

    def SetDDGLitePulsesPerExposure(self, channel, ui32Pulses):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetDDGLitePulsesPerExposure(channel, ui32Pulses)

            Inputs:
            channel -
            ui32Pulses -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGLitePulsesPerExposure(AT_DDGLiteChannelId channel, at_u32 ui32Pulses);

        """
        cchannel = (channel)
        cui32Pulses = c_uint(ui32Pulses)
        ret = self.dll.SetDDGLitePulsesPerExposure(cchannel, cui32Pulses)
        return (ret)

    def SetDDGLitePulseWidth(self, channel, fWidth):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetDDGLitePulseWidth(channel, fWidth)

            Inputs:
            channel -
            fWidth -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetDDGLitePulseWidth(AT_DDGLiteChannelId channel, float fWidth);

        """
        cchannel = (channel)
        cfWidth = c_float(fWidth)
        ret = self.dll.SetDDGLitePulseWidth(cchannel, cfWidth)
        return (ret)

    def SetDDGOpticalWidthEnabled(self, uiEnabled):
        """
            Description:
            This function can be used to configure a system to use optical gate width.

            Synopsis:
            ret = SetDDGOpticalWidthEnabled(uiEnabled)

            Inputs:
            uiEnabled - optical gate width option (0 - Off, 1 - On).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - State set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Optical gate width not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.

            C++ Equiv:
            unsigned int SetDDGOpticalWidthEnabled(at_u32 uiEnabled);

            See Also:
            GetCapabilities GetDDGTTLGateWidth GetDDGOpticalWidthEnabled

        """
        cuiEnabled = c_uint(uiEnabled)
        ret = self.dll.SetDDGOpticalWidthEnabled(cuiEnabled)
        return (ret)

    def SetDDGStepCoefficients(self, mode, p1, p2):
        """
            Description:
            This function will configure the coefficients used in a kinetic series with gate step active. The lowest available resolution is 25 picoseconds and the maximum permitted value is 25 seconds for a PCI iStar.
            The lowest available resolution is 10 picoseconds and the maximum permitted value is 10 seconds for a USB iStar.

            Synopsis:
            ret = SetDDGStepCoefficients(mode, p1, p2)

            Inputs:
            mode - the gate step mode.:
                0 - constant  (p1*(n-1)).
                1 - exponential (p1*exp(p2*n)).
                2 - logarithmic (p1*log(p2*n)).
                3 - linear (p1 + p2*n).
                n - = 1, 2, ..., number in kinetic series
            p1 -
            p2 -

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode coefficients set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - Gate step mode invalid.

            C++ Equiv:
            unsigned int SetDDGStepCoefficients(at_u32 mode, double p1, double p2);

            See Also:
            StartAcquisition SetDDGStepMode GetDDGStepMode GetDDGStepCoefficients

            Note: Available on iStar and USB iStar.

        """
        cmode = c_uint(mode)
        cp1 = c_double(p1)
        cp2 = c_double(p2)
        ret = self.dll.SetDDGStepCoefficients(cmode, cp1, cp2)
        return (ret)

    def SetDDGStepMode(self, mode):
        """
            Description:
            This function will set the current gate step mode.

            Synopsis:
            ret = SetDDGStepMode(mode)

            Inputs:
            mode - the gate step mode.:
                0 - constant.
                1 - exponential.
                2 - logarithmic.
                3 - linear.
                100 - off.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Gate step not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with system.
                DRV_P1INVALID - Invalid gate step mode.

            C++ Equiv:
            unsigned int SetDDGStepMode(at_u32 mode);

            See Also:
            StartAcquisition GetDDGStepMode SetDDGStepCoefficients GetDDGStepCoefficients

        """
        cmode = c_uint(mode)
        ret = self.dll.SetDDGStepMode(cmode)
        return (ret)

    def SetDDGTimes(self, t0, t1, t2):
        """
            Description:
            This function sets the properties of the gate pulse. t0 has a resolution of 16 nanoseconds whilst t1 and t2 have a resolution of 25 picoseconds.

            Synopsis:
            ret = SetDDGTimes(t0, t1, t2)

            Inputs:
            t0 - output A delay in nanoseconds.
            t1 - gate delay in picoseconds.
            t2 - pulse width in picoseconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Values for gate pulse accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Invalid output A delay.
                DRV_P2INVALID - Invalid gate delay.
                DRV_P3INVALID - Invalid pulse width.

            C++ Equiv:
            unsigned int SetDDGTimes(double t0, double t1, double t2);

            See Also:
            SetDDGGateStep

            Note: Available on iStar.

        """
        ct0 = c_double(t0)
        ct1 = c_double(t1)
        ct2 = c_double(t2)
        ret = self.dll.SetDDGTimes(ct0, ct1, ct2)
        return (ret)

    def SetDDGTriggerMode(self, mode):
        """
            Description:
            This function will set the trigger mode of the internal delay generator to either internal or External

            Synopsis:
            ret = SetDDGTriggerMode(mode)

            Inputs:
            mode - trigger mode:
                0 - internal
                1 - External

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Trigger mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Trigger mode invalid.

            C++ Equiv:
            unsigned int SetDDGTriggerMode(int mode);

            Note: Available on iStar.

        """
        cmode = c_int(mode)
        ret = self.dll.SetDDGTriggerMode(cmode)
        return (ret)

    def SetDDGVariableGateStep(self, mode, p1, p2):
        """
            Description:
            This function will set a varying value for the gate step in a kinetic series. The lowest available resolution is 25 picoseconds and the maximum permitted value is 25 seconds.

            Synopsis:
            ret = SetDDGVariableGateStep(mode, p1, p2)

            Inputs:
            mode - the gate step mode.:
                1 - Exponential (p1*exp(p2*n))
                2 - Logarithmic (p1*log(p2*n))
                3 - Linear (p1 + p2*n)
                n - = 1, 2, ..., number in kinetic series
            p1 -
            p2 -

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Gate step mode invalid.

            C++ Equiv:
            unsigned int SetDDGVariableGateStep(int mode, double p1, double p2);

            See Also:
            StartAcquisition

            Note: Available on iStar.

        """
        cmode = c_int(mode)
        cp1 = c_double(p1)
        cp2 = c_double(p2)
        ret = self.dll.SetDDGVariableGateStep(cmode, cp1, cp2)
        return (ret)

    def SetDDGWidthStepCoefficients(self, mode, p1, p2):
        """
            Description:
            This function will configure the coefficients used in a kinetic series with gate width step active. The lowest available resolution is 25 picoseconds and the maximum permitted value is 25 seconds for a PCI iStar.
            The lowest available resolution is 10 picoseconds and the maximum permitted value is 10 seconds for a USB iStar.

            Synopsis:
            ret = SetDDGWidthStepCoefficients(mode, p1, p2)

            Inputs:
            mode - the gate step mode.:
                0 - constant  (p1*(n-1)).
                1 - exponential (p1*exp(p2*n)).
                2 - logarithmic (p1*log(p2*n)).
                3 - linear (p1 + p2*n).
                - n = 1, 2, ..., number in kinetic series
            p1 - The first coefficient
            p2 - The second coefficient

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode coefficients set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Gate step mode invalid.

            C++ Equiv:
            unsigned int SetDDGWidthStepCoefficients(at_u32 mode, double p1, double p2);

            See Also:
            SetDDGWidthStepMode GetDDGWidthStepMode GetDDGWidthStepCoefficients

        """
        cmode = c_uint(mode)
        cp1 = c_double(p1)
        cp2 = c_double(p2)
        ret = self.dll.SetDDGWidthStepCoefficients(cmode, cp1, cp2)
        return (ret)

    def SetDDGWidthStepMode(self, mode):
        """
            Description:
            This function will set the current gate width step mode.

            Synopsis:
            ret = SetDDGWidthStepMode(mode)

            Inputs:
            mode - the gate step mode.:
                0 - constant.
                1 - exponential.
                2 - logarithmic.
                3 - linear.
                100 - off.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gate step mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Gate step not supported.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid gate step mode.

            C++ Equiv:
            unsigned int SetDDGWidthStepMode(at_u32 mode);

            See Also:
            SetDDGWidthStepCoefficients GetDDGWidthStepMode GetDDGWidthStepCoefficients

        """
        cmode = c_uint(mode)
        ret = self.dll.SetDDGWidthStepMode(cmode)
        return (ret)

    def SetDelayGenerator(self, board, address, typ):
        """
            Description:
            This function sets parameters to control the delay generator through the GPIB card in your computer.

            Synopsis:
            ret = SetDelayGenerator(board, address, typ)

            Inputs:
            board - The GPIB board number of the card used to interface with the Delay Generator.:
                short - address: The number that allows the GPIB board to identify and send commands to the delay generator.
            address - The number that allows the GPIB board to identify and send commands to the delay generator.

            typ - The type of your Delay Generator.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Delay Generator set up.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - GPIB board invalid.
                DRV_P2INVALID - GPIB address invalid
                DRV_P3INVALID - Delay generator type invalid.

            C++ Equiv:
            unsigned int SetDelayGenerator(int board, short address, int typ);

            See Also:
            SetGate

            Note: Available on ICCD.

        """
        cboard = c_int(board)
        caddress = c_short(address)
        ctyp = c_int(typ)
        ret = self.dll.SetDelayGenerator(cboard, caddress, ctyp)
        return (ret)

    def SetDMAParameters(self, MaxImagesPerDMA, SecondsPerDMA):
        """
            Description:
            In order to facilitate high image readout rates the controller card may wait for multiple images to be acquired before notifying the SDK that new data is available. Without this facility, there is a chance that hardware interrupts may be lost as the operating system does not have enough time to respond to each interrupt. The drawback to this is that you will not get the data for an image until all images for that interrupt have been acquired.
            There are 3 settings involved in determining how many images will be acquired for each notification (DMA interrupt) of the controller card and they are as follows:
            1. The size of the DMA buffer gives an upper limit on the number of images that can be stored within it and is usually set to the size of one full image when installing the software. This will usually mean that if you acquire full frames there will never be more than one image per DMA.
            2. A second setting that is used is the minimum amount of time (SecondsPerDMA) that should expire between interrupts. This can be used to give an indication of the reponsiveness of the operating system to interrupts. Decreasing this value will allow more interrupts per second and should only be done for faster pcs. The default value is 0.03s (30ms), finding the optimal value for your pc can only be done through experimentation.
            3. The third setting is an overide to the number of images calculated using the previous settings. If the number of images per dma is calculated to be greater than MaxImagesPerDMA then it will be reduced to MaxImagesPerDMA. This can be used to, for example, ensure that there is never more than 1 image per DMA by setting MaxImagesPerDMA to 1. Setting MaxImagesPerDMA to zero removes this limit. Care should be taken when modifying these parameters as missed interrupts may prevent the acquisition from completing.

            Synopsis:
            ret = SetDMAParameters(MaxImagesPerDMA, SecondsPerDMA)

            Inputs:
            MaxImagesPerDMA - Override to the number of images per DMA if the calculated value is higher than this. (Default=0, ie. no override)
            SecondsPerDMA - Minimum amount of time to elapse between interrrupts. (Default=0.03s)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - DMA Parameters setup successfully.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - MaxImagesPerDMA invalid
                DRV_P2INVALID - SecondsPerDMA invalid

            C++ Equiv:
            unsigned int SetDMAParameters(int MaxImagesPerDMA, float SecondsPerDMA);

        """
        cMaxImagesPerDMA = c_int(MaxImagesPerDMA)
        cSecondsPerDMA = c_float(SecondsPerDMA)
        ret = self.dll.SetDMAParameters(cMaxImagesPerDMA, cSecondsPerDMA)
        return (ret)

    def SetDriverEvent(self, event):
        """
            Description:
            This function passes a Win32 Event handle to the SDK via which the the user software can be informed that something has occurred.
            For example the SDK can set the event when an acquisition has completed thus relieving the user code of having
            to continually pole to check on the status of the acquisition.
            The event will be set under the follow conditions:
            1) Acquisition completed or aborted.
            2) As each scan during an acquisition is completed.
            3) Temperature as stabilized, drifted from stabilization or could not be reached.
            When an event is triggered the user software can then use other SDK functions to determine what actually happened.
            Condition 1 and 2 can be tested via GetStatus function, while condition 3 checked via GetTemperature function.
            You must reset the event after it has been handled in order to receive additional triggers. Before deleting the event you must call SetDriverEvent with NULL as the parameter.

            Synopsis:
            ret = SetDriverEvent(event)

            Inputs:
            event - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Event set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Function not supported for operating system

            C++ Equiv:
            unsigned int SetDriverEvent(HANDLE driverEvent);

            See Also:
            GetStatus GetTemperature GetAcquisitionProgress

            Note: Not all programming environments allow the use of multiple threads and WIN32 events.

        """
        cevent = c_int(event)
        ret = self.dll.SetDriverEvent(cevent)
        return (ret)

    def SetDualExposureMode(self, mode):
        """
            Description:
            This function turns on and off the option to acquire 2 frames for each external trigger pulse.
            This mode is only available for certain sensors in run till abort mode, external trigger, full image.

            Synopsis:
            ret = SetDualExposureMode(mode)

            Inputs:
            mode - Enables/Disables dual exposure mode:
                1 - Enable mode
                0 - Disable mode

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Dual exposure mode not supported on this camera.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Mode parameter was not zero or one.

            C++ Equiv:
            unsigned int SetDualExposureMode(int mode);

            See Also:
            GetCapabilities SetDualExposureTimes GetDualExposureTimes

        """
        cmode = c_int(mode)
        ret = self.dll.SetDualExposureMode(cmode)
        return (ret)

    def SetDualExposureTimes(self, expTime1, expTime2):
        """
            Description:
            This function configures the two exposure times used in dual exposure mode.
            This mode is only available for certain sensors in run till abort mode, external trigger, full image.

            Synopsis:
            ret = SetDualExposureTimes(expTime1, expTime2)

            Inputs:
            expTime1 - the exposure time in seconds for each odd numbered frame.
            expTime2 - the exposure time in seconds for each even numbered frame.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Dual exposure mode not supported on this camera.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - First exposure out of range.
                DRV_P2INVALID - Second exposure out of range.

            C++ Equiv:
            unsigned int SetDualExposureTimes(float expTime1, float expTime2);

            See Also:
            GetCapabilities SetDualExposureMode GetDualExposureTimes

        """
        cexpTime1 = c_float(expTime1)
        cexpTime2 = c_float(expTime2)
        ret = self.dll.SetDualExposureTimes(cexpTime1, cexpTime2)
        return (ret)

    def SetEMAdvanced(self, state):
        """
            Description:
            This function turns on and off access to higher EM gain levels within the SDK.
            Typically, optimal signal to noise ratio and dynamic range is achieved between x1 to x300 EM Gain.
            Higher gains of > x300 are recommended for single photon counting only.
            Before using higher levels, you should ensure that light levels do not exceed the regime of tens of photons per pixel,
            otherwise accelerated ageing of the sensor can occur.

            Synopsis:
            ret = SetEMAdvanced(state)

            Inputs:
            state - Enables/Disables access to higher EM gain levels:
                1 - Enable access
                1 - Disable access

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_AVAILABLE - Advanced EM gain not available for this camera.
                DRV_ACQUIRING. - Acquisition in progress.
                DRV_P1INVALID - State parameter was not zero or one.

            C++ Equiv:
            unsigned int SetEMAdvanced(int state);

            See Also:
            GetCapabilities GetEMCCDGain SetEMCCDGain SetEMGainMode

        """
        cstate = c_int(state)
        ret = self.dll.SetEMAdvanced(cstate)
        return (ret)

    def SetEMCCDGain(self, gain):
        """
            Description:
            Allows the user to change the gain value. The valid range for the gain depends on what gain mode the camera is operating in.
            See SetEMGainMode to set the mode and GetEMGainRange to get the valid range to work with.
            To access higher gain values (>x300) see SetEMAdvanced.

            Synopsis:
            ret = SetEMCCDGain(gain)

            Inputs:
            gain - amount of gain applied.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for gain accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Gain value invalid.

            C++ Equiv:
            unsigned int SetEMCCDGain(int gain);

            See Also:
            GetEMCCDGain SetEMGainMode GetEMGainRange SetEMAdvanced

            Note: Only available on EMCCD sensor systems.

        """
        cgain = c_int(gain)
        ret = self.dll.SetEMCCDGain(cgain)
        return (ret)

    def SetEMClockCompensation(self, EMClockCompensationFlag):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetEMClockCompensation(EMClockCompensationFlag)

            Inputs:
            EMClockCompensationFlag -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetEMClockCompensation(int EMClockCompensationFlag);

        """
        cEMClockCompensationFlag = c_int(EMClockCompensationFlag)
        ret = self.dll.SetEMClockCompensation(cEMClockCompensationFlag)
        return (ret)

    def SetEMGainMode(self, mode):
        """
            Description:
            Set the EM Gain mode to one of the following possible settings.
            Mode 0: The EM Gain is controlled by DAC settings in the range 0-255. Default mode.
            1: The EM Gain is controlled by DAC settings in the range 0-4095.
            2: Linear mode.
            3: Real EM gain
            To access higher gain values (if available) it is necessary to enable advanced EM gain, see SetEMAdvanced.

            Synopsis:
            ret = SetEMGainMode(mode)

            Inputs:
            mode - EM Gain mode.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - EM Gain mode invalid.

            C++ Equiv:
            unsigned int SetEMGainMode(int mode);

        """
        cmode = c_int(mode)
        ret = self.dll.SetEMGainMode(cmode)
        return (ret)

    def SetESDEvent(self, event):
        """
            Description:
            This function passes a Win32 Event handle to the driver via which the driver can inform the user software that an ESD event has occurred.

            Synopsis:
            ret = SetESDEvent(event)

            Inputs:
            event - Win32 event handle

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Event set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_AVAILABLE - Function not supported for operating system

            C++ Equiv:
            unsigned int SetESDEvent(HANDLE event);

            See Also:
            GetCapabilities

        """
        cevent = c_int(event)
        ret = self.dll.SetESDEvent(cevent)
        return (ret)

    def SetExposureTime(self, time):
        """
            Description:
            This function will set the exposure time to the nearest valid value not less than the given value.
            The actual exposure time used is obtained by GetAcquisitionTimingsGetAcquisitionTimings.
            Please refer to SECTION 5 - ACQUISITION MODES for further information.

            Synopsis:
            ret = SetExposureTime(time)

            Inputs:
            time - the exposure time in seconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Exposure time accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Exposure Time invalid.

            C++ Equiv:
            unsigned int SetExposureTime(float time);

            See Also:
            GetAcquisitionTimings

            Note: For Classics, if the current acquisition mode is Single-Track, Multi-Track or Image then this function will actually set the Shutter Time. The actual exposure time used is obtained from the GetAcquisitionTimings functionGetAcquisitionTimings.

        """
        ctime = c_float(time)
        ret = self.dll.SetExposureTime(ctime)
        return (ret)

    def SetExternalTriggerTermination(self, uiTermination):
        """
            Description:
            This function can be used to set the external trigger termination mode.

            Synopsis:
            ret = SetExternalTriggerTermination(uiTermination)

            Inputs:
            uiTermination - trigger termination option.:
                0 - 50 ohm.
                1 - hi-Z.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Termination set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Trigger termination not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.

            C++ Equiv:
            unsigned int SetExternalTriggerTermination(at_u32 uiTermination);

            See Also:
            GetCapabilities GetExternalTriggerTermination

        """
        cuiTermination = c_uint(uiTermination)
        ret = self.dll.SetExternalTriggerTermination(cuiTermination)
        return (ret)

    def SetFanMode(self, mode):
        """
            Description:
            Allows the user to control the mode of the camera fan. If the system is cooled, the fan should only be turned off for short periods of time. During this time the body of the camera will warm up which could compromise cooling capabilities.
            If the camera body reaches too high a temperature, depends on camera, the buzzer will sound. If this happens, turn off the external power supply and allow the system to stabilize before continuing.

            Synopsis:
            ret = SetFanMode(mode)

            Inputs:
            mode - Fan mode setting:
                0 - Fan on full.
                1 - Fan on low.
                2 - Fan off,

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for mode accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Mode value invalid.

            C++ Equiv:
            unsigned int SetFanMode(int mode);

            See Also:
            GetCapabilities

        """
        cmode = c_int(mode)
        ret = self.dll.SetFanMode(cmode)
        return (ret)

    def SetFastExtTrigger(self, mode):
        """
            Description:
            This function will enable fast external triggering. When fast external triggering is enabled the system will NOT wait until a Keep Clean cycle has been completed before accepting the next trigger. This setting will only have an effect if the trigger mode has been set to External via SetTriggerModeSetTriggerMode.

            Synopsis:
            ret = SetFastExtTrigger(mode)

            Inputs:
            mode - 0    Disabled:
                1 - Enabled

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters accepted.

            C++ Equiv:
            unsigned int SetFastExtTrigger(int mode);

            See Also:
            SetTriggerMode

        """
        cmode = c_int(mode)
        ret = self.dll.SetFastExtTrigger(cmode)
        return (ret)

    def SetFastKinetics(self, exposedRows, seriesLength, time, mode, hbin, vbin):
        """
            Description:
            This function will set the parameters to be used when taking a fast kinetics acquisition.

            Synopsis:
            ret = SetFastKinetics(exposedRows, seriesLength, time, mode, hbin, vbin)

            Inputs:
            exposedRows - sub-area height in rows.
            seriesLength - number in series.
            time - exposure time in seconds.
            mode - binning mode (0 - FVB , 4 - Image).
            hbin - horizontal binning.
            vbin - vertical binning (only used when in image mode).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - System not initialized.
                DRV_NOT_INITIALIZED - Acquisition in progress.
                DRV_ACQUIRING - Invalid height.
                DRV_P1INVALID - Invalid number in series.
                DRV_P2INVALID DRV_P3INVALID - Exposure time must be greater than 0.
                DRV_P4INVALID DRV_P5INVALID - Mode must be equal to 0 or 4.
                DRV_P6INVALID - Horizontal binning.
                All parameters accepted. - Vertical binning.

            C++ Equiv:
            unsigned int SetFastKinetics(int exposedRows, int seriesLength, float time, int mode, int hbin, int vbin);

            See Also:
            SetFKVShiftSpeed SetFastKineticsEx SetFKVShiftSpeed

            Note: For classic cameras the vertical and horizontal binning must be 1
                For non classic cameras it is recommended that you use SetFastKineticsEx


        """
        cexposedRows = c_int(exposedRows)
        cseriesLength = c_int(seriesLength)
        ctime = c_float(time)
        cmode = c_int(mode)
        chbin = c_int(hbin)
        cvbin = c_int(vbin)
        ret = self.dll.SetFastKinetics(
            cexposedRows, cseriesLength, ctime, cmode, chbin, cvbin)
        return (ret)

    def SetFastKineticsEx(self, exposedRows, seriesLength, time, mode, hbin, vbin, offset):
        """
            Description:
            This function is the same as SetFastKinetics with the addition of an Offset parameter, which will inform the SDK of the first row to be used.

            Synopsis:
            ret = SetFastKineticsEx(exposedRows, seriesLength, time, mode, hbin, vbin, offset)

            Inputs:
            exposedRows - sub-area height in rows.
            seriesLength - number in series.
            time - exposure time in seconds.
            mode - binning mode (0 - FVB , 4 - Image).
            hbin - horizontal binning.
            vbin - vertical binning (only used when in image mode).
            offset - offset of first row to be used in Fast Kinetics from the bottom of the CCD.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid height.
                DRV_P2INVALID - Invalid number in series.
                DRV_P3INVALID - Exposure time must be greater than 0.
                DRV_P4INVALID - Mode must be equal to 0 or 4.
                DRV_P5INVALID - Horizontal binning.
                DRV_P6INVALID - Vertical binning.
                DRV_P7INVALID - Offset not within CCD limits

            C++ Equiv:
            unsigned int SetFastKineticsEx(int exposedRows, int seriesLength, float time, int mode, int hbin, int vbin, int offset);

            See Also:
            SetFKVShiftSpeed SetFastKinetics SetFKVShiftSpeed

            Note: For classic cameras the offset must be 0 and the vertical and horizontal binning must be 1
                For iDus, it is recommended that you set horizontal binning to 1


        """
        cexposedRows = c_int(exposedRows)
        cseriesLength = c_int(seriesLength)
        ctime = c_float(time)
        cmode = c_int(mode)
        chbin = c_int(hbin)
        cvbin = c_int(vbin)
        coffset = c_int(offset)
        ret = self.dll.SetFastKineticsEx(
            cexposedRows, cseriesLength, ctime, cmode, chbin, cvbin, coffset)
        return (ret)

    def SetFastKineticsStorageMode(self, mode):
        """
            Description:
            Allows the user to increase the number of frames which can be acquired in fast kinetics mode when using vertical binning. When ‘binning in storage area’ is selected the offset cannot be adjusted from the bottom of the sensor and the maximum signal level will be reduced.

            Synopsis:
            ret = SetFastKineticsStorageMode(mode)

            Inputs:
            mode - vertically bin in readout register (0)                 vertically bin in storage area (1)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for mode accepted
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported
                DRV_P1INVALID - Mode value invalid

            C++ Equiv:
            unsigned int SetFastKineticsStorageMode(int mode);

            See Also:
            GetCapabilities

        """
        cmode = c_int(mode)
        ret = self.dll.SetFastKineticsStorageMode(cmode)
        return (ret)

    def SetFastKineticsTimeScanMode(self, exposedRows, seriesLength, mode):
        """
            Description:
            When triggered, the camera starts shifting collected data until exposedRows of data are collected. There is no dwell time between row shifts. After the data is collected the entire Sensor Width x exposedRows image is readout and available.  This is repeated seriesLength times.  The data can be accumulated or presented as a series.

            Synopsis:
            ret = SetFastKineticsTimeScanMode(exposedRows, seriesLength, mode)

            Inputs:
            exposedRows - sub-area height in rows
            seriesLength - number in series
            mode - Off (0), Accumulate (1), Series (2)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for mode accepted
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported
                DRV_P1INVALID - Invalid height
                DRV_P2INVALID - Invalid number in series
                DRV_P3INVALID - Mode must be equal to 0, 1 or 2

            C++ Equiv:
            unsigned int SetFastKineticsTimeScanMode(int exposedRows, int seriesLength, int mode);

            See Also:
            GetCapabilities

        """
        cexposedRows = c_int(exposedRows)
        cseriesLength = c_int(seriesLength)
        cmode = c_int(mode)
        ret = self.dll.SetFastKineticsTimeScanMode(
            cexposedRows, cseriesLength, cmode)
        return (ret)

    def SetFilterMode(self, mode):
        """
            Description:
            This function will set the state of the cosmic ray filter mode for future acquisitions. If the filter mode is on, consecutive scans in an accumulation will be compared and any cosmic ray-like features that are only present in one scan will be replaced with a scaled version of the corresponding pixel value in the correct scan.

            Synopsis:
            ret = SetFilterMode(mode)

            Inputs:
            mode - current state of filter:
                0 - OFF
                2 - ON

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Filter mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Mode is out off range.

            C++ Equiv:
            unsigned int SetFilterMode(int mode);

            See Also:
            GetFilterMode

        """
        cmode = c_int(mode)
        ret = self.dll.SetFilterMode(cmode)
        return (ret)

    def SetFilterParameters(self, width, sensitivity, range, accept, smooth, noise):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetFilterParameters(width, sensitivity, range, accept, smooth, noise)

            Inputs:
            width -
            sensitivity -
            range -
            accept -
            smooth -
            noise -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetFilterParameters(int width, float sensitivity, int range, float accept, int smooth, int noise);

        """
        cwidth = c_int(width)
        csensitivity = c_float(sensitivity)
        crange = c_int(range)
        caccept = c_float(accept)
        csmooth = c_int(smooth)
        cnoise = c_int(noise)
        ret = self.dll.SetFilterParameters(
            cwidth, csensitivity, crange, caccept, csmooth, cnoise)
        return (ret)

    def SetFKVShiftSpeed(self, index):
        """
            Description:
            This function will set the fast kinetics vertical shift speed to one of the possible speeds of the system. It will be used for subsequent acquisitions.

            Synopsis:
            ret = SetFKVShiftSpeed(index)

            Inputs:
            index - the speed to be used:
                0 - to GetNumberFKVShiftSpeedsGetNumberFKVShiftSpeeds-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Fast kinetics vertical shift speed set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Index is out off range.

            C++ Equiv:
            unsigned int SetFKVShiftSpeed(int index);

            See Also:
            GetNumberFKVShiftSpeeds GetFKVShiftSpeedF

            Note: Only available if camera is Classic or iStar.

        """
        cindex = c_int(index)
        ret = self.dll.SetFKVShiftSpeed(cindex)
        return (ret)

    def SetFPDP(self, state):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetFPDP(state)

            Inputs:
            state -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetFPDP(int state);

        """
        cstate = c_int(state)
        ret = self.dll.SetFPDP(cstate)
        return (ret)

    def SetFrameTransferMode(self, mode):
        """
            Description:
            This function will set whether an acquisition will readout in Frame Transfer Mode. If the acquisition mode is Single Scan or Fast Kinetics this call will have no affect.

            Synopsis:
            ret = SetFrameTransferMode(mode)

            Inputs:
            mode - mode:
                0 - OFF
                1 - ON

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Frame transfer mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid parameter.

            C++ Equiv:
            unsigned int SetFrameTransferMode(int mode);

            See Also:
            SetAcquisitionMode

            Note: Only available if CCD is a Frame Transfer chip.

        """
        cmode = c_int(mode)
        ret = self.dll.SetFrameTransferMode(cmode)
        return (ret)

    def SetFrontEndEvent(self, event):
        """
            Description:
            This function passes a Win32 Event handle to the driver via which the driver can inform the user software that the Front End cooler has overheated or returned to a normal state. To determine what event has actually occurred call the GetFrontEndStatus function. This may give the user software an opportunity to perform other actions that will not affect the readout of the current acquisition.

            Synopsis:
            ret = SetFrontEndEvent(event)

            Inputs:
            event - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Event set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_AVAILABLE - Function not supported for operating system

            C++ Equiv:
            unsigned int SetFrontEndEvent(at_32 driverEvent);

            See Also:
            GetFrontEndStatus

        """
        cevent = c_int(event)
        ret = self.dll.SetFrontEndEvent(cevent)
        return (ret)

    def SetFullImage(self, hbin, vbin):
        """
            Description:
            Deprecated see Note:
            This function will set the horizontal and vertical binning to be used when taking a full resolution image.

            Synopsis:
            ret = SetFullImage(hbin, vbin)

            Inputs:
            hbin - number of pixels to bin horizontally
            vbin - number of pixels to bin vertically

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Binning parameters accepted
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - Horizontal binning parameter invalid
                DRV_P2INVALID - Vertical binning parameter invalid

            C++ Equiv:
            unsigned int SetFullImage(int hbin, int vbin); // deprecated

            See Also:
            SetReadMode

            Note: Deprecated by SetImageGetNumberHSSpeeds

        """
        chbin = c_int(hbin)
        cvbin = c_int(vbin)
        ret = self.dll.SetFullImage(chbin, cvbin)
        return (ret)

    def SetFVBHBin(self, bin):
        """
            Description:
            This function sets the horizontal binning used when acquiring in Full Vertical Binned read mode.

            Synopsis:
            ret = SetFVBHBin(bin)

            Inputs:
            bin - Binning size.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Binning set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid binning size.

            C++ Equiv:
            unsigned int SetFVBHBin(int bin);

            See Also:
            SetReadMode

            Note: 1) If the detector width is not a multiple of the binning DRV_BINNING_ERROR will be returned from PrepareAcquisition and/or StartAcquisition
                2) For iDus, it is recommended that you set horizontal binning to 1

        """
        cbin = c_int(bin)
        ret = self.dll.SetFVBHBin(cbin)
        return (ret)

    def SetGain(self, gain):
        """
            Description:
            Deprecated for SetMCPGain.

            Synopsis:
            ret = SetGain(gain)

            Inputs:
            gain -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetGain(int gain); // deprecated

        """
        cgain = c_int(gain)
        ret = self.dll.SetGain(cgain)
        return (ret)

    def SetGate(self, delay, width, stepRenamed):
        """
            Description:
            This function sets the Gater parameters for an ICCD system. The image intensifier of the Andor ICCD acts as a shutter on nanosecond time-scales using a process known as gating.

            Synopsis:
            ret = SetGate(delay, width, stepRenamed)

            Inputs:
            delay - Sets the delay(>=0) between the T0 and C outputs on the SRS box to delay nanoseconds.
            width - Sets the width(>=0) of the gate in nanoseconds
            stepRenamed - Sets the amount(<>0, in nanoseconds) by which the gate position is moved in time after each scan in a kinetic series.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gater parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_GPIBERROR - Error communicating with GPIB card.
                DRV_P1INVALID - Invalid delay
                DRV_P2INVALID - Invalid width.
                DRV_P3INVALID - Invalid step.

            C++ Equiv:
            unsigned int SetGate(float delay, float width, float stepRenamed);

            See Also:
            SetDelayGenerator

            Note: Available on ICCD.

        """
        cdelay = c_float(delay)
        cwidth = c_float(width)
        cstepRenamed = c_float(stepRenamed)
        ret = self.dll.SetGate(cdelay, cwidth, cstepRenamed)
        return (ret)

    def SetGateMode(self, gatemode):
        """
            Description:
            Allows the user to control the photocathode gating mode.

            Synopsis:
            ret = SetGateMode(gatemode)

            Inputs:
            gatemode - the gate mode.:
                0 - Fire ANDed with the Gate input.
                1 - Gating controlled from Fire pulse only.
                2 - Gating controlled from SMB Gate input only.
                3 - Gating ON continuously.
                4 - Gating OFF continuously.
                5 - Gate using DDG

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Gating mode accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Gating mode invalid.

            C++ Equiv:
            unsigned int SetGateMode(int gatemode);

            See Also:
            GetCapabilities SetMCPGain SetMCPGating

        """

        cgatemode = c_int(gatemode)
        ret = self.dll.SetGateMode(cgatemode)
        return (ret)

    def SetHighCapacity(self, state):
        """
            Description:
            This function switches between high sensitivity and high capacity functionality. With high capacity enabled the output amplifier is switched to a mode of operation which reduces the responsivity thus allowing the reading of larger charge packets during binning operations.

            Synopsis:
            ret = SetHighCapacity(state)

            Inputs:
            state - Enables/Disables High Capacity functionality:
                1 - Enable High Capacity (Disable High Sensitivity)
                0 - Disable High Capacity (Enable High Sensitivity)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - State parameter was not zero or one.

            C++ Equiv:
            unsigned int SetHighCapacity(int state);

            See Also:
            GetCapabilities

        """
        cstate = c_int(state)
        ret = self.dll.SetHighCapacity(cstate)
        return (ret)

    def SetHorizontalSpeed(self, index):
        """
            Description:
            Deprecated see Note:
            This function will set the horizontal speed to one of the possible speeds of the system. It will be used for subsequent acquisitions.

            Synopsis:
            ret = SetHorizontalSpeed(index)

            Inputs:
            index - the horizontal speed to be used:
                0 - to GetNumberHorizontalSpeedsGetNumberHorizontalSpeeds-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Horizontal speed set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Index is out off range.

            C++ Equiv:
            unsigned int SetHorizontalSpeed(int index); // deprecated

            See Also:
            GetNumberHorizontalSpeeds GetHorizontalSpeed

            Note: Deprecated by SetHSSpeedGetNumberHSSpeeds

        """
        cindex = c_int(index)
        ret = self.dll.SetHorizontalSpeed(cindex)
        return (ret)

    def SetHSSpeed(self, typ, index):
        """
            Description:
            This function will set the speed at which the pixels are shifted into the output node during the readout phase of an acquisition. Typically your camera will be capable of operating at several horizontal shift speeds. To get the actual speed that an index corresponds to use the GetHSSpeed function. Ensure the desired A/D channel has been set with SetADChannel before calling SetHSSpeed.

            Synopsis:
            ret = SetHSSpeed(typ, index)

            Inputs:
            typ - output amplification.:
                0 - electron multiplication/Conventional(clara).
                1 - conventional/Extended NIR mode(clara).
            index - the horizontal speed to be used:
                0 - to GetNumberHSSpeeds()GetNumberHSSpeeds-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Horizontal speed set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Mode is invalid.
                DRV_P2INVALID - Index is out off range.

            C++ Equiv:
            unsigned int SetHSSpeed(int typ, int index);

            See Also:
            GetNumberHSSpeeds GetHSSpeed GetNumberAmp

        """
        ctyp = c_int(typ)
        cindex = c_int(index)
        ret = self.dll.SetHSSpeed(ctyp, cindex)
        return (ret)

    def SetImage(self, hbin, vbin, hstart, hend, vstart, vend):
        """
            Description:
            This function will set the horizontal and vertical binning to be used when taking a full resolution image.

            Synopsis:
            ret = SetImage(hbin, vbin, hstart, hend, vstart, vend)

            Inputs:
            hbin - number of pixels to bin horizontally.
            vbin - number of pixels to bin vertically.
            hstart - Start column (inclusive).
            hend - End column (inclusive).
            vstart - Start row (inclusive).
            vend - End row (inclusive).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Binning parameters invalid.
                DRV_P2INVALID - Binning parameters invalid.
                DRV_P3INVALID - Sub-area co-ordinate is invalid.
                DRV_P4INVALID - Sub-area co-ordinate is invalid.
                DRV_P5INVALID - Sub-area co-ordinate is invalid.
                DRV_P6INVALID - Sub-area co-ordinate is invalid.

            C++ Equiv:
            unsigned int SetImage(int hbin, int vbin, int hstart, int hend, int vstart, int vend);

            See Also:
            SetReadMode

            Note: For iDus, it is recommended that you set horizontal binning to 1

        """
        chbin = c_int(hbin)
        cvbin = c_int(vbin)
        chstart = c_int(hstart)
        chend = c_int(hend)
        cvstart = c_int(vstart)
        cvend = c_int(vend)
        ret = self.dll.SetImage(chbin, cvbin, chstart, chend, cvstart, cvend)
        return (ret)

    def SetImageFlip(self, iHFlip, iVFlip):
        """
            Description:
            This function will cause data output from the SDK to be flipped on one or both axes. This flip is not done in the camera, it occurs after the data is retrieved and will increase processing overhead. If flipping could be implemented by the user more efficiently then use of this function is not recomended. E.g writing to file or displaying on screen.

            Synopsis:
            ret = SetImageFlip(iHFlip, iVFlip)

            Inputs:
            iHFlip - Sets horizontal flipping.
            iVFlip - Sets vertical flipping..:
                1 - Enables Flipping
                0 - Disables Flipping

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - HFlip parameter invalid.
                DRV_P2INVALID - VFlip parameter invalid

            C++ Equiv:
            unsigned int SetImageFlip(int iHFlip, int iVFlip);

            See Also:
            SetImageRotate

            Note: If this function is used in conjunction with the SetImageRotate function the rotation will occur before the flip regardless of which order the functions are called.

        """
        ciHFlip = c_int(iHFlip)
        ciVFlip = c_int(iVFlip)
        ret = self.dll.SetImageFlip(ciHFlip, ciVFlip)
        return (ret)

    def SetImageRotate(self, iRotate):
        """
            Description:
            This function will cause data output from the SDK to be rotated on one or both axes. This rotate is not done in the camera, it occurs after the data is retrieved and will increase processing overhead. If the rotation could be implemented by the user more efficiently then use of this function is not recomended. E.g writing to file or displaying on screen.

            Synopsis:
            ret = SetImageRotate(iRotate)

            Inputs:
            iRotate - Rotation setting:
                0 - No rotation.
                1 - Rotate 90 degrees clockwise.
                2 - Rotate 90 degrees anti-clockwise.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - All parameters accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_P1INVALID - Rotate parameter invalid.

            C++ Equiv:
            unsigned int SetImageRotate(int iRotate);

            See Also:
            SetImageFlip

            Note: If this function is used in conjunction with the SetImageFlip function the rotation will occur before the flip regardless of which order the functions are called. 180 degree rotation can be achieved using the SetImageFlip function by selecting both horizontal and vertical flipping.

        """
        ciRotate = c_int(iRotate)
        ret = self.dll.SetImageRotate(ciRotate)
        return (ret)

    def SetIODirection(self, index, iDirection):
        """
            Description:
            Available in some systems are a number of IOs that can be configured to be inputs or outputs. This function sets the current state of a particular IO.

            Synopsis:
            ret = SetIODirection(index, iDirection)

            Inputs:
            index - IO index:
                0 - to GetNumberIO() - 1
            iDirection - requested direction for this index.:
                0 - 0 Output
                1 - 1 Input

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IO direction set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Invalid direction.
                DRV_NOT_AVAILABLE - Feature not available.

            C++ Equiv:
            unsigned int SetIODirection(int index, int iDirection);

            See Also:
            GetNumberIO GetIOLevel GetIODirection SetIOLevel

        """
        cindex = c_int(index)
        ciDirection = c_int(iDirection)
        ret = self.dll.SetIODirection(cindex, ciDirection)
        return (ret)

    def SetIOLevel(self, index, iLevel):
        """
            Description:
            Available in some systems are a number of IOs that can be configured to be inputs or outputs. This function sets the current state of a particular IO.

            Synopsis:
            ret = SetIOLevel(index, iLevel)

            Inputs:
            index - IO index:
                0 - to GetNumberIO() - 1
            iLevel - current level for this index.:
                0 - 0 Low
                1 - 1 High

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - IO level set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid index.
                DRV_P2INVALID - Invalid level.
                DRV_NOT_AVAILABLE - Feature not available.

            C++ Equiv:
            unsigned int SetIOLevel(int index, int iLevel);

            See Also:
            GetNumberIO GetIOLevel GetIODirection SetIODirection

        """
        cindex = c_int(index)
        ciLevel = c_int(iLevel)
        ret = self.dll.SetIOLevel(cindex, ciLevel)
        return (ret)

    def SetIRIGModulation(self, mode):
        """
            Description:
            This function allows the camera to be configured to expect the IRIG modulation type produced by the external IRIG device.

            Synopsis:
            ret = SetIRIGModulation(mode)

            Inputs:
            mode - char mode: mode.
                0  unmodulated
                1  modulated :
                DRV_SUCCESS - Mode successfully selected
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported on this camera
                DRV_P1INVALID - Requested mode isn’t valid

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Mode successfully selected
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported on this camera
                DRV_P1INVALID - Requested mode isn’t valid

            C++ Equiv:
            unsigned int SetIRIGModulation(char mode);

            See Also:
            GetCapabilities GetIRIGData

            Note: This function allows the camera to be configured to expect the IRIG modulation type produced by the external IRIG device.

        """
        cmode = c_char(mode)
        ret = self.dll.SetIRIGModulation(cmode)
        return (ret)

    def SetIsolatedCropMode(self, active, cropheight, cropwidth, vbin, hbin):
        """
            Description:
            This function effectively reduces the dimensions of the CCD by excluding some rows or columns to achieve higher throughput. In isolated crop mode iXon, Newton and iKon cameras can operate in either Full Vertical Binning or Imaging read modes. iDus can operate in Full Vertical Binning read mode only.
            Note: It is important to ensure that no light falls on the excluded region otherwise the acquired data will be corrupted.

            Synopsis:
            ret = SetIsolatedCropMode(active, cropheight, cropwidth, vbin, hbin)

            Inputs:
            active - Crop mode active:
                1 - Crop mode is ON.
                Crop - 0 - Crop mode is OFF.
            cropheight - The selected crop height. This value must be between 1 and the CCD height.
            cropwidth - The selected crop width. This value must be between 1 and the CCD width.
            vbin - The selected vertical binning.
            hbin - The selected horizontal binning.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - active parameter was not zero or one
                DRV_P2INVALID - Invalid crop height
                DRV_P3INVALID - Invalid crop width
                DRV_P4INVALID - Invalid vertical binning
                DRV_P5INVALID - Invalid horizontal binning
                DRV_NOT_SUPPORTED - Either the camera does not support isolated Crop mode or the read mode is invalid

            C++ Equiv:
            unsigned int SetIsolatedCropMode(int active, int cropheight, int cropwidth, int vbin, int hbin);

            See Also:
            GetDetector SetReadMode

            Note: For iDus, it is recommended that you set horizontal binning to 1

        """
        cactive = c_int(active)
        ccropheight = c_int(cropheight)
        ccropwidth = c_int(cropwidth)
        cvbin = c_int(vbin)
        chbin = c_int(hbin)
        ret = self.dll.SetIsolatedCropMode(
            cactive, ccropheight, ccropwidth, cvbin, chbin)
        return (ret)

    def SetIsolatedCropModeEx(self, active, cropheight, cropwidth, vbin, hbin, cropleft, cropbottom):
        """
            Description:
            This function effectively reduces the dimensions of the CCD by excluding some rows or columns to achieve higher throughput. This feature is currently only available for iXon Ultra and can only be used in Image readout mode with the EM output amplifier.
            Note: It is important to ensure that no light falls on the excluded region otherwise the acquired data will be corrupted.
            The following centralized regions of interest are recommended to be used with this mode to achieve the fastest possible frame rates. The table below shows the optimally positioned ROI coordinates recommended to be used with this mode:
            ROI
            Crop Left Start Position
            Crop Right Position
            Crop Bottom Start Position
            Crop Top Position
            32 x 32
            241
            272
            240
            271
            64 x 64
            219
            282
            224
            287
            96 x 96
            209
            304
            208
            303
            128 x 128
            189
            316
            192
            319
            192 x 192
            157
            348
            160
            351
            256 x 256
            123
            378
            128
            383
            496 x 4
            8
            503
            254
            257
            496 x 8
            8
            503
            252
            259
            496 x 16
            8
            503
            249
            262

            Synopsis:
            ret = SetIsolatedCropModeEx(active, cropheight, cropwidth, vbin, hbin, cropleft, cropbottom)

            Inputs:
            active - Crop mode active.:
                1 - Crop mode is ON.
                0 - Crop mode is OFF.
            cropheight - The selected crop height. This value must be between 1 and the CCD height.
            cropwidth - The selected crop width. This value must be between 1 and the CCD width.
            vbin - vbinThe selected vertical binning.
            hbin - hbinThe selected horizontal binning.
            cropleft - The selected crop left start position
            cropbottom - The selected crop bottom start position

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - active parameter was not zero or one
                DRV_P2INVALID - Invalid crop height
                DRV_P3INVALID - Invalid crop width
                DRV_P4INVALID - Invalid vertical binning
                DRV_P5INVALID - Invalid horizontal binning
                DRV_P6INVALID - Invalid crop left start position
                DRV_P7INVALID - Invalid crop bottom start position
                DRV_NOT_SUPPORTED - The camera does not support isolated crop mode
                DRV_NOT_AVAILABLE - Invalid read mode

            C++ Equiv:
            unsigned int SetIsolatedCropModeEx(int active, int cropheight, int cropwidth, int vbin, int hbin, int cropleft, int cropbottom);

            See Also:
            GetDetector SetReadMode

        """
        cactive = c_int(active)
        ccropheight = c_int(cropheight)
        ccropwidth = c_int(cropwidth)
        cvbin = c_int(vbin)
        chbin = c_int(hbin)
        ccropleft = c_int(cropleft)
        ccropbottom = c_int(cropbottom)
        ret = self.dll.SetIsolatedCropModeEx(
            cactive, ccropheight, ccropwidth, cvbin, chbin, ccropleft, ccropbottom)
        return (ret)

    def SetIsolatedCropModeType(self, mode):
        """
            Description:
            This function determines the method by which data is transferred in isolated crop mode. The default method is High Speed where multiple frames may be stored in the storage area of the sensor before they are read out.  In Low Latency mode, each cropped frame is read out as it happens.

            Synopsis:
            ret = SetIsolatedCropModeType(mode)

            Inputs:
            mode - 0 – High Speed.  1 – Low Latency.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameter set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_P1INVALID - mode parameter was not zero or one
                DRV_NOT_SUPPORTED - Either the camera does not support isolated Crop mode or the read mode is invalid

            C++ Equiv:
            unsigned int SetIsolatedCropModeType(int mode);

            See Also:
            GetDetector SetIsolatedCropMode GetCapabilities

        """
        cmode = c_int(mode)
        ret = self.dll.SetIsolatedCropModeType(cmode)
        return (ret)

    def SetKineticCycleTime(self, time):
        """
            Description:
            This function will set the kinetic cycle time to the nearest valid value not less than the given value. The actual time used is obtained by GetAcquisitionTimingsGetAcquisitionTimings. . Please refer to SECTION 5 - ACQUISITION MODES for further information.

            Synopsis:
            ret = SetKineticCycleTime(time)

            Inputs:
            time - the kinetic cycle time in seconds.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Cycle time accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Time invalid.

            C++ Equiv:
            unsigned int SetKineticCycleTime(float time);

            See Also:
            SetNumberKinetics

        """
        ctime = c_float(time)
        ret = self.dll.SetKineticCycleTime(ctime)
        return (ret)

    def SetMCPGain(self, gain):
        """
            Description:
            Allows the user to control the voltage across the microchannel plate. Increasing the gain increases the voltage and so amplifies the signal. The gain range can be returned using GetMCPGainRange.

            Synopsis:
            ret = SetMCPGain(gain)

            Inputs:
            gain - amount of gain applied.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for gain accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with device.
                DRV_P1INVALID - Gain value invalid.

            C++ Equiv:
            unsigned int SetMCPGain(int gain);

            See Also:
            GetMCPGainRange SetGateMode SetMCPGating

            Note: Available on iStar.

        """
        cgain = c_int(gain)
        ret = self.dll.SetMCPGain(cgain)
        return (ret)

    def SetMCPGating(self, gating):
        """
            Description:
            This function controls the MCP gating.

            Synopsis:
            ret = SetMCPGating(gating)

            Inputs:
            gating - ON/OFF switch for the MCP gating.:
                0 - to switch MCP gating OFF.
                1 - to switch MCP gating ON.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Value for gating accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_I2CTIMEOUT - I2C command timed out.
                DRV_I2CDEVNOTFOUND - I2C device not present.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Value for gating invalid.

            C++ Equiv:
            unsigned int SetMCPGating(int gating);

            See Also:
            SetMCPGain SetGateMode

            Note: Available on some ICCD models.

        """
        cgating = c_int(gating)
        ret = self.dll.SetMCPGating(cgating)
        return (ret)

    def SetMessageWindow(self, wnd):
        """
            Description:
            This function is reserved.

            Synopsis:
            ret = SetMessageWindow(wnd)

            Inputs:
            wnd -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetMessageWindow(at_32 wnd);

        """
        cwnd = c_int(wnd)
        ret = self.dll.SetMessageWindow(cwnd)
        return (ret)

    def SetMetaData(self, state):
        """
            Description:
            This function activates the meta data option.

            Synopsis:
            ret = SetMetaData(state)

            Inputs:
            state - ON/OFF switch for the meta data option.:
                0 - to switch meta data OFF.
                1 - to switch meta data ON.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Meta data option accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid state.
                DRV_NOT_AVAILABLE - Feature not available.

            C++ Equiv:
            unsigned int SetMetaData(int state);

            See Also:
            GetMetaDataInfo

        """
        cstate = c_int(state)
        ret = self.dll.SetMetaData(cstate)
        return (ret)

    def SetMultiTrack(self, number, height, offset):
        """
            Description:
            This function will set the multi-Track parameters. The tracks are automatically spread evenly over the detector. Validation of the parameters is carried out in the following order:
            * Number of tracks,
            * Track height
            * Offset.
            The first pixels row of the first track is returned via bottom.
            The number of rows between each track is returned via gap.

            Synopsis:
            (ret, bottom, gap) = SetMultiTrack(number, height, offset)

            Inputs:
            number - number tracks (1 to number of vertical pixels)
            height - height of each track (>0 (maximum depends on number of tracks))
            offset - vertical displacement of tracks.   (depends on number of tracks and track height)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number of tracks invalid.
                DRV_P2INVALID - Track height invalid.
                DRV_P3INVALID - Offset invalid.
            bottom - first pixels row of the first track
            gap - number of rows between each track (could be 0)

            C++ Equiv:
            unsigned int SetMultiTrack(int number, int height, int offset, int * bottom, int * gap);

            See Also:
            SetReadMode StartAcquisition SetRandomTracks

        """
        cnumber = c_int(number)
        cheight = c_int(height)
        coffset = c_int(offset)
        cbottom = c_int()
        cgap = c_int()
        ret = self.dll.SetMultiTrack(
            cnumber, cheight, coffset, byref(cbottom), byref(cgap))
        return (ret, cbottom.value, cgap.value)

    def SetMultiTrackHBin(self, bin):
        """
            Description:
            This function sets the horizontal binning used when acquiring in Multi-Track read mode.

            Synopsis:
            ret = SetMultiTrackHBin(bin)

            Inputs:
            bin - Binning size.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Binning set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid binning size.

            C++ Equiv:
            unsigned int SetMultiTrackHBin(int bin);

            See Also:
            SetReadMode SetMultiTrack SetReadMode

            Note: 1) If the multitrack range is not a multiple of the binning DRV_BINNING_ERROR will be returned from PrepareAcquisition and/or StartAcquisition
                2) For iDus, it is recommended that you set horizontal binning to 1

        """
        cbin = c_int(bin)
        ret = self.dll.SetMultiTrackHBin(cbin)
        return (ret)

    def SetMultiTrackHRange(self, iStart, iEnd):
        """
            Description:
            This function sets the horizontal range used when acquiring in Multi Track read mode.

            Synopsis:
            ret = SetMultiTrackHRange(iStart, iEnd)

            Inputs:
            iStart - First horizontal pixel in multi track mode.
            iEnd - iEndLast horizontal pixel in multi track mode.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Range set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_AVAILABLE - Feature not available for this camera.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid start position.
                DRV_P2INVALID - Invalid end position.

            C++ Equiv:
            unsigned int SetMultiTrackHRange(int iStart, int iEnd);

            See Also:
            SetReadMode SetMultiTrack SetReadMode

        """
        ciStart = c_int(iStart)
        ciEnd = c_int(iEnd)
        ret = self.dll.SetMultiTrackHRange(ciStart, ciEnd)
        return (ret)

    def SetMultiTrackScan(self, trackHeight, numberTracks, iSIHStart, iSIHEnd, trackHBinning, trackVBinning, trackGap, trackOffset, trackSkip, numberSubFrames):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetMultiTrackScan(trackHeight, numberTracks, iSIHStart, iSIHEnd, trackHBinning, trackVBinning, trackGap, trackOffset, trackSkip, numberSubFrames)

            Inputs:
            trackHeight -
            numberTracks -
            iSIHStart -
            iSIHEnd -
            trackHBinning -
            trackVBinning -
            trackGap -
            trackOffset -
            trackSkip -
            numberSubFrames -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetMultiTrackScan(int trackHeight, int numberTracks, int iSIHStart, int iSIHEnd, int trackHBinning, int trackVBinning, int trackGap, int trackOffset, int trackSkip, int numberSubFrames);

        """
        ctrackHeight = c_int(trackHeight)
        cnumberTracks = c_int(numberTracks)
        ciSIHStart = c_int(iSIHStart)
        ciSIHEnd = c_int(iSIHEnd)
        ctrackHBinning = c_int(trackHBinning)
        ctrackVBinning = c_int(trackVBinning)
        ctrackGap = c_int(trackGap)
        ctrackOffset = c_int(trackOffset)
        ctrackSkip = c_int(trackSkip)
        cnumberSubFrames = c_int(numberSubFrames)
        ret = self.dll.SetMultiTrackScan(ctrackHeight, cnumberTracks, ciSIHStart, ciSIHEnd,
                                         ctrackHBinning, ctrackVBinning, ctrackGap, ctrackOffset, ctrackSkip, cnumberSubFrames)
        return (ret)

    def SetNextAddress(self, lowAdd, highAdd, length, physical):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, data) = SetNextAddress(lowAdd, highAdd, length, physical)

            Inputs:
            lowAdd -
            highAdd -
            length -
            physical -

            Outputs:
            ret - Function Return Code
            data -

            C++ Equiv:
            unsigned int SetNextAddress(at_32 * data, long lowAdd, long highAdd, long length, long physical);

        """
        cdata = c_int()
        clowAdd = c_int(lowAdd)
        chighAdd = c_int(highAdd)
        clength = c_int(length)
        cphysical = c_int(physical)
        ret = self.dll.SetNextAddress(
            byref(cdata), clowAdd, chighAdd, clength, cphysical)
        return (ret, cdata.value)

    def SetNextAddress16(self, lowAdd, highAdd, length, physical):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            (ret, data) = SetNextAddress16(lowAdd, highAdd, length, physical)

            Inputs:
            lowAdd -
            highAdd -
            length -
            physical -

            Outputs:
            ret - Function Return Code
            data -

            C++ Equiv:
            unsigned int SetNextAddress16(at_32 * data, long lowAdd, long highAdd, long length, long physical);

        """
        cdata = c_int()
        clowAdd = c_int(lowAdd)
        chighAdd = c_int(highAdd)
        clength = c_int(length)
        cphysical = c_int(physical)
        ret = self.dll.SetNextAddress16(
            byref(cdata), clowAdd, chighAdd, clength, cphysical)
        return (ret, cdata.value)

    def SetNumberAccumulations(self, number):
        """
            Description:
            This function will set the number of scans accumulated in memory. This will only take effect if the acquisition mode is either Accumulate or Kinetic Series.

            Synopsis:
            ret = SetNumberAccumulations(number)

            Inputs:
            number - number of scans to accumulate

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Accumulations set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number of accumulates.

            C++ Equiv:
            unsigned int SetNumberAccumulations(int number);

            See Also:
            GetAcquisitionTimings SetAccumulationCycleTime SetAcquisitionMode SetExposureTime SetKineticCycleTime SetNumberKinetics

        """
        cnumber = c_int(number)
        ret = self.dll.SetNumberAccumulations(cnumber)
        return (ret)

    def SetNumberKinetics(self, number):
        """
            Description:
            This function will set the number of scans (possibly accumulated scans) to be taken during a single acquisition sequence. This will only take effect if the acquisition mode is Kinetic Series.

            Synopsis:
            ret = SetNumberKinetics(number)

            Inputs:
            number - number of scans to store

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Series length set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number in series invalid.

            C++ Equiv:
            unsigned int SetNumberKinetics(int number);

            See Also:
            GetAcquisitionTimings SetAccumulationCycleTime SetAcquisitionMode SetExposureTime SetKineticCycleTime

        """
        cnumber = c_int(number)
        ret = self.dll.SetNumberKinetics(cnumber)
        return (ret)

    def SetNumberPrescans(self, iNumber):
        """
            Description:
            This function will set the number of scans acquired before data is to be retrieved. This will only take effect if the acquisition mode is Kinetic Series.

            Synopsis:
            ret = SetNumberPrescans(iNumber)

            Inputs:
            iNumber - number of scans to ignore

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Prescans set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number of prescans invalid.

            C++ Equiv:
            unsigned int SetNumberPrescans(int iNumber);

            See Also:
            GetAcquisitionTimings SetAcquisitionMode SetKineticCycleTime SetNumberKinetics

        """
        ciNumber = c_int(iNumber)
        ret = self.dll.SetNumberPrescans(ciNumber)
        return (ret)

    def SetOutputAmplifier(self, typ):
        """
            Description:
            Some EMCCD systems have the capability to use a second output amplifier. This function will set the type of output amplifier to be used when reading data from the head for these systems.

            Synopsis:
            ret = SetOutputAmplifier(typ)

            Inputs:
            typ - the type of output amplifier.:
                0 - Standard EMCCD gain register (default)/Conventional(clara).
                1 - Conventional CCD register/Extended NIR mode(clara).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Series length set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Output amplifier type invalid.

            C++ Equiv:
            unsigned int SetOutputAmplifier(int typ);

            Note: 1. Available in Clara, iXon & Newton.
                2. If the current camera HSSpeed is not available when the amplifier is set then it will default to the maximum HSSpeed that is.


        """
        ctyp = c_int(typ)
        ret = self.dll.SetOutputAmplifier(ctyp)
        return (ret)

    def SetOverlapMode(self, mode):
        """
            Description:
            This function will set whether an acquisition will readout in Overlap Mode. If the acquisition mode is Single Scan or Fast Kinetics this call will have no affect.

            Synopsis:
            ret = SetOverlapMode(mode)

            Inputs:
            mode - mode:
                0 - OFF
                1 - ON

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Overlap mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid parameter.

            C++ Equiv:
            unsigned int SetOverlapMode(int mode);

            See Also:
            SetAcquisitionMode

            Note: Only available if CCD is an Overlap sensor.

        """
        cmode = c_int(mode)
        ret = self.dll.SetOverlapMode(cmode)
        return (ret)

    def SetOverTempEvent(self, event):
        """
            Description:
            This function passes a Win32 Event handle to the SDK via which the the user software can be informed that something has occurred. For example the SDK can set the event when an acquisition has completed thus relieving the user code of having to continually pole to check on the status of the acquisition.
            The event will be set when the camera temperature is too high.
            When an event is triggered the user software can then use other SDK functions to determine what actually happened.
            Condition 1 and 2 can be tested via GetStatus function, while condition 3 checked via GetTemperature function.
            You must reset the event after it has been handled in order to receive additional triggers.

            Synopsis:
            ret = SetOverTempEvent(event)

            Inputs:
            event - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Event set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Function not supported for operating system

            C++ Equiv:
            unsigned int SetOverTempEvent(HANDLE event);

            See Also:
            GetStatus GetTemperature

            Note: Not all programming environments allow the use of multiple threads and WIN32 events.

        """
        cevent = c_int(event)
        ret = self.dll.SetDriverEvent(cevent)
        return (ret)

    def SetPCIMode(self, mode, value):
        """
            Description:
            With the CCI23 card, events can be sent when the camera is starting to expose and when it has finished exposing. This function will control whether those events happen or not.

            Synopsis:
            ret = SetPCIMode(mode, value)

            Inputs:
            mode - currently must be set to 1
            value - 0 to disable the events, 1 to enable

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Acquisition Mode invalid

            C++ Equiv:
            unsigned int SetPCIMode(int mode, int value);

            See Also:
            SetAcqStatusEvent SetCameraStatusEnable

            Note: This is only supported by the CCI23 card. The software must register its event via the SetAcqStatusEvent. To specify which event the software is interested in use the SetCameraStatusEnable.

        """
        cmode = c_int(mode)
        cvalue = c_int(value)
        ret = self.dll.SetPCIMode(cmode, cvalue)
        return (ret)

    def SetPhosphorEvent(self, event):
        """
            Description:
            This function passes a Win32 Event handle to the driver via which the driver can inform the user software that the phosphor has saturated or returned to a normal state. To determine what event has actually occurred call the GetPhosphorStatus function. This may give the user software an opportunity to perform other actions that will not affect the readout of the current acquisition.

            Synopsis:
            ret = SetPhosphorEvent(event)

            Inputs:
            event - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Event set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_AVAILABLE - Function not supported for operating system

            C++ Equiv:
            unsigned int SetPhosphorEvent(at_32 driverEvent);

            See Also:
            GetPhosphorStatus

        """
        cevent = c_int(event)
        ret = self.dll.SetPhosphorEvent(cevent)
        return (ret)

    def SetPhotonCounting(self, state):
        """
            Description:
            This function activates the photon counting option.

            Synopsis:
            ret = SetPhotonCounting(state)

            Inputs:
            state - ON/OFF switch for the photon counting option.:
                0 - to switch photon counting OFF.
                1 - to switch photon counting ON.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - photon counting option accepted.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.

            C++ Equiv:
            unsigned int SetPhotonCounting(int state);

            See Also:
            SetPhotonCountingThreshold

        """
        cstate = c_int(state)
        ret = self.dll.SetPhotonCounting(cstate)
        return (ret)

    def SetPhotonCountingDivisions(self, noOfDivisions, divisions):
        """
            Description:
            This function sets the thresholds for the photon counting option.

            Synopsis:
            (ret) = SetPhotonCountingDivisions(noOfDivisions, divisions)

            Inputs:
                noOfDivisions - number of thresholds to be used.
                divisions - threshold levels.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Thresholds accepted.
                DRV_P1INVALID - Number of thresholds outside valid range
                DRV_P2INVALID - Thresholds outside valid range
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_NOT_SUPPORTED - Feature not supported.

            C++ Equiv:
            unsigned int SetPhotonCountingDivisions(at_u32 noOfDivisions, at_32 * divisions);

            See Also:
            SetPhotonCounting GetNumberPhotonCountingDivisions

        """
        cnoOfDivisions = c_uint(noOfDivisions)
        cdivisions = (c_int * noOfDivisions)(*divisions)
        ret = self.dll.SetPhotonCountingDivisions(cnoOfDivisions, cdivisions)
        return (ret)

    def SetPhotonCountingThreshold(self, min, max):
        """
            Description:
            This function sets the minimum and maximum threshold for the photon counting option.

            Synopsis:
            ret = SetPhotonCountingThreshold(min, max)

            Inputs:
            min - minimum threshold in counts for photon counting.
            max - maximum threshold in counts for photon counting

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Thresholds accepted.
                DRV_P1INVALID - Minimum threshold outside valid range (1-65535)
                DRV_P2INVALID - Maximum threshold outside valid range
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.

            C++ Equiv:
            unsigned int SetPhotonCountingThreshold(long min, long max);

            See Also:
            SetPhotonCounting

        """
        cmin = c_int(min)
        cmax = c_int(max)
        ret = self.dll.SetPhotonCountingThreshold(cmin, cmax)
        return (ret)

    def SetPixelMode(self, bitdepth, colormode):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetPixelMode(bitdepth, colormode)

            Inputs:
            bitdepth -
            colormode -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetPixelMode(int bitdepth, int colormode);

        """
        cbitdepth = c_int(bitdepth)
        ccolormode = c_int(colormode)
        ret = self.dll.SetPixelMode(cbitdepth, ccolormode)
        return (ret)

    def SetPreAmpGain(self, index):
        """
            Description:
            This function will set the pre amp gain to be used for subsequent acquisitions. The actual gain factor that will be applied can be found through a call to the GetPreAmpGain function.
            The number of Pre Amp Gains available is found by calling the GetNumberPreAmpGains function.

            Synopsis:
            ret = SetPreAmpGain(index)

            Inputs:
            index - index pre amp gain table:
                0 - to GetNumberPreAmpGainsGetNumberPreAmpGains-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Pre amp gain set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Index out of range.

            C++ Equiv:
            unsigned int SetPreAmpGain(int index);

            See Also:
            IsPreAmpGainAvailable GetNumberPreAmpGains GetPreAmpGain

            Note: Available on iDus, iXon & Newton.

        """
        cindex = c_int(index)
        ret = self.dll.SetPreAmpGain(cindex)
        return (ret)

    def SetRandomTracks(self, numTracks, areas):
        """
            Description:
            This function will set the Random-Track parameters. The positions of the tracks are validated to ensure that the tracks are in increasing order and do not overlap. The horizontal binning is set via the SetCustomTrackHBin function. The vertical binning is set to the height of each track.
            Some cameras need to have at least 1 row in between specified tracks. Ixon+ and the USB cameras allow tracks with no gaps in between.
            Example:
            Tracks specified as 20 30 31 40 tells the SDK that the first track starts at row 20 in the CCD and finishes at row 30. The next track starts at row 31 (no gap between tracks) and ends at row 40.

            Synopsis:
            ret = SetRandomTracks(numTracks, areas)

            Inputs:
            numTracks - number tracks:
                1 - to number of vertical pixels/2
            areas - pointer to an array of track positions. The array has the form:
                bottom1 - bottom1 top1, bottom2, top2 ... bottomN, topN

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Number of tracks invalid.
                DRV_P2INVALID - Track positions invalid.
                DRV_RANDOM_TRACK_ERROR - Invalid combination of tracks, out of memory or mode not available.

            C++ Equiv:
            unsigned int SetRandomTracks(int numTracks, int * areas);

            See Also:
            SetCustomTrackHBin SetReadMode StartAcquisition SetComplexImage

        """
        cnumTracks = c_int(numTracks)
        numTracks = numTracks * 2
        careas = (c_int * numTracks)(*areas)
        ret = self.dll.SetRandomTracks(cnumTracks, careas)
        return (ret)

    def SetReadMode(self, mode):
        """
            Description:
            This function will set the readout mode to be used on the subsequent acquisitions.

            Synopsis:
            ret = SetReadMode(mode)

            Inputs:
            mode - readout mode:
                0 - Full Vertical Binning
                1 - Multi-Track
                2 - Random-Track
                3 - Single-Track
                4 - Image

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Readout mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid readout mode passed.

            C++ Equiv:
            unsigned int SetReadMode(int mode);

            See Also:
            GetAcquisitionTimings SetAccumulationCycleTime SetAcquisitionMode SetExposureTime SetKineticCycleTime SetNumberAccumulations SetNumberKinetics

        """

        cmode = c_int(mode)
        ret = self.dll.SetReadMode(cmode)
        return (ret)

    def SetReadoutRegisterPacking(self, mode):
        """
            Description:
            This function will configure whether data is packed into the readout register to improve frame rates for sub-images.
            Note: It is important to ensure that no light falls outside of the sub-image area otherwise the acquired data will be corrupted.  Only currently available on iXon+ and iXon3.

            Synopsis:
            ret = SetReadoutRegisterPacking(mode)

            Inputs:
            mode - register readout mode:
                0 - Packing Off
                1 - Packing On

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Readout mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid readout mode passed.

            C++ Equiv:
            unsigned int SetReadoutRegisterPacking(int mode);

            See Also:
            GetAcquisitionTimings SetAccumulationCycleTime SetAcquisitionMode SetExposureTime SetKineticCycleTime SetNumberAccumulations SetNumberKinetics

        """
        cmode = c_int(mode)
        ret = self.dll.SetReadoutRegisterPacking(cmode)
        return (ret)

    def SetRegisterDump(self, mode):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetRegisterDump(mode)

            Inputs:
            mode -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetRegisterDump(int mode);

        """
        cmode = c_int(mode)
        ret = self.dll.SetRegisterDump(cmode)
        return (ret)

    def SetRingExposureTimes(self, numTimes, exptimes):
        """
            Description:
            This function will send up an array of exposure times to the camera if the hardware supports the feature. See GetCapabilities. Each acquisition will then use the next exposure in the ring looping round to the start again when the end is reached. There can be a maximum of 16 exposures.

            Synopsis:
            (ret) = SetRingExposureTimes(numTimes, exptimes)

            Inputs:
            numTimes - The number of exposures
            exptimes - A predeclared pointer to an array of numTimes floats

            Outputs:
            ret - Function Return Code:
                Unsigned int - DRV_NOTAVAILABLE
                DRV_SUCCESS - Success
                DRV_NOT_INITIALIZED - System not initialized
                DRV_INVALID_MODE - This mode is not available.
                DRV_P1INVALID - Must be between 1 and 16 exposures inclusive
                DRV_P2INVALID - The exposures times are invalid.            

            C++ Equiv:
            unsigned int SetRingExposureTimes(int numTimes, float * times);

            See Also:
            GetCapabilities GetNumberRingExposureTimes GetAdjustedRingExposureTimes GetRingExposureRange IsTriggerModeAvailable

        """
        cnumTimes = c_int(numTimes)
        ctimes = (c_float * numTimes)(*exptimes)
        ret = self.dll.SetRingExposureTimes(cnumTimes, ctimes)
        return (ret)

    def SetSaturationEvent(self, event):
        """
            Description:
            This is only supported with the CCI-23 PCI card. USB cameras do not have this feature.
            This function passes a Win32 Event handle to the driver via which the driver can inform the main software that an acquisition has saturated the sensor to a potentially damaging level. You must reset the event after it has been handled in order to receive additional triggers. Before deleting the event you must call SetEvent with NULL as the parameter.

            Synopsis:
            ret = SetSaturationEvent(event)

            Inputs:
            event - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED - Function not supported for operating system

            C++ Equiv:
            unsigned int SetSaturationEvent(HANDLE saturationEvent);

            See Also:
            SetDriverEvent

            Note: The programmer must reset the event after it has been handled in order to receive additional triggers, unless the event has been created with auto-reset, e.g. event = CreateEvent(NULL, FALSE, FALSE, NULL). Also, NOT all programming environments allow the use of multiple threads and Win32 events.
                Only supported with the CCI-23 card.
                USB cameras do not have this feature.


        """
        cevent = c_int(event)
        ret = self.dll.SetSaturationEvent(cevent)
        return (ret)

    def SetSensorPortMode(self, mode):
        """
            Description:
            This function selects the sensor port mode which will be used to acquire the image data.
            “Single Port Mode” - Acquires all the image data via the selected single sensor port (SelectSensorPort).
            “All Ports Mode” - Acquires the image data simultaneously from all the available ports.

            Synopsis:
            ret = SetSensorPortMode (mode)

            Inputs:
            mode - the port mode selected. Valid values:
                0 Single Port  1 All Ports

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Port mode successfully selected
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_NOT_SUPPORTED - Feature not supported on this camera
                DRV_P1INVALID - Requested port mode isn’t valid

            C++ Equiv:
            unsigned int SetSensorPortMode (int mode);

            See Also:
            SelectSensorPort GetCapabilities

            Note: This function selects the sensor port mode which will be used to acquire the image data.
                “Single Port Mode” - Acquires all the image data via the selected single sensor port (SelectSensorPort).
                “All Ports Mode” - Acquires the image data simultaneously from all the available ports.

        """
        cmode = c_int(mode)
        ret = self.dll.SetSensorPortMode(cmode)
        return (ret)

    def SetShutter(self, typ, mode, closingtime, openingtime):
        """
            Description:
            This function controls the behaviour of the shutter.
            The typ parameter allows the user to control the TTL signal output to an external shutter. The mode parameter configures whether the shutter opens & closes automatically (controlled by the camera) or is permanently open or permanently closed.
            The opening and closing time specify the time required to open and close the shutter (this information is required for calculating acquisition timings (see SHUTTER TRANSFER TIME).

            Synopsis:
            ret = SetShutter(typ, mode, closingtime, openingtime)

            Inputs:
            typ - shutter type:
                1 - Output TTL high signal to open shutter
                0 - Output TTL low signal to open shutter
            mode - Shutter mode:
                0 - Automatic
                1 - Open
                2 - Close
            closingtime - Time shutter takes to close (milliseconds)
            openingtime - Time shutter takes to open (milliseconds)

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Shutter set.
                DRV_NOT_INITIALIZED DRV_ACQUIRING - System not initialized.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_NOT_SUPPORTED - Unable to communicate with card.
                DRV_P1INVALID - Camera does not support shutter control.
                DRV_P2INVALID - Invalid TTL type.
                DRV_P3INVALID - Invalid mode.
                DRV_P4INVALID - Invalid time to open.

            C++ Equiv:
            unsigned int SetShutter(int typ, int mode, int closingtime, int openingtime);

            See Also:
            SetShutterEx

            Note: 1. The opening and closing time can be different.
                2. For cameras capable of controlling the internal and external shutter independently (capability AC_FEATURES_SHUTTEREX) you MUST use SetShutterEx.
                3. Cameras with an internal shutter (use function IsInternalMechanicalShutter to test) but no independent shutter control (capability AC_FEATURES_SHUTTEREX) will always output a "HIGH to open" TTL signal through the external shutter port.


        """

        ctyp = c_int(typ)
        cmode = c_int(mode)
        cclosingtime = c_int(closingtime)
        copeningtime = c_int(openingtime)
        ret = self.dll.SetShutter(ctyp, cmode, cclosingtime, copeningtime)
        return (ret)

    def SetShutterEx(self, typ, mode, closingtime, openingtime, extmode):
        """
            Description:
            This function expands the control offered by SetShutter to allow an external shutter and internal shutter to be controlled independently (only available on some cameras - please consult your Camera User Guide). The typ parameter allows the user to control the TTL signal output to an external shutter. The opening and closing times specify the length of time required to open and close the shutter (this information is required for calculating acquisition timings - see SHUTTER TRANSFER TIME).
            The mode and extmode parameters control the behaviour of the internal and external shutters. To have an external shutter open and close automatically in an experiment, set the mode parameter to Open and set the extmode parameter to Auto. To have an internal shutter open and close automatically in an experiment, set the extmode parameter to Open and set the mode parameter to Auto.
            To not use any shutter in the experiment, set both shutter modes to permanently open.

            Synopsis:
            ret = SetShutterEx(typ, mode, closingtime, openingtime, extmode)

            Inputs:
            typ - Shutter type:
                0 - Output TTL low signal to open shutter
                1 - Output TTL high signal to open shutter
            mode - Internal shutter mode.:
                0 - Auto
                1 - Open
                2 - Close
            closingtime - time shutter takes to close (milliseconds)
            openingtime - Time shutter takes to open (milliseconds)
            extmode - External shutter mode.:
                0 - Auto
                1 - Open
                2 - Close

            Outputs:
            ret - Function Return Code:
                Unsigned int - DRV_P5INVALID
                DRV_SUCCESS - Shutter set.
                DRV_NOT_INITIALIZED - System not initialized
                DRV_ACQUIRING - Acquisition in progress
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_NOT_SUPPORTED - Camera does not support shutter control.
                DRV_P1INVALID - Invalid TTL type.
                DRV_P2INVALID - Invalid internal mode
                DRV_P3INVALID - Invalid time to open.
                DRV_P4INVALID - Invalid time to close

            C++ Equiv:
            unsigned int SetShutterEx(int typ, int mode, int closingtime, int openingtime, int extmode);

            See Also:
            SetShutter

            Note: 1. The opening and closing time can be different.
                2. For cameras capable of controlling the internal and external shutter independently (capability AC_FEATURES_SHUTTEREX) you MUST use SetShutterEx.
                3. Cameras with an internal shutter (use function IsInternalMechanicalShutter to test) but no independent shutter control (capability AC_FEATURES_SHUTTEREX) will always output a "HIGH to open" TTL signal through the external shutter port.

        """
        ctyp = c_int(typ)
        cmode = c_int(mode)
        cclosingtime = c_int(closingtime)
        copeningtime = c_int(openingtime)
        cextmode = c_int(extmode)
        ret = self.dll.SetShutterEx(
            ctyp, cmode, cclosingtime, copeningtime, cextmode)
        return (ret)

    def SetShutters(self, typ, mode, closingtime, openingtime, exttype, extmode, dummy1, dummy2):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetShutters(typ, mode, closingtime, openingtime, exttype, extmode, dummy1, dummy2)

            Inputs:
            typ -
            mode -
            closingtime -
            openingtime -
            exttype -
            extmode -
            dummy1 -
            dummy2 -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetShutters(int typ, int mode, int closingtime, int openingtime, int exttype, int extmode, int dummy1, int dummy2);

        """
        ctyp = c_int(typ)
        cmode = c_int(mode)
        cclosingtime = c_int(closingtime)
        copeningtime = c_int(openingtime)
        cexttype = c_int(exttype)
        cextmode = c_int(extmode)
        cdummy1 = c_int(dummy1)
        cdummy2 = c_int(dummy2)
        ret = self.dll.SetShutters(
            ctyp, cmode, cclosingtime, copeningtime, cexttype, cextmode, cdummy1, cdummy2)
        return (ret)

    def SetSifComment(self, comment):
        """
            Description:
            This function will set the user text that will be added to any sif files created with the SaveAsSif function. The stored comment can be cleared by passing NULL or an empty text string.

            Synopsis:
            ret = SetSifComment(comment)

            Inputs:
            comment - The comment to add to new sif files.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Sif comment set.

            C++ Equiv:
            unsigned int SetSifComment(char * comment);

            See Also:
            SaveAsSif SaveAsCommentedSif SaveAsSif SetReadMode

            Note: To add a comment to a SIF file that will not be used in any future SIF files that are saved, use the function SaveAsCommentedSif.

        """
        ccomment = comment
        ret = self.dll.SetSifComment(ccomment)
        return (ret)

    def SetSingleTrack(self, centre, height):
        """
            Description:
            This function will set the single track parameters. The parameters are validated in the following order: centre row and then track height.

            Synopsis:
            ret = SetSingleTrack(centre, height)

            Inputs:
            centre - centre row of track:
                Valid - range 0 to number of vertical pixels.
            height - height of track:
                Valid - range > 1 (maximum value depends on centre row and number of vertical pixels).

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Center row invalid.
                DRV_P2INVALID - Track height invalid.

            C++ Equiv:
            unsigned int SetSingleTrack(int centre, int height);

            See Also:
            SetReadMode

        """
        ccentre = c_int(centre)
        cheight = c_int(height)
        ret = self.dll.SetSingleTrack(ccentre, cheight)
        return (ret)

    def SetSingleTrackHBin(self, bin):
        """
            Description:
            This function sets the horizontal binning used when acquiring in Single Track read mode.

            Synopsis:
            ret = SetSingleTrackHBin(bin)

            Inputs:
            bin - Binning size.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Binning set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid binning size.

            C++ Equiv:
            unsigned int SetSingleTrackHBin(int bin);

            See Also:
            SetReadMode

            Note: 1) If the detector width is not a multiple of the binning DRV_BINNING_ERROR will be returned from PrepareAcquisition and/or StartAcquisition
                2) For iDus, it is recommended that you set horizontal binning to 1

        """
        cbin = c_int(bin)
        ret = self.dll.SetSingleTrackHBin(cbin)
        return (ret)

    def SetSpool(self, active, method, path, framebuffersize):
        """
            Description:
            This function will enable and disable the spooling of acquired data to the hard disk or to the RAM.
            With spooling method 0, each scan in the series will be saved to a separate file composed of a sequence of 32-bit integers.
            With spooling method 1 the type of data in the output files depends on what type of acquisition is taking place (see below).
            Spooling method 2 writes out the data to file as 16-bit integers.
            Spooling method 3 creates a directory structure for storing images where multiple images may appear in each file within the directory structure and the files may be spread across multiple directories. Like method 1 the data type of the image pixels depends on whether accumulate mode is being used.
            Method 4 Creates a RAM disk for storing images so you should ensure that there is enough free RAM to store the full acquisition.
            Methods 5, 6 and 7 can be used to directly spool out to a particular file type, either FITS, SIF or TIFF respectively. In the case of FITS and TIFF the data will be written out as 16-bit values.
            Method 8 is similar to method 3, however the data is first compressed before writing to disk. In some circumstances this may improve the maximum rate of writing images to disk, however as the compression can be very CPU intensive this option may not be suitable on slower processors.
            The data is stored in row order starting with the row nearest the readout register. With the exception of methods 5, 6 and 7, the data acquired during a spooled acquisition can be retrieved through the normal functions. This is a change to previous versions; it is no longer necessary to load the data from disk from your own application.

            Synopsis:
            ret = SetSpool(active, method, path, framebuffersize)

            Inputs:
            active - Enable/disable spooling:
                0 - Disable spooling.
                1 - Enable spooling.
            method - Indicates the format of the files written to disk:
                0 - Files contain sequence of 32-bit integers
                1 - Format of data in files depends on whether multiple accumulations are being taken for each scan. Format will be 32-bit integer if data is being accumulated each scan; otherwise the format will be 16-bit integer.
                2 - Files contain sequence of 16-bit integers.
                3 - Multiple directory structure with multiple images per file and multiple files per directory.
                4 - Spool to RAM disk.
                5 - Spool to 16-bit Fits File.
                6 - Spool to Andor Sif format.
                7 - Spool to 16-bit Tiff File.
                8 - Similar to method 3 but with data compression.
            path - String containing the filename stem. May also contain the path to the directory into which the files are to be stored.
            framebuffersize - This sets the size of an internal circular buffer used as temporary storage. The value is the total number images the buffer can hold, not the size in bytes. Typical value would be 10. This value would be increased in situations where the computer is not able to spool the data to disk at the required rate.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Parameters set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.

            C++ Equiv:
            unsigned int SetSpool(int active, int method, char * path, int framebuffersize);

            See Also:
            GetSpoolProgress

            Note: Spooled images will not be post processed, i.e. flipped or rotated.

        """

        cactive = c_int(active)
        cmethod = c_int(method)
        cpath = path.encode('utf-8')
        cframebuffersize = c_int(framebuffersize)
        ret = self.dll.SetSpool(cactive, cmethod, cpath, cframebuffersize)
        return (ret)

    def SetSpoolThreadCount(self, count):
        """
            Description:
            This function sets the number of parallel threads used for writing data to disk when spooling is enabled. Increasing this to a value greater than the default of 1, can sometimes improve the data rate to the hard disk particularly with Solid State hard disks. In other cases increasing this value may actually reduce the rate at which data is written to disk.

            Synopsis:
            ret = SetSpoolThreadCount(count)

            Inputs:
            count - The number of threads to use.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Thread count is set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid thread count.

            C++ Equiv:
            unsigned int SetSpoolThreadCount(int count);

            See Also:
            SetSpool

            Note: This feature is currently only available when using the Neo camera.

        """
        ccount = c_int(count)
        ret = self.dll.SetSpoolThreadCount(ccount)
        return (ret)

    def SetStorageMode(self, mode):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetStorageMode(mode)

            Inputs:
            mode -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetStorageMode(long mode);

        """
        cmode = c_int(mode)
        ret = self.dll.SetStorageMode(cmode)
        return (ret)

    def SetTECEvent(self, event):
        """
            Description:
            This function passes a Win32 Event handle to the driver via which the driver can inform the user software that the TEC has overheated or returned to a normal state. To determine what event has actually occurred call the GetTECStatus function. This may give the user software an opportunity to perform other actions that will not affect the readout of the current acquisition.

            Synopsis:
            ret = SetTECEvent(event)

            Inputs:
            event - Win32 event handle.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Event set
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_AVAILABLE - Function not supported for operating system

            C++ Equiv:
            unsigned int SetTECEvent(HANDLE driverEvent);

            See Also:
            GetTECStatus

        """
        cevent = c_int(event)
        ret = self.dll.SetTECEvent(cevent)
        return (ret)

    def SetTemperature(self, temperature):
        """
            Description:
            This function will set the desired temperature of the detector. To turn the cooling ON and OFF use the CoolerONCoolerON and CoolerOFFCoolerOFF function respectively.

            Synopsis:
            ret = SetTemperature(temperature)

            Inputs:
            temperature - the temperature in Centigrade.:
                Valid - range is given by GetTemperatureRangeGetTemperatureRange

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Temperature set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_P1INVALID - Temperature invalid.
                DRV_NOT_SUPPORTED - The camera does not support setting the temperature.

            C++ Equiv:
            unsigned int SetTemperature(int temperature);

            See Also:
            CoolerOFF CoolerON GetTemperature GetTemperatureF GetTemperatureRange

            Note: Not available on Luca R cameras - automatically cooled to -20C.

        """
        ctemperature = c_int(temperature)
        ret = self.dll.SetTemperature(ctemperature)
        return (ret)

    def SetTemperatureEvent(self, event):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetTemperatureEvent(event)

            Inputs:
            event -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetTemperatureEvent(at_32 temperatureEvent);

        """
        cevent = c_int(event)
        ret = self.dll.SetTemperatureEvent(cevent)
        return (ret)

    def SetTriggerInvert(self, mode):
        """
            Description:
            This function will set whether an acquisition will be triggered on a rising or falling edge external trigger.

            Synopsis:
            ret = SetTriggerInvert(mode)

            Inputs:
            mode - trigger mode:
                0 - Rising Edge
                1 - Falling Edge

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Trigger mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Trigger mode invalid.
                DRV_NOT_AVAILABLE - Feature not available.

            C++ Equiv:
            unsigned int SetTriggerInvert(int mode);

            See Also:
            Trigger Modes SetTriggerMode SetFastExtTrigger

        """
        cmode = c_int(mode)
        ret = self.dll.SetTriggerInvert(cmode)
        return (ret)

    def SetTriggerLevel(self, f_level):
        """
            Description:
            This function sets the trigger voltage which the system will use.

            Synopsis:
            ret = SetTriggerLevel(f_level)

            Inputs:
            f_level - trigger voltage

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Level set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_SUPPORTED DRV_ACQUIRING - Trigger levels not supported.
                DRV_ERROR_ACK - Acquisition in progress.
                DRV_P1INVALID - Unable to communicate with system.

            C++ Equiv:
            unsigned int SetTriggerLevel(float f_level);

            See Also:
            GetCapabilities GetTriggerLevelRange

        """
        cf_level = c_float(f_level)
        ret = self.dll.SetTriggerLevel(cf_level)
        return (ret)

    def SetTriggerMode(self, mode):
        """
            Description:
            This function will set the trigger mode that the camera will operate in.

            Synopsis:
            ret = SetTriggerMode(mode)

            Inputs:
            mode - trigger mode:
                0 - internal
                1 - External
                6 - External Start
                7 - External Exposure (Bulb)
                9 - External FVB EM (only valid for EM Newton models in FVB mode)
                10. Software Trigger
                12 - External Charge Shifting

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Trigger mode set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Trigger mode invalid.

            C++ Equiv:
            unsigned int SetTriggerMode(int mode);

            See Also:
            Trigger Modes SetFastExtTrigger

        """

        cmode = c_int(mode)
        ret = self.dll.SetTriggerMode(cmode)
        return (ret)

    def SetUserEvent(self, event):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetUserEvent(event)

            Inputs:
            event -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetUserEvent(at_32 userEvent);

        """
        cevent = c_int(event)
        ret = self.dll.SetUserEvent(cevent)
        return (ret)

    def SetUSGenomics(self, width, height):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetUSGenomics(width, height)

            Inputs:
            width -
            height -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetUSGenomics(long width, long height);

        """
        cwidth = c_int(width)
        cheight = c_int(height)
        ret = self.dll.SetUSGenomics(cwidth, cheight)
        return (ret)

    def SetVerticalRowBuffer(self, rows):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetVerticalRowBuffer(rows)

            Inputs:
            rows -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetVerticalRowBuffer(int rows);

        """
        crows = c_int(rows)
        ret = self.dll.SetVerticalRowBuffer(crows)
        return (ret)

    def SetVerticalSpeed(self, index):
        """
            Description:
            Deprecated see Note:
            This function will set the vertical speed to be used for subsequent acquisitions

            Synopsis:
            ret = SetVerticalSpeed(index)

            Inputs:
            index - index into the vertical speed table:
                0 - to GetNumberVerticalSpeedsGetNumberVerticalSpeeds-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Vertical speed set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Index out of range.

            C++ Equiv:
            unsigned int SetVerticalSpeed(int index); // deprecated

            See Also:
            GetNumberVerticalSpeeds GetVerticalSpeed

            Note: Deprecated by SetVSSpeedSetVSSpeed.

        """
        cindex = c_int(index)
        ret = self.dll.SetVerticalSpeed(cindex)
        return (ret)

    def SetVirtualChip(self, state):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = SetVirtualChip(state)

            Inputs:
            state -

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int SetVirtualChip(int state);

        """
        cstate = c_int(state)
        ret = self.dll.SetVirtualChip(cstate)
        return (ret)

    def SetVSAmplitude(self, index):
        """
            Description:
            If you choose a high readout speed (a low readout time), then you should also consider increasing the amplitude of the Vertical Clock Voltage.
            There are five levels of amplitude available for you to choose from:
            * Normal
            * +1
            * +2
            * +3
            * +4
            Exercise caution when increasing the amplitude of the vertical clock voltage, since higher clocking voltages may result in increased clock-induced charge (noise) in your signal. In general, only the very highest vertical clocking speeds are likely to benefit from an increased vertical clock voltage amplitude.

            Synopsis:
            ret = SetVSAmplitude(index)

            Inputs:
            index - desired Vertical Clock Voltage Amplitude:
                0 - Normal
                1 ->4 - Increasing Clock voltage Amplitude

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Amplitude set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NOT_AVAILABLE - Your system does not support this feature
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Invalid amplitude parameter.

            C++ Equiv:
            unsigned int SetVSAmplitude(int index);

            Note: Available in iXon, iKon and Newton - full range of amplitude levels is not available on all compatible cameras.

        """
        cindex = c_int(index)
        ret = self.dll.SetVSAmplitude(cindex)
        return (ret)

    def SetVSSpeed(self, index):
        """
            Description:
            This function will set the vertical speed to be used for subsequent acquisitions

            Synopsis:
            ret = SetVSSpeed(index)

            Inputs:
            index - index into the vertical speed table:
                0 - to GetNumberVSSpeedsGetNumberVSSpeeds-1

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Vertical speed set.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_P1INVALID - Index out of range.

            C++ Equiv:
            unsigned int SetVSSpeed(int index);

            See Also:
            GetNumberVSSpeeds GetVSSpeed GetFastestRecommendedVSSpeed

        """
        cindex = c_int(index)
        ret = self.dll.SetVSSpeed(cindex)
        return (ret)

    def ShutDown(self):
        """
            Description:
            This function will close the AndorMCD system down.

            Synopsis:
            ret = ShutDown()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - System shut down.

            C++ Equiv:
            unsigned int ShutDown(void);

            See Also:
            CoolerOFF CoolerON SetTemperature GetTemperature

            Note: 1. For Classic & ICCD systems, the temperature of the detector should be above -20C before shutting down the system.
                2. When dynamically loading a DLL which is statically linked to the SDK library, ShutDown MUST be called before unloading.

        """
        ret = self.dll.ShutDown()
        return (ret)

    def StartAcquisition(self):
        """
            Description:
            This function starts an acquisition. The status of the acquisition can be monitored via GetStatus().

            Synopsis:
            ret = StartAcquisition()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition started.
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_ACQUIRING - Acquisition in progress.
                DRV_VXDNOTINSTALLED - VxD not loaded.
                DRV_ERROR_ACK - Unable to communicate with card.
                DRV_INIERROR - Error reading DETECTOR.INI.
                DRV_ACQERROR - Acquisition settings invalid.
                DRV_ERROR_PAGELOCK - Unable to allocate memory.
                DRV_INVALID_FILTER - Filter not available for current acquisition.
                DRV_BINNING_ERROR - Range not multiple of horizontal binning.
                DRV_SPOOLSETUPERROR - Error with spool settings.

            C++ Equiv:
            unsigned int StartAcquisition(void);

            See Also:
            GetStatus GetAcquisitionTimings SetAccumulationCycleTime SetAcquisitionMode SetExposureTime SetHSSpeed SetKineticCycleTime SetMultiTrack SetNumberAccumulations SetNumberKinetics SetReadMode SetSingleTrack SetTriggerMode SetVSSpeed

        """
        ret = self.dll.StartAcquisition()
        return (ret)

    def UnMapPhysicalAddress(self):
        """
            Description:
            THIS FUNCTION IS RESERVED.

            Synopsis:
            ret = UnMapPhysicalAddress()

            Inputs:
            None

            Outputs:
            ret - Function Return Code

            C++ Equiv:
            unsigned int UnMapPhysicalAddress(void);

        """
        ret = self.dll.UnMapPhysicalAddress()
        return (ret)

    def UpdateDDGTimings(self):
        """
            Description:
            This function can be used to update the gate timings and external output timings during an externally triggered kinetic series. It is only available when integrate on chip is active and gate step is not being used.

            Synopsis:
            ret = UpdateDDGTimings()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Timings updated
                DRV_NOT_INITIALIZED - System not initialized
                DRV_NOT_AVAILABLE - Camera not in correct mode
                DRV_NOT_SUPPORTED - Feature not supported

            C++ Equiv:
            unsigned int UpdateDDGTimings(void);

            See Also:
            GetCapabilities SetDDGExternalOutputTime SetDDGGateTime

            Note: This function can be used to update the gate timings and external output timings during an externally triggered kinetic series. It is only available when integrate on chip is active and gate step is not being used.

        """
        ret = self.dll.UpdateDDGTimings()
        return (ret)

    def WaitForAcquisition(self):
        """
            Description:
            WaitForAcquisition can be called after an acquisition is started using StartAcquisitionStartAcquisition to put the calling thread to sleep until an Acquisition Event occurs. This can be used as a simple alternative to the functionality provided by the SetDriverEvent function, as all Event creation and handling is performed internally by the SDK library.
            Like the SetDriverEvent functionality it will use less processor resources than continuously polling with the GetStatus function. If you wish to restart the calling thread without waiting for an Acquisition event, call the function CancelWaitCancelWait.
            An Acquisition Event occurs each time a new image is acquired during an Accumulation, Kinetic Series or Run-Till-Abort acquisition or at the end of a Single Scan Acquisition.
            If a second event occurs before the first one has been acknowledged, the first one will be ignored. Care should be taken in this case, as you may have to use CancelWaitCancelWait to exit the function.

            Synopsis:
            ret = WaitForAcquisition()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition Event occurred
                DRV_NOT_INITIALIZED - System not initialized.
                DRV_NO_NEW_DATA - Non-Acquisition Event occurred.(e.g. CancelWait () called)

            C++ Equiv:
            unsigned int WaitForAcquisition(void);

            See Also:
            StartAcquisition CancelWait

        """
        ret = self.dll.WaitForAcquisition()
        return (ret)

    def WaitForAcquisitionByHandle(self, cameraHandle):
        """
            Description:
            Whilst using multiple cameras WaitForAcquisitionByHandle can be called after an acquisition is started using StartAcquisition to put the calling thread to sleep until an Acquisition Event occurs. This can be used as a simple alternative to the functionality provided by the SetDriverEvent function, as all Event creation and handling is performed internally by the SDK library. Like the SetDriverEvent functionality it will use less processor resources than continuously polling with the GetStatus function. If you wish to restart the calling thread without waiting for an Acquisition event, call the function CancelWait. An Acquisition Event occurs each time a new image is acquired during an Accumulation, Kinetic Series or Run-Till-Abort acquisition or at the end of a Single Scan Acquisition.

            Synopsis:
            ret = WaitForAcquisitionByHandle(cameraHandle)

            Inputs:
            cameraHandle - handle of camera to put into wait state.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition Event occurred.
                DRV_P1INVALID - Handle not valid.
                DRV_NO_NEW_DATA - Non-Acquisition Event occurred.(eg CancelWait () called)

            C++ Equiv:
            unsigned int WaitForAcquisitionByHandle(long cameraHandle);

            See Also:
            CancelWait GetCameraHandle StartAcquisition WaitForAcquisition WaitForAcquisitionTimeOut WaitForAcquisitionByHandleTimeOut

        """
        ccameraHandle = c_int(cameraHandle)
        ret = self.dll.WaitForAcquisitionByHandle(ccameraHandle)
        return (ret)

    def WaitForAcquisitionByHandleTimeOut(self, cameraHandle, iTimeOutMs):
        """
            Description:
            Whilst using multiple cameras WaitForAcquisitionByHandle can be called after an acquisition is started using StartAcquisition to put the calling thread to sleep until an Acquisition Event occurs. This can be used as a simple alternative to the functionality provided by the SetDriverEvent function, as all Event creation and handling is performed internally by the SDK library. Like the SetDriverEvent functionality it will use less processor resources than continuously polling with the GetStatus function. If you wish to restart the calling thread without waiting for an Acquisition event, call the function CancelWait. An Acquisition Event occurs each time a new image is acquired during an Accumulation, Kinetic Series or Run-Till-Abort acquisition or at the end of a Single Scan Acquisition. If an Acquisition Event does not occur within _TimeOutMs milliseconds, WaitForAcquisitionTimeOut returns DRV_NO_NEW_DATA

            Synopsis:
            ret = WaitForAcquisitionByHandleTimeOut(cameraHandle, iTimeOutMs)

            Inputs:
            cameraHandle - handle of camera to put into wait state.
            iTimeOutMs - Time before returning DRV_NO_NEW_DATA if no Acquisition Event occurs.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition Event occurred.
                DRV_P1INVALID - Handle not valid.
                DRV_NO_NEW_DATA - Non-Acquisition Event occurred.(eg CancelWait () called, time out)

            C++ Equiv:
            unsigned int WaitForAcquisitionByHandleTimeOut(long cameraHandle, int iTimeOutMs);

            See Also:
            CancelWait GetCameraHandle StartAcquisition WaitForAcquisition WaitForAcquisitionByHandle WaitForAcquisitionTimeOut

        """
        ccameraHandle = c_int(cameraHandle)
        ciTimeOutMs = c_int(iTimeOutMs)
        ret = self.dll.WaitForAcquisitionByHandleTimeOut(
            ccameraHandle, ciTimeOutMs)
        return (ret)

    def WaitForAcquisitionTimeOut(self, iTimeOutMs):
        """
            Description:
            WaitForAcquisitionTimeOut can be called after an acquisition is started using StartAcquisition to put the calling thread to sleep until an Acquisition Event occurs. This can be used as a simple alternative to the functionality provided by the SetDriverEvent function, as all Event creation and handling is performed internally by the SDK library. Like the SetDriverEvent functionality it will use less processor resources than continuously polling with the GetStatus function. If you wish to restart the calling thread without waiting for an Acquisition event, call the function CancelWait. An Acquisition Event occurs each time a new image is acquired during an Accumulation, Kinetic Series or Run-Till-Abort acquisition or at the end of a Single Scan Acquisition. If an Acquisition Event does not occur within _TimeOutMs milliseconds, WaitForAcquisitionTimeOut returns DRV_NO_NEW_DATA

            Synopsis:
            ret = WaitForAcquisitionTimeOut(iTimeOutMs)

            Inputs:
            iTimeOutMs - Time before returning DRV_NO_NEW_DATA if no Acquisition Event occurs.

            Outputs:
            ret - Function Return Code:
                DRV_SUCCESS - Acquisition Event occurred.
                DRV_NO_NEW_DATA - Non-Acquisition Event occurred.(eg CancelWait () called, time out)

            C++ Equiv:
            unsigned int WaitForAcquisitionTimeOut(int iTimeOutMs);

            See Also:
            CancelWait StartAcquisition WaitForAcquisition WaitForAcquisitionByHandle WaitForAcquisitionByHandleTimeOut

        """
        ciTimeOutMs = c_int(iTimeOutMs)
        ret = self.dll.WaitForAcquisitionTimeOut(ciTimeOutMs)
        return (ret)

    def WhiteBalance(self):
        """
            Description:
            For colour sensors only
            Calculates the red and blue relative to green factors to white balance a colour image using the parameters stored in info.
            Before passing the address of an WhiteBalanceInfo structure to the function the iSize member of the structure should be set to the size of the structure. In C++ this can be done with the line:
            nfo-> iSize = sizeof(WhiteBalanceInfo);
            Below is the WhiteBalanceInfo structure definition and a description of its members:
            struct WHITEBALANCEINFO {
            int iSize;  // Structure size.
            int iX;      // Number of X pixels. Must be >2.
            int iY;     // Number of Y pixels. Must be >2.
            int iAlgorithm; // Algorithm to used to calculate white balance.
            int iROI_left; // Region Of interest from which white balance is calculated
            int iROI_right; // Region Of interest from which white balance is calculated
            int iROI_top; // Region Of interest from which white balance is calculated
            int iROI_bottom; // Region Of interest from which white balance is calculated
            WhiteBalanceInfo;
            iX and iY are the image dimensions. The number of elements of the input, red, green and blue arrays are iX x iY.
            iAlgorithm sets the algorithm to use. The function sums all the colour values per each colour field within the Region Of interest (ROI) and calculates the relative to green values as: 0) _fRelR = GreenSum / RedSum and _fRelB = GreenSum / BlueSum; 1) _fRelR = 2/3 GreenSum / RedSum and _fRelB = 2/3 GreenSum / BlueSum, giving more importance to the green field.
            iROI_left, iROI_right, iROI_top and iROI_bottom define the ROI with the constraints:
            iROI_left0 <= iROI_left < iROI_right <= iX and 0 <= iROI_ bottom < iROI_ top <= iX

            Synopsis:
            (ret, wRed, wGreen, wBlue, fRelR, fRelB, info) = WhiteBalance()

            Inputs:
            None

            Outputs:
            ret - Function Return Code:
                SUCCESS - White balance calculated.
                DRV_P1INVALID - Invalid pointer (i.e. NULL).
                DRV_P2INVALID - Invalid pointer (i.e. NULL).
                DRV_P3INVALID - Invalid pointer (i.e. NULL).
                DRV_P4INVALID - Invalid pointer (i.e. NULL).
                DRV_P5INVALID - Invalid pointer (i.e. NULL).
                DRV_P6INVALID - One or more parameters in info is out of range
                DRV_DIVIDE_BY_ZERO_ERROR - The sum of the green field within the ROI is zero. _fRelR and _fRelB are set to 1
            wRed - pointer to red field.
            wGreen - pointer to green field.
            wBlue - pointer to blue field.
            fRelR - pointer to the relative to green red factor.
            fRelB - pointer to the relative to green blue factor.
            info - pointer to white balance information structure

            C++ Equiv:
            unsigned int WhiteBalance(WORD * wRed, WORD * wGreen, WORD * wBlue, float * fRelR, float * fRelB, WhiteBalanceInfo * info);

            See Also:
            DemosaicImage GetMostRecentColorImage16

        """
        cwRed = c_short()
        cwGreen = c_short()
        cwBlue = c_short()
        cfRelR = c_float()
        cfRelB = c_float()
        cinfo = WhiteBalanceInfo()
        ret = self.dll.WhiteBalance(byref(cwRed), byref(cwGreen), byref(
            cwBlue), byref(cfRelR), byref(cfRelB), byref(cinfo))
        return (ret, cwRed.value, cwGreen.value, cwBlue.value, cfRelR.value, cfRelB.value, cinfo)
