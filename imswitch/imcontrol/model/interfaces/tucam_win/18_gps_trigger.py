#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-11
@author:fdy
'''

import ctypes
from ctypes import *
from TUCam import *
from enum import Enum
import time

class Tucam():
    def __init__(self):
        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        print(self.TUCAMINIT.uiCamCount)
        print(self.TUCAMINIT.pstrConfigPath)
        print('Connect %d camera' %self.TUCAMINIT.uiCamCount)

    def OpenCamera(self, Idx):

        if  Idx >= self.TUCAMINIT.uiCamCount:
            return

        self.TUCAMOPEN = TUCAM_OPEN(Idx, 0)

        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))

        if 0 == self.TUCAMOPEN.hIdxTUCam:
            print('Open the camera failure!')
            return
        else:
            print('Open the camera success!')

    def CloseCamera(self):
        if 0 != self.TUCAMOPEN.hIdxTUCam:
            TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        print('Close the camera success')

    def UnInitApi(self):
        TUCAM_Api_Uninit()

    def DoGPSTrigger(self):
        m_tgr = TUCAM_TRIGGER_ATTR()
        m_frame = TUCAM_FRAME()
        m_fra = TUCAM_IMG_HEADER()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        # ch:设置相机GPS触发模式 | en:Set gps trigger mode
        TUCAM_Cap_GetTrigger(self.TUCAMOPEN.hIdxTUCam, pointer(m_tgr))
        m_tgr.nTgrMode = m_capmode.TUCCM_TRIGGER_GPS.value
        m_tgr.nBufFrames = 1
        TUCAM_Cap_SetTrigger(self.TUCAMOPEN.hIdxTUCam, m_tgr)

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        # Set GPS parameter
        # GPS Start Time dwTime = (m_nHour << 16) + (m_nMin << 8) + m_nSec;
        TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDPROP.TUIDP_START_TIME.value, 0);
        # GPS Frame Number
        TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDPROP.TUIDP_FRAME_NUMBER.value, 1);
        # GPS Interval Time
        TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDPROP.TUIDP_INTERVAL_TIME.value, 1);

        # Get GPS parameter
        valinfo =  TUCAM_VALUE_INFO()
        valinfo.nID = TUCAM_IDINFO.TUIDI_UTCTIME.value
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(valinfo))
        print('UTC Reference Time %#s', valinfo.pText)

        valinfo.nID = TUCAM_IDINFO.TUIDI_UTCTIME.value
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(valinfo))
        print('UTC pos %#s', valinfo.pText)

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_tgr.nTgrMode)

        nTimes = 10
        for i in range(nTimes):
            try:
                # ch:获取数据打印GPS数据 | en:Get stream print gps data
                result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
                pointer_data = c_void_p(m_frame.pBuffer)
                memmove(pointer(m_fra), pointer_data, m_frame.usHeader)
                print( "Get the gps parameter success, index number is %#d, Year:%#d, Month:%#d, Day:%#d, Hour:%#d, Min:%#d, Sec:%#d, Ns:%#d" %(i, m_frame.ucGPSTimeStampYear, m_frame.ucGPSTimeStampMonth,
                    m_frame.ucGPSTimeStampDay, m_frame.ucGPSTimeStampHour, m_frame.ucGPSTimeStampMin, m_frame.ucGPSTimeStampSec , m_frame.nGPSTimeStampNs))
            except Exception:
                print('Get the gps parameter failure, index number is %#d'%i)
                continue

        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.DoGPSTrigger()
        demo.CloseCamera()
    demo.UnInitApi()