#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-03
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
        # ch:初始化相机枚举相机个数 | en:Initializing Cameras Enumerates the number of cameras
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        print(self.TUCAMINIT.uiCamCount)
        print(self.TUCAMINIT.pstrConfigPath)
        print('Connect %d camera' %self.TUCAMINIT.uiCamCount)

    def OpenCamera(self, Idx):

        if  Idx >= self.TUCAMINIT.uiCamCount:
            return

        self.TUCAMOPEN = TUCAM_OPEN(Idx, 0)

        # ch:打开相机Idx | en:Open camera Idx
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))

        if 0 == self.TUCAMOPEN.hIdxTUCam:
            print('Open the camera failure!')
            return
        else:
            print('Open the camera success!')

    def CloseCamera(self):
        # ch:关闭相机 | en:Close camera
        if 0 != self.TUCAMOPEN.hIdxTUCam:
            TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        print('Close the camera success')

    def UnInitApi(self):
        # ch:反初始化相机 | en:Uninitial Cameras
        TUCAM_Api_Uninit()

    def SetROI(self):
        roi = TUCAM_ROI_ATTR()
        roi.bEnable  = 1
        roi.nHOffset = 0
        roi.nVOffset = 0
        roi.nWidth   = 320
        roi.nHeight  = 240

        try:
            # ch:设置相机感兴趣区域 | en:Set ROI
           TUCAM_Cap_SetROI(self.TUCAMOPEN.hIdxTUCam, roi)
           print('Set ROI state success, HOffset:%#d, VOffset:%#d, Width:%#d, Height:%#d'%(roi.nHOffset,
                    roi.nVOffset, roi.nWidth, roi.nHeight))
        except Exception:
            print('Set ROI state failure, HOffset:%#d, VOffset:%#d, Width:%#d, Height:%#d' % (roi.nHOffset,
                    roi.nVOffset, roi.nWidth,roi.nHeight))

    # ch:获取相机数据流 | en:Get camera stream
    def WaitForImageData(self):
        m_frame = TUCAM_FRAME()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0;
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)

        nTimes = 10
        for i in range(nTimes):
            try:
                result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
                print(
                    "Grab the frame success, index number is %#d, width:%d, height:%#d, channel:%#d, elembytes:%#d, image size:%#d"%(i, m_frame.usWidth, m_frame.usHeight, m_frame.ucChannels,
                    m_frame.ucElemBytes, m_frame.uiImgSize)
                    )
            except Exception:
                print('Grab the frame failure, index number is %#d',  i)
                continue

        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.SetROI()
        demo.WaitForImageData()
        demo.CloseCamera()
    demo.UnInitApi()