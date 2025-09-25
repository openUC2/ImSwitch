#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-04-02
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

    # ch:获取黑白帧直方图数据 | en:Get mono frame histogram data
    def CalculateMonoHistogramData(self, m_frame):

        buf = create_string_buffer(m_frame.uiImgSize)
        pointer_data = c_void_p(m_frame.pBuffer + m_frame.usHeader)
        memmove(buf, pointer_data, m_frame.uiImgSize)

        size = m_frame.usWidth * m_frame.usHeight
        data = [0] * 65536
        buffers = bytes(buf)
        count = 65536

        if 1 == m_frame.ucElemBytes:
            count = 256

        for i in range(size):
            if 1 == m_frame.ucElemBytes:
                value = buffers[i]
                data[value] += 1
            else:
                value = buffers[i*2] + buffers[i*2+1] * 256
                data[value] += 1

        for j in range(count):
            print("gray :", j, "count :", data[j])

    # ch:获取彩色帧直方图数据 | en:Get color frame histogram data
    def CalculateColorHistogramData(self, m_frame):

        buf = create_string_buffer(m_frame.uiImgSize)
        pointer_data = c_void_p(m_frame.pBuffer + m_frame.usHeader)
        memmove(buf, pointer_data, m_frame.uiImgSize)

        size = m_frame.usWidth * m_frame.usHeight
        data = [0] * 65536
        buffers = bytes(buf)
        count = 65536

        if 1 == m_frame.ucElemBytes:
            count = 256

        for i in range(size):
            if 1 == m_frame.ucElemBytes:
                value = buffers[i*3] * 0.114 + buffers[i*3+1] * 0.587 + buffers[i*3+2] * 0.299
                data[value] += 1
            else:
                value = (buffers[i*6] + buffers[i*6+1] * 256) * 0.114 + (buffers[i*6+2] + buffers[i*6+3] * 256) * 0.587 + (buffers[i*6+4] + buffers[i*6+5] * 256) * 0.299
                data[value] += 1

        for j in range(count):
            print("gray :", j, "count :", data[j])

    # ch:获取帧直方图数据 | en:Get frame histogram data
    def CalculateHistogramData(self, m_frame):

        if 1 == m_frame.ucChannels:
            self.CalculateMonoHistogramData(m_frame)
        else:
            self.CalculateColorHistogramData(m_frame)

    def TestCalculateHistogramData(self):
        m_frame = TUCAM_FRAME()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, 1)
        time.sleep(3.0)

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
                self.CalculateHistogramData(m_frame)
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
        demo.TestCalculateHistogramData()
        demo.CloseCamera()
    demo.UnInitApi()