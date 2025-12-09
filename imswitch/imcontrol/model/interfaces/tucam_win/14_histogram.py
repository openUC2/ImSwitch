#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-09
@author:fdy
'''

import ctypes
from ctypes import *
from TUCam import *
from enum import Enum
import time
import struct

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

    # ch:获取直方图数据 | en:Get histogram data
    def GetHistogramData(self):
        m_frame = TUCAM_FRAME()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)
        print('Set histogram enable')
        TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_HISTC.value, c_int32(1))

        try:
            result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
            print("Grab the frame success, index number is %#d, width:%d, height:%#d, channel:%#d, elembytes:%#d, image size:%#d" %(m_frame.uiIndex, m_frame.usWidth, m_frame.usHeight, m_frame.ucChannels,
            m_frame.ucElemBytes, m_frame.uiImgSize))

            offset = m_frame.uiImgSize + m_frame.usHeader
            hislen = 65536*4  # int32 65536
            chnn   = 4

            hischannel = 0   # color 0:Y 1:R 2:G 3:B
            hissize    = hislen

            if 1 == m_frame.ucElemBytes:
                hissize = 256*4 # 256 Gray int32

            #print(m_frame.ucChannels)
            his = create_string_buffer(hissize)
            if 1 == m_frame.ucChannels:  # mono  0:Y
                buf = (c_uint8 * (offset + hislen)).from_address(m_frame.pBuffer)

                for i in range(hissize):
                    his[i] = buf[i+offset]
            else:                        # color 0:Y 1:R 2:G 3:B
                print('Get channel = %#d histogram' %hischannel)
                TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_CHANNELS.value, hischannel)
                buf = (c_uint8 * (offset + hislen*chnn)).from_address(m_frame.pBuffer)

                for i in range(hissize):
                    his[i] = buf[i + offset + hislen*hischannel]

            # for byte in bytes(his):
            #     print(byte)
            # test_path = "his.raw"
            # with open(test_path, 'wb') as f:
            #     f.write(bytes(his))
            #     f.close()

        except Exception:
            print('Grab the frame failure')

        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

    def closecamera(self):
        TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Api_Uninit()

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.GetHistogramData()
        demo.CloseCamera()
    demo.UnInitApi()