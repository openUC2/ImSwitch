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

# the call back function
class CallBack():
    def __init__(self, TUCAMOPEN_PARA):
        self.TUCAMOPEN = TUCAMOPEN_PARA
    def OnCallbackBuffer(self):
        m_rawHeader = TUCAM_RAWIMG_HEADER()
        print('TUCAM_Buf_GetData')
        try:
            # ch:回调获取数据流打印数据戳数据 | en:Callback get stream print timestamp data
            Result = TUCAM_Buf_GetData(self.TUCAMOPEN.hIdxTUCam, pointer(m_rawHeader))
            print("TUCAM_Buf_GetData index number is %#d, Time_start:%lf, Time_last:%#lf" %(
            m_rawHeader.uiIndex, m_rawHeader.dblTimeStamp, m_rawHeader.dblTimeLast))
            # buf = create_string_buffer(m_rawHeader.uiImgSize)
            # pointer_data = c_void_p(m_rawHeader.pImgData)
            # memmove(buf, pointer_data, m_rawHeader.uiImgSize)
            #
            # test_path = "result.raw"
            # with open(test_path, 'wb') as f:
            #     f.write(bytes(buf))
            #     f.close()
            # print(m_rawHeader.uiIndex)
            # print(m_rawHeader.uiImgSize)
            # print(m_rawHeader.usWidth)
            # print(m_rawHeader.usHeight)
        except Exception:
            print('except')

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

    def WaitForImageData(self):
        m_frame = TUCAM_FRAME()
        m_fra   = TUCAM_IMG_HEADER()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)

        nTimes = 10
        for i in range(nTimes):
            try:
                result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
                pointer_data = c_void_p(m_frame.pBuffer)
                memmove(pointer(m_fra), pointer_data, m_frame.usHeader)
                #print("Grab the frame success, index number is %#d, Time_start:%lf, Time_last:%#lf" %(m_fra.uiIndex, m_fra.dblTimeStamp, m_fra.dblTimeLast))
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
        m_callback = CallBack(demo.TUCAMOPEN)
        CALL_BACK_FUN = BUFFER_CALLBACK(m_callback.OnCallbackBuffer)
        CALL_BACK_USER = CONTEXT_CALLBACK(m_callback.__class__)
        TUCAM_Buf_DataCallBack(demo.TUCAMOPEN.hIdxTUCam, CALL_BACK_FUN, CALL_BACK_USER)
        demo.WaitForImageData()
        demo.CloseCamera()
    demo.UnInitApi()