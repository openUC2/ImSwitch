#!/usr/bin/env python
# coding: utf-8
'''
Created on 2023-12-11
@author:fdy
'''

import ctypes
from ctypes import *
from TUCam import *
from enum import Enum
import time

# the call back fuction
class CallBack():
    def __init__(self, TUCAMOPEN_PARA):
        self.TUCAMOPEN = TUCAMOPEN_PARA
    def OnCallbackBuffer(self):
        m_rawHeader = TUCAM_RAWIMG_HEADER()
        print('TUCAM_Buf_GetData')
        try:
            # ch:回调获取数据流 | en:Callback get stream
            Result = TUCAM_Buf_GetData(self.TUCAMOPEN.hIdxTUCam, pointer(m_rawHeader))
            print('TUCAM_Buf_GetData = ', Result)
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
        TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        # ch:初始化相机枚举相机个数 | en:Initializing Cameras Enumerates the number of cameras
        TUCAM_Api_Init(pointer(TUCAMINIT), 5000)
        print(TUCAMINIT.uiCamCount)
        print(TUCAMINIT.pstrConfigPath)
        self.count = TUCAMINIT.uiCamCount
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)

        if TUCAMINIT.uiCamCount == 0:
            print('No Camera found!')
            return

        # ch:打开相机 | en:Open camera
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))  # TODO need camera connected
        print(self.TUCAMOPEN.uiIdxOpen)
        print(self.TUCAMOPEN.hIdxTUCam)

        if self.TUCAMOPEN.hIdxTUCam == 0:
            print('Open Camera fail!')
            return

    def startcapture(self):
        m_frame = TUCAM_FRAME()
        m_frformat= TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer     = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_RAW.value
        m_frame.uiRsdSize   = 1
        #print(m_frame.pBuffer)
        #print(m_frame.ucFormatGet)
        # ch:分配内存 | en:Alloc buffer
        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        # ch:开始采集 | en:Start capture
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)
        # ch:等待数据帧 | en:Wait for frames
        TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
        time.sleep(5)
        #print(m_frame.pBuffer)
        # ch:跳出等待数据帧 | en:Abort wait for frames
        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        # ch:停止采集 | en:Stop capture
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        # ch:释放内存 | en:Release buffer
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

    def closecamera(self):
        # ch:关闭相机 | en:Close camera
        TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Api_Uninit()

if __name__ == '__main__':
    demo = Tucam()

    if demo.TUCAMOPEN.hIdxTUCam != 0:
        m_callback = CallBack(demo.TUCAMOPEN)
        CALL_BACK_FUN = BUFFER_CALLBACK(m_callback.OnCallbackBuffer)
        CALL_BACK_USER = CONTEXT_CALLBACK(m_callback.__class__)
        # ch:注册回调函数 | en:Register callback functions
        TUCAM_Buf_DataCallBack(demo.TUCAMOPEN.hIdxTUCam, CALL_BACK_FUN, CALL_BACK_USER)
        demo.startcapture()
        demo.closecamera()