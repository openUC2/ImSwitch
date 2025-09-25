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
            # print('TUCAM_Buf_GetData = ', Result)
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
            # print(m_rawHeader.uiWidth)
            # print(m_rawHeader.uiHeight)
        except Exception:
            print('except')

class Tucam():
    def __init__(self):
        self.Path = './'
        TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        TUCAM_Api_Init(pointer(TUCAMINIT), 5000)
        print(TUCAMINIT.uiCamCount)
        print(TUCAMINIT.pstrConfigPath)
        self.count = TUCAMINIT.uiCamCount
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)

        if TUCAMINIT.uiCamCount == 0:
            print('No Camera found!')
            return

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

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)
        TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
        time.sleep(1)
        #print(m_frame.pBuffer)
        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

    # ch:获取Genicam相机参数列表信息 | en:Get genicam camera attribute information list
    def getallelementattr(self):

        s_access = ["NI","NA", "WO", "RO","RW" ]

        s_elemType = [
            "Value",
            "Base",
            "Integer",
            "Boolean",
            "Command",
            "Float",
            "String",
            "Register",
            "Category",
            "Enumeration",
            "EnumEntry",
            "Port"]

        elemtype = TUELEM_TYPE
        result = TUCAMRET
        level = c_int(0)
        node = TUCAM_ELEMENT()
        node.pName = c_char_p(b"Root")
        m_xmldevice = TUXML_DEVICE
        print('Get element attribute list:')

        try:
            while(result.TUCAMRET_SUCCESS.value == TUCAM_GenICam_ElementAttrNext(self.TUCAMOPEN.hIdxTUCam, pointer(node), node.pName, m_xmldevice.TU_CAMERA_XML.value)):
                if elemtype.TU_ElemCategory.value == node.Type:
                    level = node.Level
                    print('%#d [%#d] %#s' %(level, node.Level, node.pName))

                if elemtype.TU_ElemBoolean.value == node.Type:
                    print('%#d [%#d] [%#s][%#s] %#s, %d' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName, node.uValue.Int64.nVal))

                if elemtype.TU_ElemCommand.value == node.Type:
                    print('%#d [%#d] [%#s][%#s] %#s, %d' % (level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName,node.uValue.Int64.nVal))

                if elemtype.TU_ElemInteger.value == node.Type:
                    print('%#d [%#d] [%#s][%#s] %#s, %d' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName, node.uValue.Int64.nVal))

                if elemtype.TU_ElemFloat.value == node.Type:
                    print('%#d [%#d] [%#s][%#s] %#s, %#lf' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName, node.uValue.Double.dbVal))

                if elemtype.TU_ElemString.value == node.Type:
                    buf = create_string_buffer(node.uValue.Int64.nMax + 1)
                    memset(buf, 0, (node.uValue.Int64.nMax + 1))
                    node.pTransfer = cast(buf, c_char_p)

                    TUCAM_GenICam_GetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node), m_xmldevice.TU_CAMERA_XML.value)
                    print('%#d [%#d] [%#s][%#s]%#s, %#s' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName, node.pTransfer))

                if elemtype.TU_ElemEnumeration.value == node.Type:
                    strlist = ctypes.cast(node.pEntries, ctypes.POINTER(ctypes.c_char_p))
                    print('%#d [%#d] [%#s][%#s]%#s, %#s' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName, strlist[node.uValue.Int64.nVal]))
                    # cnt = node.uValue.Int64.nMax - node.uValue.Int64.nMin + 1
                    # for i in range(cnt):
                    #    #str = ctypes.cast(node.pEntries, ctypes.POINTER(ctypes.c_char_p)).contents.value
                    #    strlist = ctypes.cast(node.pEntries, ctypes.POINTER(ctypes.c_char_p))
                    #    print('%#d [%#d] [%#s][%#s]%#s, %#s' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName, strlist[i]))

                if elemtype.TU_ElemRegister.value == node.Type:
                   print('%#d [%#d] [%#s][%#s]%#s' %(level, node.Level, s_access[node.Access], s_elemType[node.Type], node.pName))

        except Exception:
            print('Get element attribute list finish')


    def closecamera(self):
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
        demo.getallelementattr()
        demo.closecamera()