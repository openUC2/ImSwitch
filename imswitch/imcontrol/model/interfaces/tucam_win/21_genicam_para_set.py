#!/usr/bin/env python
# coding: utf-8
'''
Created on 2023-12-12
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

    # ch:设置Genicam相机参数列表信息 | en:Set genicam camera attribute information list
    def setpropertyvalue(self, name, type, value):

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
        node = TUCAM_ELEMENT()
        node.pName = name
        m_xmldevice = TUXML_DEVICE
        print('Set property value:')
        TUCAM_GenICam_ElementAttr(self.TUCAMOPEN.hIdxTUCam, pointer(node), node.pName, m_xmldevice.TU_CAMERA_XML.value)

        try:
            if elemtype.TU_ElemBoolean.value == type:
                node.uValue.Int64.nVal = value
                TUCAM_GenICam_SetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node),
                                                    m_xmldevice.TU_CAMERA_XML.value)

                print('[%#s] Set %#s value is %#d' % (s_elemType[node.Type], node.pName, node.uValue.Int64.nVal))

            if elemtype.TU_ElemCommand.value == type:
                node.uValue.Int64.nVal = value
                TUCAM_GenICam_SetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node),
                                                    m_xmldevice.TU_CAMERA_XML.value)

                print('[%#s] Set %#s value is %#d' % (s_elemType[node.Type], node.pName, node.uValue.Int64.nVal))

            if elemtype.TU_ElemInteger.value == type:
                node.uValue.Int64.nVal = value
                TUCAM_GenICam_SetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node),
                                                    m_xmldevice.TU_CAMERA_XML.value)

                print('[%#s] Set %#s value is %#d' % (s_elemType[node.Type], node.pName, node.uValue.Int64.nVal))

            if elemtype.TU_ElemFloat.value == type:
                node.uValue.Double.dbVal = value
                TUCAM_GenICam_SetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node),
                                                    m_xmldevice.TU_CAMERA_XML.value)

                print('[%#s] Set %#s value is %#lf' % (s_elemType[node.Type], node.pName, node.uValue.Double.dbVal))

            if elemtype.TU_ElemString.value == type or elemtype.TU_ElemRegister.value == type:
                node.pTransfer = value
                TUCAM_GenICam_SetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node), m_xmldevice.TU_CAMERA_XML.value)
                print('[%#s] Set %#s value is %#s' % (s_elemType[node.Type], node.pName, node.pTransfer))
                buf = create_string_buffer(node.uValue.Int64.nMax + 1)
                memset(buf, 0, (node.uValue.Int64.nMax + 1))
                node.pTransfer = cast(buf, c_char_p)
                TUCAM_GenICam_GetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node), m_xmldevice.TU_CAMERA_XML.value)
                print('[%#s] Get %#s value is %#s' %(s_elemType[node.Type], node.pName, node.pTransfer))

            if elemtype.TU_ElemEnumeration.value == type:
                node.uValue.Int64.nVal = max(0, value)
                node.uValue.Int64.nVal = min(node.uValue.Int64.nVal, node.uValue.Int64.nMax)
                TUCAM_GenICam_SetElementValue(self.TUCAMOPEN.hIdxTUCam, pointer(node), m_xmldevice.TU_CAMERA_XML.value)
                strlist = ctypes.cast(node.pEntries, ctypes.POINTER(ctypes.c_char_p))
                print('[%#s] Set %#s value is %#s ' %(s_elemType[node.Type], node.pName, strlist[node.uValue.Int64.nVal]))

        except Exception:
            print('Set property value fail')


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
        elemtype = TUELEM_TYPE

        # String or Register ch:设置用户ID | en:Set user id
        demo.setpropertyvalue(b"DeviceUserID", elemtype.TU_ElemString.value, b"Dhyana 2100")

        # Enumeration ch:设置模拟增益 | en:Set analog gain
        demo.setpropertyvalue(b"AnalogGain", elemtype.TU_ElemEnumeration.value, 1)

        # Float ch:设置曝光时间 | en:Set ExposureTime
        demo.setpropertyvalue(b"ExposureTime", elemtype.TU_ElemFloat.value, 5.0)

        # Integer ch:设置黑电平 | en:Set BlackLevel
        demo.setpropertyvalue(b"BlackLevel", elemtype.TU_ElemInteger.value, 100)

        # Bool ch:设置水平镜像 | en:Set ReverseX
        demo.setpropertyvalue(b"ReverseX", elemtype.TU_ElemBoolean.value, 1)

        demo.closecamera()