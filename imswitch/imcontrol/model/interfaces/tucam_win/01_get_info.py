#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-02
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
        print('Connect %d camera' % self.TUCAMINIT.uiCamCount)

    def OpenCamera(self, Idx):

        if Idx >= self.TUCAMINIT.uiCamCount:
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

    # Get Camera Info
    def PrintCameraInfo(self):
        # Camera name: ch:获取相机名称 | en:Get camera name
        m_infoid = TUCAM_IDINFO
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_CAMERA_MODEL.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        print('Camera Name:%#s' %TUCAMVALUEINFO.pText)

        # Camera VID
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_VENDOR.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        print('Camera  VID:%#X' % TUCAMVALUEINFO.nValue)

        # Camera PID
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_PRODUCT.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        print('Camera  PID:%#X' % TUCAMVALUEINFO.nValue)

        # Camera Channel ch:获取相机通道数 | en:Get camera channel
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_CAMERA_CHANNELS.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        print('Camera  Channels:%#X' % TUCAMVALUEINFO.nValue)

        # Camera Bus ch:获取相机通讯号 | en:Get camera bus number
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_BUS.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        print('Camera  Bus:%#X' % TUCAMVALUEINFO.nValue)

        # Sdk API ch:获取SDK版本号 | en:Get sdk version number
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_VERSION_API.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        print('Version API:%#s' %TUCAMVALUEINFO.pText)

        # FW ch:获取相机固件号 | en:Get camera firmware number
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_VERSION_FRMW.value, 0, 0, 0)
        TUCAM_Dev_GetInfo(self.TUCAMOPEN.hIdxTUCam, pointer(TUCAMVALUEINFO))
        if 0 == TUCAMVALUEINFO.nValue:
            print('Version Firmware:%#s'%TUCAMVALUEINFO.pText)
        else:
            print('Version Firmware:%#X' %TUCAMVALUEINFO.nValue)

        # SN ch:获取相机序列号 | en:Get camera serial number
        TUCAM_Reg_Read = TUSDKdll.TUCAM_Reg_Read
        cSN = (c_char * 64)()
        pSN = cast(cSN, c_char_p)
        TUCAMREGRW = TUCAM_REG_RW(1, pSN, 64)
        TUCAM_Reg_Read(self.TUCAMOPEN.hIdxTUCam, TUCAMREGRW)
        # print(bytes(bytearray(cSN)))
        print(string_at(pSN))

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.PrintCameraInfo()
        demo.CloseCamera()
    demo.UnInitApi()