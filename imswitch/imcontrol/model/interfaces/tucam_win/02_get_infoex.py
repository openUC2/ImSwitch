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

    def UnInitApi(self):
        # ch:反初始化相机 | en:Uninitial Cameras
        TUCAM_Api_Uninit()

    # Get Camera Info
    def PrintCameraInfoEx(self, uiIdex):

        if uiIdex >= self.TUCAMINIT.uiCamCount:
            print('PrintCameraInfoEx: The index number of camera is out of range')
            return

        # Camera name: ch:获取相机名称 | en:Get camera name
        m_infoid = TUCAM_IDINFO
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_CAMERA_MODEL.value, 0, 0, 0)
        TUCAM_Dev_GetInfoEx(uiIdex, pointer(TUCAMVALUEINFO))
        print('Camera Name:%#s' %TUCAMVALUEINFO.pText)

        # Camera VID
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_VENDOR.value, 0, 0, 0)
        TUCAM_Dev_GetInfoEx(uiIdex, pointer(TUCAMVALUEINFO))
        print('Camera  VID:%#X' % TUCAMVALUEINFO.nValue)

        # Camera PID
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_PRODUCT.value, 0, 0, 0)
        TUCAM_Dev_GetInfoEx(uiIdex, pointer(TUCAMVALUEINFO))
        print('Camera  PID:%#X' % TUCAMVALUEINFO.nValue)

        # Camera Bus ch:获取相机通讯号 | en:Get camera bus number
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_BUS.value, 0, 0, 0)
        TUCAM_Dev_GetInfoEx(uiIdex, pointer(TUCAMVALUEINFO))
        print('Camera  Bus:%#X' % TUCAMVALUEINFO.nValue)

        # Sdk API ch:获取SDK版本号 | en:Get sdk version number
        TUCAMVALUEINFO = TUCAM_VALUE_INFO(m_infoid.TUIDI_VERSION_API.value, 0, 0, 0)
        TUCAM_Dev_GetInfoEx(uiIdex, pointer(TUCAMVALUEINFO))
        print('Version API:%#s' %TUCAMVALUEINFO.pText)

if __name__ == '__main__':
    demo = Tucam()
    demo.PrintCameraInfoEx(0)
    demo.UnInitApi()