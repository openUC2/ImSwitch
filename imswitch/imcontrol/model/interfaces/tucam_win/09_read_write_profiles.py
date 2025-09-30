#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-05
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

    def ReadWriteProfiles(self):

        strParam = 'Parameter_A'
        try:
            # ch:保存相机的配置文件 | en:Save camera profiles
            TUCAM_File_SaveProfiles(self.TUCAMOPEN.hIdxTUCam, strParam.encode('utf-8'))
            print('Save %#s profile success'%strParam)
        except Exception:
            print('Save %#s profile failure'%strParam)

        try:
            # ch:加载相机的配置文件 | en:Load camera profiles
            TUCAM_File_LoadProfiles(self.TUCAMOPEN.hIdxTUCam, strParam.encode('utf-8'))
            print('Load %#s profile success'%strParam)
        except Exception:
            print('Load %#s profile failure'%strParam)

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.ReadWriteProfiles()
        demo.CloseCamera()
    demo.UnInitApi()