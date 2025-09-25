#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-08
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

    def DoOutputTrigger(self):

        m_tgrout = TUCAM_TRGOUT_ATTR()
        try:
            # ch:获取相机触发输出 | en:Get camera trigger output
            TUCAM_Cap_GetTriggerOut(self.TUCAMOPEN.hIdxTUCam, pointer(m_tgrout))
            print('Get output trigger success, Port:%#d, Kind:%#d, Edge:%#d, Delay time:%#d, Width:%#d'%(m_tgrout.nTgrOutPort,m_tgrout.nTgrOutMode,m_tgrout.nEdgeMode,m_tgrout.nDelayTm,m_tgrout.nWidth))
        except Exception:
            print('Get output trigger failure')

        try:
            # ch:设置相机触发输出 | en:Set camera trigger output
            TUCAM_Cap_SetTriggerOut(self.TUCAMOPEN.hIdxTUCam, m_tgrout)
            print('Set output trigger success, Port:%#d, Kind:%#d, Edge:%#d, Delay time:%#d, Width:%#d'%(m_tgrout.nTgrOutPort,m_tgrout.nTgrOutMode,m_tgrout.nEdgeMode,m_tgrout.nDelayTm,m_tgrout.nWidth))
        except Exception:
            print('Set output trigger failure')

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.DoOutputTrigger()
        demo.CloseCamera()
    demo.UnInitApi()