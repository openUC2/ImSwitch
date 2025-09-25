#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-03
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
        print('Connect %d camera' %self.TUCAMINIT.uiCamCount)

    def OpenCamera(self, Idx):

        if  Idx >= self.TUCAMINIT.uiCamCount:
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

    def PrintCameraCapabilityList(self):
        value = c_int32(0)
        valText = TUCAM_VALUE_TEXT()

        nums = range(TUCAM_IDCAPA.TUIDC_RESOLUTION.value, TUCAM_IDCAPA.TUIDC_ENDCAPABILITY.value)
        # ch:获取相机性能信息列表 | en:Get capability information list
        print('Get capability information list')
        for num in nums:
            capa = TUCAM_CAPA_ATTR()
            capa.idCapa = num
            try:
                result = TUCAM_Capa_GetAttr(self.TUCAMOPEN.hIdxTUCam, pointer(capa))
                if num == TUCAM_IDCAPA.TUIDC_RESOLUTION.value:
                    cnt = capa.nValMax - capa.nValMin + 1
                    szRes = (c_char * 64)()
                    for j in range(cnt):
                        valText.nID = num
                        valText.dbValue = j
                        valText.nTextSize = 64
                        valText.pText = cast(szRes, c_char_p)
                        TUCAM_Capa_GetValueText(self.TUCAMOPEN.hIdxTUCam, pointer(valText))
                        print('%#d, Resolution =%#s'%(j,valText.pText))

                print('CapaID=%#d Min=%#d Max=%#d Dft=%#d Step=%#d' %(capa.idCapa, capa.nValMin, capa.nValMax, capa.nValDft, capa.nValStep))
            except Exception:
                #print('CapaID=%#d Not support' %(num))
                continue

        # ch:设置相机性能默认值 | en:Set capability default value
        print('Set capability default value')
        for num in nums:
            capa = TUCAM_CAPA_ATTR()
            capa.idCapa = num
            try:
                result = TUCAM_Capa_GetAttr(self.TUCAMOPEN.hIdxTUCam, pointer(capa))
                TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, capa.idCapa, capa.nValDft)
                print('CapaID=%#d Set default value %#d success'%(capa.idCapa, capa.nValDft))
            except Exception:
                continue

        # ch:获取相机性能当前值 | en:Get capability default value
        print('Get capability current value')
        for num in nums:
            try:
                result = TUCAM_Capa_GetValue(self.TUCAMOPEN.hIdxTUCam, num, pointer(value))
                print("CapaID=", num, "The current value is=", value)
            except Exception:
                continue

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.PrintCameraCapabilityList()
        demo.CloseCamera()
    demo.UnInitApi()