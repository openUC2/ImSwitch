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

    def PrintCameraPropertyList(self):
        value = c_double(0)

        nums = range(TUCAM_IDPROP.TUIDP_GLOBALGAIN.value, TUCAM_IDPROP.TUIDP_ENDPROPERTY.value)
        prop = TUCAM_PROP_ATTR()
        # ch:获取相机属性信息列表 | en:Get property information list
        print('Get property information list')
        for num in nums:
            prop.idProp = num
            prop.nIdxChn = 0
            try:
                result = TUCAM_Prop_GetAttr(self.TUCAMOPEN.hIdxTUCam, pointer(prop))
                print('PropID=%#d Min=%#d Max=%#d Dft=%#d Step=%#d' %(prop.idProp, prop.dbValMin, prop.dbValMax, prop.dbValDft, prop.dbValStep))
            except Exception:
                #print('CapaID=%#d Not support' %(num))
                continue

        # ch:设置相机属性默认值 | en:Set property default value
        print('Set property default value')
        for num in nums:
            prop.idProp = num
            prop.nIdxChn = 0
            try:
                result = TUCAM_Prop_GetAttr(self.TUCAMOPEN.hIdxTUCam, pointer(prop))
                if num == TUCAM_IDPROP.TUIDP_HDR_KVALUE.value:
                    continue

                if num == TUCAM_IDPROP.TUIDP_BLACKLEVELHG.value:
                    continue

                if num == TUCAM_IDPROP.TUIDP_BLACKLEVELLG.value:
                    continue

                TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, prop.idProp, c_double(prop.dbValDft), 0);
                print("PropID=",prop.idProp, "Set default value", prop.dbValDft)
            except Exception:
                #print('PropID=%#d Not support' %(num))
                continue

        # ch:获取相机属性当前值 | en:Get property default value
        print('Get property current value')
        for num in nums:
            try:
                if num == TUCAM_IDPROP.TUIDP_HDR_KVALUE.value:
                    continue

                if num == TUCAM_IDPROP.TUIDP_BLACKLEVELHG.value:
                    continue

                if num == TUCAM_IDPROP.TUIDP_BLACKLEVELLG.value:
                    continue

                result = TUCAM_Prop_GetValue(self.TUCAMOPEN.hIdxTUCam, num, pointer(value), 0)
                print("PropID=", num, "The current value is=", value)
                #print('CapaID=%#X Current=%#X' %(num, value))
            except Exception:
                continue

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.PrintCameraPropertyList()
        demo.CloseCamera()
    demo.UnInitApi()