#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-04-02
@author:fdy
'''

import ctypes
from ctypes import *
from TUCam import *
from enum import Enum
import time
import threading
import numpy as np
import logging

# C:\Users\benir\Documents\ImSwitch\imswitch\imcontrol\model\interfaces\tucam_win\lib\x64
class Tucam():
    def __init__(self):
        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)

        # TODO: We have to deal with the situation that a camera may not be disconnected properly
        # and TUCAM_Api_Init fails the second time. For now, we just try to Uninit first.
        # I think this is not really smart
        # if we do not disconnect the camera correctly, the camera gets stuck in a timeout or code completiely gets stuck

       
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 2000)

        print(self.TUCAMINIT.uiCamCount)
        print(self.TUCAMINIT.pstrConfigPath)
        print('Connect %d camera' %self.TUCAMINIT.uiCamCount)

        # threading additions
        self._thread = None
        self._stop_evt = threading.Event()


    # add to class Tucam

    def _shutdown_capture(self):
        """Abort wait -> stop capture -> release buffer (safe no-op if not started)."""
        try:
            TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)  # 1) abort wait
        except Exception:
            pass
        try:
            TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)       # 2) stop capture
        except Exception:
            pass
        try:
            TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)    # 3) release buffer
        except Exception:
            pass


    @staticmethod
    def emergency_cleanup():
        """Emergency cleanup that can be called even without a proper instance"""
        print("Performing emergency cleanup...")
        try:
            # Try to uninit any existing API session
            TUCAM_Api_Uninit()
            time.sleep(0.5)
        except Exception:
            pass
        
        try:
            # Try to init and immediately uninit to reset state
            dummy_init = TUCAM_INIT(0, b'./')
            TUCAM_Api_Init(pointer(dummy_init), 1000)
            time.sleep(0.1)
            TUCAM_Api_Uninit()
            time.sleep(0.5)
        except Exception:
            pass
        
        print("Emergency cleanup completed")


    def _open_camera_with_timeout(self, timeout_seconds=5):
        """Open camera with timeout to prevent hanging"""
        result = {'success': False, 'exception': None}
        
        def open_camera_thread():
            try:
                TUCAM_Dev_Open(pointer(self.TUCAMOPEN))
                result['success'] = True
            except Exception as e:
                result['exception'] = e
        
        thread = threading.Thread(target=open_camera_thread, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            print(f"Camera open timed out after {timeout_seconds} seconds")
            return False
        
        if result['exception']:
            print(f"TUCAM_Dev_Open failed: {result['exception']}")
            return False
            
        return result['success']

    def OpenCamera(self, Idx):

        if  Idx >= self.TUCAMINIT.uiCamCount:
            return
        
        print(f"Camera count: {self.TUCAMINIT.uiCamCount}, requested index: {Idx}")
        
        # Cleanup any existing session before opening
        print("Cleaning up any existing camera session...")
        self._shutdown_capture()
        
        # Try to close any existing handle
        if hasattr(self, 'TUCAMOPEN') and getattr(self.TUCAMOPEN, 'hIdxTUCam', 0) != 0:
            try:
                TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
                print("Closed existing camera handle")
            except Exception as e:
                print(f"Error closing existing handle: {e}")
        
        self.TUCAMOPEN = TUCAM_OPEN(Idx, 0)
        print(f"Opening camera index {Idx}") 
        
        # Use timeout to prevent hanging
        if not self._open_camera_with_timeout(timeout_seconds=10):
            print("Failed to open camera (timeout or error)")
            return
            
        print(f"Camera handle: {self.TUCAMOPEN.hIdxTUCam}")
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

    # ch:计算黑白图片平均灰度值 | en:Calculate mono image average gray 
    def CalculateMonoAverageGray(self, m_frame):

        buf = create_string_buffer(m_frame.uiImgSize)
        pointer_data = c_void_p(m_frame.pBuffer + m_frame.usHeader)
        memmove(buf, pointer_data, m_frame.uiImgSize)

        sum = c_double()
        avg = c_double()
        sum = 0
        size = m_frame.usWidth * m_frame.usHeight

        buffers = bytes(buf)

        for i in range(size):
            if 1 == m_frame.ucElemBytes:
                value = buffers[i]
                sum += value
            else:
                value = buffers[i*2] + buffers[i*2+1] * 256
                sum += value

        avg = sum / size
        print("Grab the frame average is:", avg)

    # ch:计算彩色图片平均灰度值 | en:Calculate color image average gray
    def CalculateColorAverageGray(self, m_frame):

        buf = create_string_buffer(m_frame.uiImgSize)
        pointer_data = c_void_p(m_frame.pBuffer + m_frame.usHeader)
        memmove(buf, pointer_data, m_frame.uiImgSize)

        sum = c_double()
        avg = c_double()
        sum = 0
        size = m_frame.usWidth * m_frame.usHeight

        buffers = bytes(buf)

        for i in range(size):
            if 1 == m_frame.ucElemBytes:
                value = buffers[i*3] * 0.114 + buffers[i*3+1] * 0.587 + buffers[i*3+2] * 0.299
                sum += value
            else:
                value = (buffers[i*6] + buffers[i*6+1] * 256) * 0.114 + (buffers[i*6+2] + buffers[i*6+3] * 256) * 0.587 + (buffers[i*6+4] + buffers[i*6+5] * 256) * 0.299
                sum += value

        avg = sum / size
        print("Grab the frame average is:", avg)

    # ch:计算帧平均灰度值 | en:Calculate frame average gray
    def CalculateAverageGray(self, m_frame):

        if 1 == m_frame.ucChannels:
            self.CalculateMonoAverageGray(m_frame)
        else:
            self.CalculateColorAverageGray(m_frame)

    # ch:打印帧平均灰度值 | en:Print frame average gray
    def ShowAverageGray(self):
        m_frame = TUCAM_FRAME()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1
        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)
        t0 = time.time()
        nTimes = 1
        for i in range(nTimes):
            try:
                print(t0-time.time())
                result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
                print(
                    "Grab the frame success, index number is %#d, width:%d, height:%#d, channel:%#d, elembytes:%#d, image size:%#d"%(i, m_frame.usWidth, m_frame.usHeight, m_frame.ucChannels,
                    m_frame.ucElemBytes, m_frame.uiImgSize)
                    )
                self.CalculateAverageGray(m_frame)
            except Exception:
                print('Grab the frame failure, index number is %#d',  i)
                continue

        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

    def _convert_frame_to_numpy(self, frame: "TUCAM_FRAME"):
        try:
            if frame.uiImgSize == 0 or frame.pBuffer == 0:
                return None
            buf = create_string_buffer(frame.uiImgSize)
            pointer_data = c_void_p(frame.pBuffer + frame.usHeader)
            memmove(buf, pointer_data, frame.uiImgSize)
            data = bytes(buf)
            if frame.ucElemBytes == 1:
                dtype = np.uint8
            elif frame.ucElemBytes == 2:
                dtype = np.uint16
            else:
                print(f"Unsupported elem size: {frame.ucElemBytes}")
                return None
            arr = np.frombuffer(data, dtype=dtype)
            if frame.ucChannels == 1:
                arr = arr.reshape((int(frame.usHeight), int(frame.usWidth)))
            elif frame.ucChannels == 3:
                arr = arr.reshape((int(frame.usHeight), int(frame.usWidth), 3))
            else:
                print(f"Unsupported channels: {frame.ucChannels}")
                return None
            return arr
        except Exception as e:
            print(f"Convert frame failed: {e}")
            return None
        
    # ===== Threaded version preserving original syntax/flow =====
    def _thread_loop(self):
        print("Entered thread loop")
        m_frame = TUCAM_FRAME()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)

        i = 0
        while not self._stop_evt.is_set():
            try:
                result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
                print(
                    "Grab the frame success, index number is %#d, width:%d, height:%#d, channel:%#d, elembytes:%#d, image size:%#d"%(i, m_frame.usWidth, m_frame.usHeight, m_frame.ucChannels,
                    m_frame.ucElemBytes, m_frame.uiImgSize)
                    )
                #self.CalculateAverageGray(m_frame)
                mFrameNP = self._convert_frame_to_numpy(m_frame)

                i += 1
            except Exception:
                print('Grab the frame failure, index number is %#d',  i)
                continue

        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

    def StartAverageGrayThread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._thread_loop, name="TUCamAverageGray", daemon=True)
        self._thread.start()

    def StopAverageGrayThread(self):
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join()
        self._thread = None


if __name__ == '__main__':
    # Perform emergency cleanup first to handle any stuck sessions
    Tucam.emergency_cleanup()
    
    demo = Tucam()
    
    # Attempt to open camera with cleanup
    print("Attempting to open camera...")
    try:
        demo.OpenCamera(0)
    except Exception as e:
        print(f"OpenCamera failed: {e}")

    if demo.TUCAMOPEN.hIdxTUCam != 0:
        # one-shot (original)
        # demo.ShowAverageGray()

        # threaded continuous grab (press Ctrl+C to stop or call StopAverageGrayThread elsewhere)
        try:
            demo.StartAverageGrayThread()
            time.sleep(5)  # run for a short while; adjust as needed
        except KeyboardInterrupt:
            pass
        finally:
            demo.StopAverageGrayThread()

        demo.CloseCamera()
    demo.UnInitApi()
