from typing import TYPE_CHECKING, Any
import psygnal.utils
psygnal.utils.decompile() # https://github.com/pyapp-kit/psygnal/pull/331#issuecomment-2455192644
from psygnal import emit_queued
import psygnal
import asyncio
import threading
import os
import json
from functools import lru_cache
import numpy as np
from socketio import AsyncServer, ASGIApp
import uvicorn
import imswitch.imcommon.framework.base as abstract
import cv2
import base64
from imswitch import SOCKET_STREAM
import time
import queue
if TYPE_CHECKING:
    from typing import Tuple, Callable, Union
import imswitch
from imswitch import __ssl__, __socketport__
import logging

# Try to import config and binary streaming - handle gracefully if not available
from imswitch.config import get_config
from imswitch.imcommon.framework.binary_streaming import BinaryFrameEncoder
HAS_BINARY_STREAMING = True
class Mutex(abstract.Mutex):
    """Wrapper around the `threading.Lock` class."""
    def __init__(self) -> None:
        self.__lock = threading.Lock()

    def lock(self) -> None:
        self.__lock.acquire()

    def unlock(self) -> None:
        self.__lock.release()


# Initialize Socket.IO server
sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = ASGIApp(sio)

# Fallback message queue and worker thread for Socket.IO failures
_fallback_message_queue = queue.Queue()
_fallback_worker_thread = None


def _fallback_worker():
    """Background worker thread to handle Socket.IO fallback messages."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                message = _fallback_message_queue.get(timeout=1.0)
                if message is None:  # Poison pill to stop worker
                    break
                loop.run_until_complete(sio.emit("signal", message))
                _fallback_message_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Silently handle errors to avoid spam
                pass
    finally:
        loop.close()


def _start_fallback_worker():
    """Start the fallback worker thread if not already running."""
    global _fallback_worker_thread
    if _fallback_worker_thread is None or not _fallback_worker_thread.is_alive():
        _fallback_worker_thread = threading.Thread(target=_fallback_worker,
                                                    daemon=True)
        _fallback_worker_thread.start()

class SignalInterface(abstract.SignalInterface):
    """Base implementation of abstract.SignalInterface."""
    def __init__(self) -> None:
        ...

class SignalInstance(psygnal.SignalInstance):
    last_emit_time = 0
    emit_interval = 0.0  # Emit at most every 100ms
    last_image_emit_time = 0
    image_emit_interval = .2  # Emit at most every 200ms
    IMG_QUALITY = 80  # Set the desired quality level (0-100)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize timing for binary streaming
        self._last_binary_emit_time = 0
    def emit(
        self, *args: Any, check_nargs: bool = False, check_types: bool = False
    ) -> None:
        """Emit the signal and broadcast to Socket.IO clients."""
        super().emit(*args, check_nargs=check_nargs, check_types=check_types)

        if not args:
            return

        # Skip large data signals
        if self.name in ["sigUpdateImage", "sigExperimentImageUpdate"]:  #, "sigImageUpdated"]:
            now = time.time()
            if SOCKET_STREAM and (now - self.last_image_emit_time > self.image_emit_interval) or self.name == "sigExperimentImageUpdate":
                self._handle_image_signal(args)
                self.last_image_emit_time = now
            return
        elif self.name in ["sigImageUpdated"]:
            return # ignore for now TODO:

        try:
            message = self._generate_json_message(args)
        except Exception as e:
            print(f"Error creating JSON message: {e}")
            return

        self._safe_broadcast_message(message)
        del message

    def _handle_image_signal(self, args):
        """Compress and broadcast image signals."""
        detectorName = args[0]
        try:pixelSize = np.min(args[3])
        except:pixelSize = 1
        
        # Get stream parameters from globalDetectorParams (like compressionlevel)
        try:
            # Get the global detector parameters if available
            global_params = args[5] if len(args) > 5 and isinstance(args[5], dict) else {}
        except:
            global_params = {}
        
        try:
            for arg in args:
                if isinstance(arg, np.ndarray):
                    output_frame = np.ascontiguousarray(arg)  # Avoid memory fragmentation
                    
                    # Check stream parameters for binary streaming
                    stream_compression_algorithm = global_params.get('stream_compression_algorithm')
                    if stream_compression_algorithm in ["LZ4", "lz4", "ZSTD", "zstd", "ZStandard"]:
                        self._emit_binary_frame(output_frame, detectorName, pixelSize, global_params)
                    elif stream_compression_algorithm == "jpeg":
                        # emit JPEG (legacy behavior for compatibility)
                        self._emit_jpeg_frame(output_frame, detectorName, pixelSize, global_params)
                    
                        
        except Exception as e:
            print(f"Error processing image signal: {e}")
    
    def _emit_binary_frame(self, img: np.ndarray, detector_name: str, pixel_size: float, global_params: dict):
        """Emit binary frame via Socket.IO."""
        if not HAS_BINARY_STREAMING:
            return
            
        # Get throttle interval from global params
        throttle_ms = global_params.get('stream_throttle_ms', 200) / 1000.0
        
        now = time.time()
        if now - self._last_binary_emit_time < throttle_ms:
            return  # Throttle binary emissions
        
        # Update encoder config from global params
        compression_algorithm = global_params.get('stream_compression_algorithm', 'lz4')
        compression_level = global_params.get('stream_compression_level', 0)
        subsampling_factor = global_params.get('stream_subsampling_factor', 1)
        
        # Create or update encoder with current parameters
        try:
            from imswitch.imcommon.framework.binary_streaming import BinaryFrameEncoder
            encoder = BinaryFrameEncoder(
                compression_algorithm=compression_algorithm,
                compression_level=compression_level,
                subsampling_factor=subsampling_factor,
            )
        except ImportError:
            return  # Binary streaming not available
        
        try:
            # Encode frame
            packet, metadata = encoder.encode_frame(img)
            
            # Emit binary frame
            sio.start_background_task(sio.emit, "frame", packet)
            
            # Emit metadata as JSON signal
            meta_message = {
                "name": "frame_meta",
                "detectorname": detector_name,
                "pixelsize": int(pixel_size),
                "format": "binary",
                "metadata": metadata
            }
            sio.start_background_task(sio.emit, "signal", json.dumps(meta_message))
            
            self._last_binary_emit_time = now
            
        except Exception as e:
            print(f"Error emitting binary frame: {e}")
            # Fallback to JPEG if binary fails
            self._emit_jpeg_frame(img, detector_name, pixel_size, global_params)
    
    def _emit_jpeg_frame(self, output_frame: np.ndarray, detector_name: str, pixel_size: float, global_params: dict):
        """Emit JPEG frame (legacy path)."""
        try:
            # Apply legacy subsampling logic
            if output_frame.shape[0] > 640 or output_frame.shape[1] > 480:
                everyNthsPixel = np.min((np.min([output_frame.shape[0]//240, output_frame.shape[1]//320]), 3))
            else:
                everyNthsPixel = 1

            try:
                output_frame = output_frame[::everyNthsPixel, ::everyNthsPixel]
            except:
                output_frame = np.zeros((640,460))
                
            # convert 16 bit to 8 bit for visualization
            if output_frame.dtype == np.uint16:
                output_frame = np.uint8(output_frame//128) 

            # Get JPEG quality from global params
            jpegQuality = global_params.get("compressionlevel", self.IMG_QUALITY)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpegQuality]

            # Compress image using JPEG format
            flag, compressed = cv2.imencode(".jpg", output_frame, encode_params)
            encoded_image = base64.b64encode(compressed).decode('utf-8')

            # Create a minimal message
            message = {
                "name": self.name, # e.g. sigUpdateImage
                "detectorname": detector_name,
                "pixelsize": int(pixel_size), # must not be int64
                "format": "jpeg",
                "image": encoded_image,
            }
            self._safe_broadcast_message(message)
            del message
        except Exception as e:
            print(f"Error processing JPEG image signal: {e}")

            # Create a minimal message
            message = {
                "name": self.name,
                "detectorname": detector_name,
                "pixelsize": int(pixel_size), # must not be int64
                "format": "jpeg",
                "image": encoded_image,
            }
            self._safe_broadcast_message(message)
            del message
        except Exception as e:
            self._logger.error(f"Error emitting JPEG frame: {e}")
    
    def update_binary_config(self, **kwargs):
        """Update binary streaming configuration at runtime."""
        if self._binary_encoder is not None:
            self._binary_encoder.update_config(**kwargs)
        
        # Update throttle if provided
        if 'throttle_ms' in kwargs:
            self._binary_throttle_ms = kwargs['throttle_ms'] / 1000.0

    def _generate_json_message(self, args):  # Consider using msgpspec for efficiency
        param_names = list(self.signature.parameters.keys())
        data = {"name": self.name, "args": {}}

        for i, arg in enumerate(args):
            param_name = param_names[i] if i < len(param_names) else f"arg{i}"
            if isinstance(arg, np.ndarray):
                data["args"][param_name] = arg.tolist().copy()
            elif isinstance(arg, (str, int, float, bool)):
                data["args"][param_name] = arg
            elif isinstance(arg, dict):
                data["args"][param_name] = arg.copy()
            else:
                data["args"][param_name] = str(arg)

        return data



    def _safe_broadcast_message(self, mMessage: dict) -> None:
        """Throttle the emit to avoid task buildup."""
        now = time.time()
        if now - self.last_emit_time < self.emit_interval:
            # print("too fast")
            return  # Skip if emit interval hasn't passed
        self.last_emit_time = now

        try:
            sio.start_background_task(sio.emit, "signal", json.dumps(mMessage))
        except Exception as e:
            # Use fallback worker thread instead of creating new threads
            try:
                _start_fallback_worker()
                message = (json.dumps(mMessage) if isinstance(mMessage, dict)
                           else mMessage)
                _fallback_message_queue.put_nowait(message)
            except queue.Full:
                # Queue is full, drop the message to prevent memory buildup
                pass
            except Exception as e:
                # Silently handle any other errors
                print(f"Error broadcasting message: {e}")
                pass
        del mMessage

class Signal(psygnal.Signal):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signal_instance_class = SignalInstance
        self._info = "ImSwitch signal"

    def connect(self, func: 'Union[Callable, abstract.Signal]') -> None:
        if isinstance(func, abstract.Signal):
            if any([t1 != t2 for t1, t2 in zip(self.types, func.types)]):
                raise TypeError(f"Source and destination must have the same signature.")
            func = func.emit
        super().connect(func)

    def disconnect(self, func: 'Union[Callable, abstract.Signal, None]' = None) -> None:
        if func is None:
            super().disconnect()
        super().disconnect(func)

    def emit(self, *args) -> None:
        instance = self._signal_instance_class(*args)
        asyncio.create_task(instance.emit(*args))

    @property
    @lru_cache
    def types(self) -> 'Tuple[type, ...]':
        return tuple([param.annotation for param in self._signature.parameters.values()])

    @property
    def info(self) -> str:
        return self._info

# Threaded workers for async tasks
class Worker(abstract.Worker):
    def __init__(self) -> None:
        self._thread = None

    def moveToThread(self, thread: abstract.Thread) -> None:
        self._thread = thread
        thread._worker = self

class Thread(abstract.Thread):
    _started = Signal()
    _finished = Signal()

    def __init__(self):
        self._thread = None
        self._loop = None
        self._running = threading.Event()
        self._worker: Worker = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            if self._worker is not None:
                self._worker._thread = self
            self._running.set()
            self._thread.start()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.emit()
        try:
            while self._running.is_set():
                self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            self._loop = None
            self._finished.emit()

    def quit(self):
        self._running.clear()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def wait(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def isRunning(self):
        """Check if the thread is currently running (for Qt compatibility)."""
        return self._running.is_set() and self._thread is not None and self._thread.is_alive()

    @property
    def started(self) -> Signal:
        return self._started

    @property
    def finished(self) -> Signal:
        return self._finished

# Timer utility
class Timer(abstract.Timer):
    _timeout = Signal()

    def __init__(self, singleShot=False):
        self._task = None
        self._singleShot = singleShot
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self, millisecs):
        self._interval = millisecs / 1000.0
        if self._task:
            self._task.cancel()
        self._task = asyncio.run_coroutine_threadsafe(self._run(), self._loop)

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None
            self._loop.stop()

    async def _run(self):
        await asyncio.sleep(self._interval)
        self.timeout.emit()
        if not self._singleShot:
            self._task = self._loop.create_task(self._run())

    @property
    def timeout(self) -> Signal:
        return self._timeout

def threadCount() -> int:
    """Returns the current number of active threads of this framework."""
    return threading.active_count()

class FrameworkUtils(abstract.FrameworkUtils):
    @staticmethod
    def processPendingEventsCurrThread():
        emit_queued()

# Function to run Uvicorn server with Socket.IO app
def run_uvicorn():
    try:
        _baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(imswitch.__file__)), '_data')
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=__socketport__,
            ssl_keyfile=os.path.join(_baseDataFilesDir, "ssl", "key.pem") if __ssl__ else None,
            ssl_certfile=os.path.join(_baseDataFilesDir, "ssl", "cert.pem") if __ssl__ else None,
            timeout_keep_alive=2,
        )
        try:
            uvicorn.Server(config).run()
        except Exception as e:
            print(f"Couldn't start server: {e}")
    except Exception as e:
        print(f"Couldn't start server: {e}")

def start_websocket_server():
    server_thread = threading.Thread(target=run_uvicorn, daemon=True)
    server_thread.start()

start_websocket_server()
