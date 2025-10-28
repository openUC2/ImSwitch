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

# Per-client frame acknowledgement tracking
_client_frame_ready = {}  # sid -> bool (True if client is ready for next frame)
_client_frame_lock = threading.Lock()
_frame_drop_counter = 0  # Track how many frames we've dropped


class SignalInterface(abstract.SignalInterface):
    """Base implementation of abstract.SignalInterface."""
    def __init__(self) -> None:
        ...

class SignalInstance(psygnal.SignalInstance):
    last_emit_time = 0
    emit_interval = 0.0  # Emit at most every 100ms
    image_id = 0
    _sending_image = False  # To avoid re-entrance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize timing for binary streaming
        self._last_frame_emit_time = 0

    def emit(
        self, *args: Any, check_nargs: bool = False, check_types: bool = False
    ) -> None:
        """Emit the signal and broadcast to Socket.IO clients."""
        super().emit(*args, check_nargs=check_nargs, check_types=check_types)

        if not args:
            return

        # Handle pre-formatted stream messages from LiveViewController
        if self.name == "sigUpdateImage":
            self._handle_stream_frame(args[0])
            return
        elif self.name in ["sigImageUpdated", "sigStreamFrame"]:
            return # ignore for now TODO: This signal is mapped into sigImageUpdated of MasterController - we should think about a better way of handling this

        try:
            message = self._generate_json_message(args)
        except Exception as e:
            print(f"Error creating JSON message: {e}")
            return

        self._safe_broadcast_message(message)
        del message

    def _handle_stream_frame(self, message: dict):
        """
        Handle pre-formatted stream frame message from LiveViewController.
        Message format: {'type': 'binary_frame' or 'jpeg_frame', 'event': str, 'data': bytes or dict, 'metadata': dict}
        Uses explicit frame_ack event from frontend for flow control.
        Implements proper backpressure - only sends to clients that are ready.
        """
        try:
            msg_type = message.get('type')
            event = message.get('event', 'frame')
            data = message.get('data')
            metadata = message.get('metadata', {})

            # Get ready clients
            with _client_frame_lock:
                ready_clients = [sid for sid, ready in _client_frame_ready.items() if ready] # TODO: if we have two browser tabs open, this mechanism won't work properly anymore, timing issue I guess

            if not ready_clients:
                # No clients ready - drop this frame (implements backpressure)
                global _frame_drop_counter
                _frame_drop_counter += 1
                if _frame_drop_counter % 10 == 0:  # Log every 30 dropped frames
                    # set all available clients to ready to avoid infinite dropping
                    with _client_frame_lock:
                        print(f"Dropped {_frame_drop_counter} frames due to client backpressure, try to send some frames anyway using the protocol {msg_type}") # TODO: if we have e.g. 100 dropped frames, we should stop the live stream but not the camera
                        for sid in _client_frame_ready.keys():
                            _client_frame_ready[sid] = True
                            ready_clients.append(sid)
                else:
                    return # TODO: this is global stop

            # Mark clients as NOT ready (waiting for acknowledgement via frame_ack event)
            # Frontend will send frame_ack event after processing the frame
            with _client_frame_lock:
                for sid in ready_clients: #TODO: check if
                    _client_frame_ready[sid] = False  # Wait for frame_ack event!

            # Unified frame emission for both binary and JPEG
            # Both use the same backpressure mechanism via frame_ack event handler
            if msg_type == 'binary_frame':
                # Binary frame with metadata - Socket.IO supports sending binary + JSON together!
                # Send as array: [metadata, binaryData] - Socket.IO automatically handles serialization
                frame_payload = [metadata, data]  # First element: JSON metadata, Second: binary data

                for sid in ready_clients:
                    try:
                        # Emit binary frame with embedded metadata
                        # Client will send frame_ack event after processing
                        sio.start_background_task(
                            sio.emit,
                            event,  # 'frame' event
                            frame_payload,  # [metadata, binaryData]
                            to=sid
                        )
                    except RuntimeError: #RuntimeError: There is no current event loop in thread 'Thread-16 (run)'.
                        # No event loop in thread - use fallback frame queue (with dropping)
                        #_start_fallback_worker()
                        print("you are still broken")


            elif msg_type == 'jpeg_frame':
                # JPEG frame - emit as JSON signal
                # Client will send frame_ack event after processing
                json_message = json.dumps(data)
                for sid in ready_clients:
                    try:
                        # Emit JPEG frame
                        # Client will send frame_ack event after processing
                        sio.start_background_task(
                            sio.emit,
                            "signal",
                            json_message,
                            to=sid
                        )
                    except RuntimeError:
                        # No event loop in thread - use fallback FRAME queue (not message queue!)
                        # This ensures JPEG frames get same dropping behavior as binary frames
                        #_start_fallback_worker()
                        print("you are still broken")

        except Exception as e:
            print(f"Error handling stream frame: {e}")

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
            # Create JSON string before async task to avoid closure issues
            message_json = json.dumps(mMessage)

            # Wrap the emit call in an async function to properly await it
            async def emit_signal():
                await sio.emit("signal", message_json)

            sio.start_background_task(emit_signal)
        except Exception as e:
            # Use fallback worker thread instead of creating new threads
            #message = (json.dumps(mMessage) if isinstance(mMessage, dict)
            #            else mMessage)
            print("you are still broken")

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

# Socket.IO event handlers for client connection management
@sio.event
async def connect(sid, environ):
    """Handle client connection - mark as ready for frames"""
    print(f"SocketIO Client connected: {sid}")
    with _client_frame_lock:
        _client_frame_ready[sid] = True


@sio.event
async def disconnect(sid):
    """Handle client disconnection - cleanup state"""
    print(f"SocketIO Client disconnected: {sid}")
    with _client_frame_lock:
        _client_frame_ready.pop(sid, None)

@sio.event
async def frame_ack(sid):
    """Client explicitly acknowledges frame processing complete"""
    with _client_frame_lock:
        _client_frame_ready[sid] = True
        #print(f"Client {sid} acknowledged frame")

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
