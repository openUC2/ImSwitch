from typing import TYPE_CHECKING, Any
import psygnal.utils
psygnal.utils.decompile() # https://github.com/pyapp-kit/psygnal/pull/331#issuecomment-2455192644
from psygnal import emit_queued
import psygnal
import asyncio
import threading
from functools import lru_cache
import numpy as np
from socketio import AsyncServer, ASGIApp
import imswitch.imcommon.framework.base as abstract
import time
if TYPE_CHECKING:
    from typing import Tuple, Callable, Union
import msgpack  # MessagePack for efficient binary serialization


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
socket_app = ASGIApp(sio, socketio_path=None)  # Renamed to socket_app - will be mounted on FastAPI app

# Per-client frame acknowledgement tracking
_client_sent_frame_id = {}  # sid -> last sent frame id (int or None)
_client_ack_frame_id = {}  # sid -> last acked frame id
_client_frame_lock = threading.Lock()

# Event loop reference - will be set by ImSwitchServer
_shared_event_loop = None


class SignalInterface(abstract.SignalInterface):
    """Base implementation of abstract.SignalInterface."""
    def __init__(self) -> None:
        ...

class SignalInstance(psygnal.SignalInstance):
    last_emit_time = 0
    emit_interval = 0.0  # Emit at most every 100ms

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

        # Handle log signal specially
        if self.name == "sigLog":
            # For log signals, args[0] should be a dict with log data
            try:
                log_data = args[0] if args else {}
                message = {
                    "signal": "sigLog",
                    "args": log_data
                }
                self._safe_broadcast_message(message)
            except Exception as e:
                print(f"Error broadcasting log message: {e}")
            return

        # Handle pre-formatted stream messages from LiveViewController
        elif self.name == "sigUpdateImage" or self.name == "sigUpdateFrame" or self.name == "sigImageReceived" or self.name == "sigHoloImageComputed" or self.name == "sigHoloProcessed": # both stemp from the liveviewcontroller
            return
        elif self.name == "sigUpdateStreamFrame":
            # this can be binary or jpeg frame
            self._handle_stream_frame(args[0])

            return
        elif self.name in ["sigImageUpdated", "sigStreamFrame"]:
            # These signals are internal to the streaming pipeline:
            # - sigStreamFrame: emitted by StreamWorkers (already handled by sigUpdateImage)
            # - sigImageUpdated: legacy signal from MasterController (deprecated in favor of unified frame events)
            # Skip broadcasting to avoid duplicate frame transmissions
            return

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
        Thread-safe using asyncio.run_coroutine_threadsafe for cross-thread calls.

        UNIFIED HANDLING: Both binary and JPEG frames use the same 'frame' event
        with MessagePack-encoded metadata for efficiency and consistency.
        """
        try:
            msg_type = message.get('type')
            event = message.get('event', 'frame')
            data = message.get('data')
            metadata = message.get('metadata', {})

            def get_ready_clients(last_ack, last_sent):
                """
                Determine which clients are ready for the next frame.
                Implements rollover-safe backpressure check.
                """
                FRAME_ID_MODULO = 256  # Small value to test rollover frequently
                MAX_FRAME_LAG = 1  # Allow client to be 1 frame behind

                next_id = {}
                for sid, sent_id in last_sent.items():
                    # Initialize: client is ready for first frame
                    if sent_id is None or last_ack[sid] is None:
                        next_id[sid] = 0
                        continue

                    # Calculate distance between sent and ack with rollover awareness
                    # Distance = (sent - ack) mod MODULO
                    # If distance <= MAX_FRAME_LAG, client is ready
                    distance = (sent_id - last_ack[sid]) % FRAME_ID_MODULO

                    if distance <= MAX_FRAME_LAG:
                        # Client is ready for next frame
                        next_id[sid] = (sent_id + 1) % FRAME_ID_MODULO
                    else:
                        # Client is lagging too much - apply backpressure
                        pass # print(f"Client {sid} not ready for new frame (last sent: {sent_id}, last ack: {last_ack[sid]}, distance: {distance})")

                return next_id

            # Get ready clients
            with _client_frame_lock:
                ready_clients = get_ready_clients(_client_ack_frame_id, _client_sent_frame_id)

                if not ready_clients:
                    # print("No clients ready for new frame, dropping frame to avoid buildup")
                    return

                # Thread-safe emission using the shared event loop
                if not _shared_event_loop or not _shared_event_loop.is_running():
                    print("Warning: Event loop not available for frame emission")
                    return

                # Unified frame emission for both binary and JPEG using MessagePack
                if msg_type == 'binary_frame':
                    # Binary frame: Send complete payload as MessagePack
                    for sid, next_frame_id in ready_clients.items():
                        # Create a copy of metadata for each client to avoid race conditions
                        client_metadata = metadata.copy()
                        client_metadata['frame_id'] = next_frame_id

                        # Pack entire frame (metadata + data) with MessagePack
                        frame_payload = msgpack.packb({
                            'metadata': client_metadata,
                            'data': data
                        }, use_bin_type=True)

                        # print(f"Sending binary frame #{next_frame_id} to client {sid} (total: {len(frame_payload)} bytes)")
                        _client_sent_frame_id[sid] = next_frame_id

                        # Emit using socket.io's native binary support
                        asyncio.run_coroutine_threadsafe(
                            sio.emit(event, frame_payload, to=sid),
                            _shared_event_loop
                        )

                elif msg_type == 'jpeg_frame':
                    # JPEG frame: Send complete payload as MessagePack
                    # Note: For JPEG, metadata is in data['metadata'], image is in data['image']
                    for sid, next_frame_id in ready_clients.items():
                        # Create a copy of metadata for each client to avoid race conditions
                        client_metadata = data.get('metadata', {}).copy()
                        client_metadata['frame_id'] = next_frame_id

                        # Pack entire frame (metadata + image) with MessagePack
                        frame_payload = msgpack.packb({
                            'metadata': client_metadata,
                            'image': data['image']
                        }, use_bin_type=True)

                        _client_sent_frame_id[sid] = next_frame_id

                        # Emit on same 'frame' event as binary for unified frontend handling
                        asyncio.run_coroutine_threadsafe(
                            sio.emit(event, frame_payload, to=sid),
                            _shared_event_loop
                        )


        except Exception as e:
            print(f"Error handling stream frame: {e}")

    def _generate_json_message(self, args):
        """Generate message dict from signal arguments (will be serialized with MessagePack or JSON)."""
        param_names = list(self.signature.parameters.keys())
        data = {"name": self.name, "args": {}}

        for i, arg in enumerate(args):
            param_name = param_names[i] if i < len(param_names) else f"arg{i}"
            if isinstance(arg, np.ndarray):
                # Convert numpy arrays to lists for serialization
                data["args"][param_name] = arg.tolist()
            elif isinstance(arg, (str, int, float, bool)):
                data["args"][param_name] = arg
            elif isinstance(arg, dict):
                data["args"][param_name] = arg.copy()
            else:
                data["args"][param_name] = str(arg)

        return data

    def _safe_broadcast_message(self, mMessage: dict) -> None:
        """Throttle the emit to avoid task buildup. Thread-safe using call_soon_threadsafe."""
        now = time.time()
        if now - self.last_emit_time < self.emit_interval:
            # print("too fast")
            return  # Skip if emit interval hasn't passed
        self.last_emit_time = now

        try:
            # Serialize message using MessagePack
            message_bytes = msgpack.packb(mMessage, use_bin_type=True)
            event_name = "signal_msgpack"

            # Thread-safe emission using call_soon_threadsafe
            if _shared_event_loop and _shared_event_loop.is_running():
                async def emit_signal():
                    await sio.emit(event_name, message_bytes)

                # Schedule the coroutine in the shared event loop from any thread
                asyncio.run_coroutine_threadsafe(emit_signal(), _shared_event_loop)
            else:
                # Event loop not available yet - This happens during app startup
                # before ImSwitchServer has initialized the shared event loop
                # Signals emitted during this phase will be dropped (acceptable during init)
                pass
        except Exception as e:
            print(f"Error in safe broadcast message: {e}")

class Signal(psygnal.Signal):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signal_instance_class = SignalInstance
        self._info = "ImSwitch signal"

    def connect(self, func: 'Union[Callable, abstract.Signal]') -> None:
        if isinstance(func, abstract.Signal):
            if any([t1 != t2 for t1, t2 in zip(self.types, func.types)]):
                raise TypeError("Source and destination must have the same signature.")
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


# Global signal for log messages
sigLog = Signal(dict)


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
    """Handle client connection - mark as ready for frames and send capabilities"""
    print(f"SocketIO Client connected: {sid}")
    with _client_frame_lock:
        _client_sent_frame_id[sid] = None
        _client_ack_frame_id[sid] = None

    # Send server capabilities to client
    capabilities = {
        "messagepack": True,
        "binary_streaming": HAS_BINARY_STREAMING,
        "protocol_version": "1.0"
    }
    await sio.emit("server_capabilities", capabilities, to=sid)
    print(f"Sent capabilities to {sid}: {capabilities}")


@sio.event
async def disconnect(sid):
    """Handle client disconnection - cleanup state"""
    print(f"SocketIO Client disconnected: {sid}")
    with _client_frame_lock:
        _client_sent_frame_id.pop(sid, None)
        _client_ack_frame_id.pop(sid, None)

@sio.event
async def frame_ack(sid, data):
    """Client explicitly acknowledges frame processing complete"""
    with _client_frame_lock:
        _client_ack_frame_id[sid] = data.get('frame_id', None)  # Unified field name
        # print(f"Client {sid} acknowledged frame", data)


# Function to set the shared event loop (called by ImSwitchServer)
def set_shared_event_loop(loop):
    """Set the shared event loop for thread-safe signal emission."""
    global _shared_event_loop
    _shared_event_loop = loop
    print(f"Shared event loop set: {loop}")


# Function to get the socket app for mounting on FastAPI
def get_socket_app():
    """Returns the Socket.IO ASGI app to be mounted on FastAPI."""
    return socket_app
