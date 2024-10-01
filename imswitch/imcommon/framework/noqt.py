from typing import TYPE_CHECKING, Any, List
from psygnal import SignalInstance, emit_queued
import psygnal

from functools import lru_cache
import asyncio
import threading
import imswitch.imcommon.framework.base as abstract

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from typing import Tuple, Callable, Any, Union

class Mutex(abstract.Mutex):
    """ Wrapper around the `threading.Lock` class. 
    """
    def __init__(self) -> None:
        self.__lock = threading.Lock()
    
    def lock(self) -> None:
        self.__lock.acquire()
    
    def unlock(self) -> None:
        self.__lock.release()



app = FastAPI()

# List to keep track of connected clients
connected_clients: List[WebSocket] = []

class FastAPIWebSocketManager:
    """Manages connected WebSocket clients and broadcasts messages."""
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Add a new client to the list and accept the connection."""
        await websocket.accept()
        self.clients.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove the client from the list when they disconnect."""
        self.clients.remove(websocket)

    async def broadcast(self, message: str):
        """Send a message to all connected clients."""
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception as e:
                print(f"Error sending message to client: {e}")

# Instantiate the WebSocket manager
ws_manager = FastAPIWebSocketManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; no need to receive from client in this case
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


class SignalInstance(psygnal.SignalInstance):
    async def emit(
        self, *args: Any, check_nargs: bool = False, check_types: bool = False
    ) -> None:
        super().emit(*args, check_nargs=check_nargs, check_types=check_types)
        print("Signal emitted:", self, args)
        # Send data to all connected WebSocket clients
        if connected_clients:
            message = f"Signal emitted with args: {args}"
            await ws_manager.broadcast(message)
        # add your custom websocket code here if you want to emit the signal to a
        # websocket AFTER the signal has been emitted to all python listeners

    def _run_emit_loop(self, args: tuple[Any, ...]) -> None:
        # add your custom websocket code here if you want to emit the signal to a
        # websocket BEFORE the signal has been emitted to all python listeners
        return super()._run_emit_loop(args)


class Signal(psygnal.Signal):      
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  
        self._signal_instance_class = SignalInstance
        self._info = "ImSwitch signal"
        

    def connect(self, func: 'Union[Callable, abstract.Signal]') -> None:
        if isinstance(func, abstract.Signal):
            if any([t1 != t2 for t1, t2 in zip(self.types, func.types)]):
                raise TypeError(f"Source and destination must have the same signature. Source signature: {self.types}, destination signature: {func.types}")
            func = func.emit
        super().connect(func)
    
    def disconnect(self, func: 'Union[Callable, abstract.Signal, None]' = None) -> None:
        if func is None:
            super().disconnect()
        super().disconnect(func)
    
    def emit(self, *args) -> None:        
        # super().emit(*args)
        instance = self._signal_instance_class(*args)
        asyncio.create_task(instance.emit(*args))
    
    @property
    @lru_cache
    def types(self) -> 'Tuple[type, ...]':
        return tuple([param.annotation for param in self._signature.parameters.values()])
    
    @property
    def info(self) -> str:
        return self._info

class Worker(abstract.Worker):
    def __init__(self) -> None:
        self._thread = None

    def moveToThread(self, thread : abstract.Thread) -> None:
        self._thread = thread
        thread._worker = self

class Thread(abstract.Thread):
    
    _started = Signal()
    _finished = Signal()
    
    def __init__(self):
        self._thread = None
        self._loop = None
        self._running = threading.Event()
        self._worker : Worker = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            if self._worker is not None:
                # reassign worker to the new thread in case
                # it was moved to another thread before
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
    
    @property
    def started(self) -> Signal:
        return self._started

    @property
    def finished(self) -> Signal:
        return self._finished

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

    async def _run(self):
        await asyncio.sleep(self._interval)
        self.timeout.emit()
        if not self._singleShot:
            self._task = self._loop.create_task(self._run())
    
    @property
    def timeout(self) -> Signal:
        return self._timeout

def threadCount() -> int:
    """ Returns the current number of active threads of this framework.
    
    Returns:
        ``int``: number of active threads
    """
    return threading.active_count()

class FrameworkUtils(abstract.FrameworkUtils):
    @staticmethod
    def processPendingEventsCurrThread():
        emit_queued()