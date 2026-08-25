from abc import ABCMeta
from qtpy import QtCore

import imswitch.imcommon.framework.base as base

# Check which Qt backend is active and pick the right deleted-object detector
import qtpy as _qtpy
if _qtpy.API_NAME in ('PySide6', 'PySide2'):
    try:
        import shiboken6 as _shibo
        def _is_deleted(obj):
            return not _shibo.isValid(obj)
    except ImportError:
        def _is_deleted(obj):
            return False
else:
    try:
        from PyQt5 import sip as _sip
        def _is_deleted(obj):
            try:
                _sip.unwrapinstance(obj)
            except RuntimeError:
                return True
            return False
    except ImportError:
        def _is_deleted(obj):
            return False


class QObjectMeta(type(QtCore.QObject), ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # sip/Shiboken metaclass doesn't call ABCMeta.__new__, so _abc_impl
        # is never set — fix it explicitly so issubclass() works correctly.
        if not hasattr(cls, '_abc_impl'):
            from _abc import _abc_init
            _abc_init(cls)
        return cls


class Mutex(QtCore.QMutex, base.Mutex, metaclass=QObjectMeta):
    pass


class Signal(base.Signal):
    def __new__(cls, *argtypes) -> base.Signal:
        return QtCore.Signal(*argtypes)


class SignalInterface(QtCore.QObject, base.SignalInterface, metaclass=QObjectMeta):
    pass


class Thread(QtCore.QThread, base.Thread, metaclass=QObjectMeta):
    def quit(self) -> None:
        if not _is_deleted(self):
            super().quit()

    def wait(self) -> None:
        if not _is_deleted(self):
            super().wait()

    def __isWrappedCObjDeleted(self) -> bool:
        return _is_deleted(self)


class Timer(QtCore.QTimer, base.Timer, metaclass=QObjectMeta):
    pass


class Worker(QtCore.QObject, base.Worker, metaclass=QObjectMeta):
    pass


class FrameworkUtils(base.FrameworkUtils):
    @staticmethod
    def processPendingEventsCurrThread():
        QtCore.QAbstractEventDispatcher.instance(
            QtCore.QThread.currentThread()
        ).processEvents(QtCore.QEventLoop.AllEvents)
