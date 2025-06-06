import inspect
import asyncio
from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Mutex, Signal, SignalInterface
from importlib.resources import files
from fastapi.staticfiles import StaticFiles


class APIExport:
    """ Decorator for methods that should be exported to API. """

    def __init__(self, *, runOnUIThread=False, asyncExecution=False, requestType="GET"):
        self._APIExport = True
        self._APIRunOnUIThread = runOnUIThread
        self._APIAsyncExecution = asyncExecution
        self._APIRequestType = requestType

    def __call__(self, func):
        func._APIExport = self._APIExport
        func._APIRunOnUIThread = self._APIRunOnUIThread
        func._APIAsyncExecution = self._APIAsyncExecution
        func._APIRequestType = self._APIRequestType
        return func


class UIExport:
    """
    Attach a small manifest to a controller class
    and remember where its built widget‑bundle lives.
    """
    def __init__(self, *,
                 path:str,                 # directory that contains remoteEntry.js
                 name:str,
                 icon:str,
    ):
        self._UIExport = True
        self._UIPath   = path
        self._UIName   = name
        self._UIIcon   = icon


    def __call__(self, cls):
        cls._UIExport = self._UIExport
        cls._ui_meta  = {
            "path"   : self._UIPath,
            "name"   : self._UIName,
            "icon"   : self._UIIcon,
        }
        return cls

def generateUI(widgetClassList, *, missingAttributeErrorMsg=None):
    """ Generates a UI from UIExport-decorated classes in the object. Must be
    called from the main thread. """

    from imswitch.imcommon.model import pythontools

    exportedFuncs = {}
    for widgetClass in widgetClassList.values():
        # list comes from the UIExport decorator and contains
        # the path to the widget, the name of the widget and
        # the class itself
        widgetClassName = widgetClass[0]
        widgetModule = widgetClass[1]
        widgetClassObj = widgetClass[2]
        if widgetClassObj is None:
            continue
        for subObjName in dir(widgetClassObj):
            subObj = getattr(widgetClassObj, subObjName)
            if not callable(subObj):
                continue
            if not hasattr(subObj, '_UIExport') or not subObj._UIExport:
                continue
            if subObjName in exportedFuncs:
                raise NameError(f'UI method name "{widgetClassName}" is already in use')
            exportedFuncs[subObjName] = subObj

    return pythontools.dictToROClass(exportedFuncs,
                                     missingAttributeErrorMsg=missingAttributeErrorMsg)

def generateAPI(objs, *, missingAttributeErrorMsg=None):
    """ Generates an API from APIExport-decorated methods in the objects in the
    passed array objs. Must be called from the main thread. """

    from imswitch.imcommon.model import pythontools

    exportedFuncs = {}
    for obj in objs:
        for subObjName in dir(obj):
            subObj = getattr(obj, subObjName)
            if not callable(subObj):
                continue

            if not hasattr(subObj, '_APIExport') or not subObj._APIExport:
                continue

            if subObjName in exportedFuncs:
                raise NameError(f'API method name "{subObjName}" is already in use')

            runOnUIThread = hasattr(subObj, '_APIRunOnUIThread') and subObj._APIRunOnUIThread

            if runOnUIThread and not IS_HEADLESS:
                wrapper = _UIThreadExecWrapper(subObj)
                exportedFuncs[subObjName] = wrapper
                wrapper.module = subObj.__module__.split('.')[-1]
            else:
                exportedFuncs[subObjName] = subObj

    return pythontools.dictToROClass(exportedFuncs,
                                     missingAttributeErrorMsg=missingAttributeErrorMsg)


class _UIThreadExecWrapper(SignalInterface):
    """ Wrapper for executing the specified function on the UI thread. """

    wrappingSignal = Signal()

    def __init__(self, apiFunc):
        super().__init__()

        self.__name__ = apiFunc.__name__
        self.__signature__ = inspect.signature(apiFunc)
        self.__doc__ = apiFunc.__doc__

        self._apiFunc = apiFunc
        self._execMutex = Mutex()
        self.wrappingSignal.connect(self._apiCall)

    def __call__(self, *args, **kwargs):
        self._execMutex.lock()
        self._args = args
        self._kwargs = kwargs
        self.wrappingSignal.emit()

    def _apiCall(self):
        try:
            if asyncio.iscoroutinefunction(self._apiFunc):
                self._execAsync()
            else:
                self._apiFunc(*self._args, **self._kwargs)
        finally:
            self._execMutex.unlock()

    async def _execAsync(self):
        await self._apiFunc(*self._args, **self._kwargs)


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
