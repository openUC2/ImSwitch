import threading
from imswitch.imcommon.framework import Worker
from imswitch.imcommon.model import dirtools, initLogger
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from imswitch.imcontrol.model import Options
from imswitch.imcommon.model import ostools
from imswitch.imcontrol.view.guitools import ViewSetupInfo
import dataclasses
from typing import List
import os
import shutil
from fastapi.responses import FileResponse
import zipfile
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Optional
from fastapi import Request
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import imswitch
from functools import wraps
import socket
from typing import Dict
from imswitch import __ssl__, __httpport__, __version__
from imswitch.imcontrol.model import configfiletools
from fastapi.responses import RedirectResponse
import asyncio
from datetime import datetime
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles

# Import Socket.IO app from noqt framework
from imswitch.imcommon.framework.noqt import get_socket_app, set_shared_event_loop

try:
    pass
#    from arkitekt_next import easy
except ImportError:
    print("Arkitekt not found")


PORT = __httpport__
IS_SSL = __ssl__
VERSION = __version__

_baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(imswitch.__file__)), '_data')
static_dir = os.path.join(_baseDataFilesDir,  'static')
imswitchapp_dir = os.path.join(_baseDataFilesDir,  'static', 'imswitch')
images_dir =  os.path.join(_baseDataFilesDir, 'images')
app = FastAPI(root_path="/imswitch", docs_url=None, redoc_url=None)
api_router = APIRouter(prefix="/api")

# Mount Socket.IO app at root path for WebSocket connections
# This allows Socket.IO to handle all socket.io/* paths
socket_app = get_socket_app()
app.mount('/socket.io', socket_app)
print("Socket.IO app mounted at /socket.io")

# Mount static files and other apps
app.mount("/static", StaticFiles(directory=static_dir), name="static")  # serve static files such as the swagger UI
app.mount("/ui", StaticFiles(directory=imswitchapp_dir), name="imswitch") # serve react app
app.mount("/images", StaticFiles(directory=images_dir), name="images") # serve images for GUI
# provide data path via static files
app.mount("/data", StaticFiles(directory=dirtools.UserFileDirs.getValidatedDataPath()), name="data")  # serve user data files
# manifests for the react app
_ui_manifests = []


if IS_SSL:
    app.add_middleware(HTTPSRedirectMiddleware)

origins = [
    "http://localhost:8001",
    "http://localhost:8000",
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''Add Endpoints for Filemanager'''

# Pydantic Model for folder creation
class CreateFolderRequest(BaseModel):
    name: str
    parentId: Optional[str] = None

@api_router.get("/version")
def get_version():
    """
    Returns the current version of the ImSwitch server.
    """
    return {"version": VERSION}


# Storage Management API Endpoints are now handled via APIExport in StorageController
# The controller is automatically initialized by MasterController


# Create a Folder
@api_router.post("/folder")
def create_folder(request: CreateFolderRequest):
    """
    Create a folder using JSON payload.
    """
    # Resolve folder path
    parent_path = request.parentId or ""
    folder_path = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), parent_path, request.name)

    # Check if folder already exists
    if os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Folder already exists")

    # Create folder
    try:
        os.makedirs(folder_path, exist_ok=True)
        return {"message": f"Folder '{request.name}' created successfully", "path": folder_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FileSystemItem(BaseModel):
    name: str
    isDirectory: bool
    path: str
    size: int = None
    mimeType: str = None


# Utility: List files/folders
def list_items(base_path: str) -> List[Dict]:
    items = []

    def scan_directory(path):
        with os.scandir(path) as it:
            for entry in it:
                try:
                    full_path = entry.path
                    rel_path = f"/{os.path.relpath(full_path, dirtools.UserFileDirs.getValidatedDataPath()).replace(os.path.sep, '/')}"
                    preview_url = f"/preview{rel_path}" if entry.is_file() else None
                    stat_info = entry.stat()
                    items.append({
                        "name": entry.name,
                        "isDirectory": entry.is_dir(),
                        "path": rel_path,
                        "size": stat_info.st_size if entry.is_file() else None,
                        "filePreviewPath": preview_url,
                        "modifiedTime": stat_info.st_mtime  # Add modification timestamp
                    })
                    if entry.is_dir():
                        scan_directory(full_path)
                except Exception as e:
                    print(f"Error scanning {entry.path}: {e}")

    scan_directory(base_path)
    # Sort items by modification time (newest first)
    items.sort(key=lambda x: x["modifiedTime"], reverse=True)
    return items



@api_router.get("/FileManager/preview/{file_path:path}")
def preview_file(file_path: str):
    """
    Provides file previews by serving the file from disk.
    - `file_path` is the relative path to the file within dirtools.UserFileDirs.getValidatedDataPath().
    """
    # Resolve the absolute file path
    absolute_path = dirtools.UserFileDirs.getValidatedDataPath() / file_path

    # Check if the file exists and is a file
    if not absolute_path.exists() or not absolute_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Optionally: Prevent serving unsupported file types
    unsupported_extensions = [".js"]
    if absolute_path.suffix in unsupported_extensions:
        raise HTTPException(status_code=400, detail="File type not supported for preview")

    # Serve the file
    return FileResponse(absolute_path, filename=absolute_path.name)

@api_router.get("/FileManager/")
def get_items(path: str = ""):
    directory = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), path)
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail="Path not found")
    return list_items(directory)

@api_router.post("/FileManager/upload")
def upload_file(file: UploadFile = File(...), target_path: Optional[str] = Form("")):
    """
    Upload a file to the specified target directory.
    - `file`: The file being uploaded.
    - `target_path`: The relative path where the file should be uploaded.
    """
    # Resolve target directory
    upload_dir = dirtools.UserFileDirs.getValidatedDataPath() / target_path
    upload_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Save the uploaded file
    file_location = upload_dir / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"File '{file.filename}' uploaded successfully", "path": str(file_location)}

# 📋 Copy File(s) or Folder(s)
@api_router.post("/FileManager/copy")
def copy_item(source: str = Form(...), destination: str = Form(...)):
    src = dirtools.UserFileDirs.getValidatedDataPath() / source
    dest = dirtools.UserFileDirs.getValidatedDataPath() / destination / src.name
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    if dest.exists():
        raise HTTPException(status_code=400, detail="Destination already exists")
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return {"message": "Item copied successfully", "destination": str(dest)}


# 📤 Move File(s) or Folder(s)
@api_router.put("/FileManager/move")
def move_item(source: str = Form(...), destination: str = Form(...)):
    src = dirtools.UserFileDirs.getValidatedDataPath() / source
    dest = dirtools.UserFileDirs.getValidatedDataPath() / destination / src.name
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    shutil.move(src, dest)
    return {"message": "Item moved successfully", "destination": str(dest)}


# ✏️ Rename a File or Folder
@api_router.patch("/FileManager/rename")
def rename_item(source: str = Form(...), new_name: str = Form(...)):
    src = dirtools.UserFileDirs.getValidatedDataPath() / source
    new_path = src.parent / new_name
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    src.rename(new_path)
    return {"message": "Item renamed successfully", "new_path": str(new_path)}


# 🗑️ Delete File(s) or Folder(s)
@api_router.delete("/FileManager")
def delete_item(paths: List[str]):
    for path in paths:
        target = dirtools.UserFileDirs.getValidatedDataPath() / path
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path '{path}' not found")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return {"message": "Item(s) deleted successfully"}


# ⬇️ Download File(s) or Folder(s)
@api_router.get("/FileManager/download/{path:path}")
def download_file(path: str):
    target = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), path.lstrip("/"))
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="File/Folder not found")
    if os.path.isfile(target):
        return FileResponse(target, filename=target)
    # If it's a folder, zip it and send
    # Assuming target is a string representing the path
    zip_path = target + ".zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(target):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, os.path.dirname(target))
                zipf.write(full_path, arcname=arcname)
    return FileResponse(zip_path, filename=os.path.basename(zip_path))


# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Parsing of Request failed. Please check your submission format.",
            "errors": exc.errors()
        },
    )

# Redirect root URL "/" to "/imswitch"
@app.get("/", include_in_schema=False)
async def root_redirect():
    # Comments in English: Redirect to the React app
    return RedirectResponse(url="/ui/index.html")


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.server = None
        # Create a new asyncio event loop for the server
        self._asyncio_loop = asyncio.new_event_loop()
        # Store reference in imswitch module for global access
        imswitch._asyncio_loop_imswitchserver = self._asyncio_loop

    def run(self):
        try:
            # Set the event loop for this thread
            asyncio.set_event_loop(self._asyncio_loop)

            # Configure the shared event loop for signal emission
            set_shared_event_loop(self._asyncio_loop)
            print("Shared event loop configured for Socket.IO and FastAPI")

            # Create Uvicorn config with the shared event loop
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=PORT,
                ssl_keyfile=os.path.join(_baseDataFilesDir, "ssl", "key.pem") if IS_SSL else None,
                ssl_certfile=os.path.join(_baseDataFilesDir, "ssl", "cert.pem") if IS_SSL else None,
                loop=self._asyncio_loop,  #loop="none",  # Use "none" to let us manage the loop # TODO: This is not yet complete
                log_level="info"
            )

            # Create server instance
            self.server = uvicorn.Server(config)

            # Run the server using the existing event loop
            self._asyncio_loop.run_until_complete(self.server.serve())
        except Exception as e:
            print(f"Couldn't start server (ImSwitchServer): {e}")
        finally:
            # Clean up the event loop
            try:
                self._asyncio_loop.close()
            except Exception as e:
                print(f"Error closing event loop: {e}")

    def stop(self):
        if self.server:
            self.server.should_exit = True
            # Schedule shutdown in the event loop
            if self._asyncio_loop and self._asyncio_loop.is_running():
                asyncio.run_coroutine_threadsafe(self.server.shutdown(), self._asyncio_loop)
            print("Server is stopping...")
class ImSwitchServer(Worker):

    def __init__(self, api, uiapi, setupInfo):
        super().__init__()

        self._api = api
        self._uiapi = uiapi
        self._name = setupInfo.pyroServerInfo.name
        self._host = setupInfo.pyroServerInfo.host
        self._port = setupInfo.pyroServerInfo.port

        self._paused = False
        self._canceled = False

        self.__logger =  initLogger(self)


    def moveToThread(self, thread) -> None:
        return super().moveToThread(thread)

    def run(self):
        # serve the fastapi
        self.createAPI()
        # Note(ethanjli): all FastAPI path operations must be added before we call `include_router`!
        app.include_router(api_router)

        # To operate remotely we need to provide https
        # openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
        # uvicorn your_fastapi_app:app --host 0.0.0.0 --port 8001 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem

        # Create and start the server thread
        self.server_thread = ServerThread()
        self.server_thread.start()
        self.__logger.debug("Started server with URI -> Fastapi:" + self._name + "@" + self._host + ":" + str(self._port))


    def stop(self):
        self.__logger.debug("Stopping ImSwitchServer")
        try:
            self.server_thread.stop()
            #self.server_thread.join()
        except Exception as e:
            self.__logger.error("Couldn't stop server: "+str(e))

    def get_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    @api_router.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html(request: Request):
        root_path = request.scope.get("root_path")
        return get_swagger_ui_html(
            openapi_url=root_path+app.openapi_url,
            title=app.title + " - ImSwitch Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url=root_path+"/static/swagger-ui-bundle.js",
            swagger_css_url=root_path+"/static/swagger-ui.css",
        )

    @api_router.get("/jupyternotebookurl")
    def get_jupyter_notebook_url():
        from imswitch import jupyternotebookurl # import it here to get the one after the notebook has been actually launched
        return {"url": jupyternotebookurl}

    @api_router.get("/plugins")
    def get_plugins():
        """
        Returns a list of available plugins
        """
        plugins = []
        for f in _ui_manifests:
            plugin = f
            plugins.append(plugin)
        return {"plugins": plugins}

    @api_router.get("/hostname")
    def get_hostname():
        """
        Returns the hostname of the server.
        """
        hostname = socket.gethostname()
        return {"hostname": hostname}

    def createAPI(self):
        api_dict = self._api._asdict()
        functions = api_dict.keys()

        def includeAPI(str, func):
            if hasattr(func, '_APIAsyncExecution') and func._APIAsyncExecution:
                if hasattr(func, '_APIRequestType') and func._APIRequestType == "POST":
                    @api_router.post(str)
                    @wraps(func)
                    async def wrapper(*args, **kwargs):
                        return await func(*args, **kwargs)
                else:
                    @api_router.get(str) # TODO: Perhaps we want POST instead?
                    @wraps(func)
                    async def wrapper(*args, **kwargs):
                        return await func(*args, **kwargs) # sometimes we need to return a future
            else:
                if hasattr(func, '_APIRequestType') and func._APIRequestType == "POST":
                    @api_router.post(str)
                    @wraps(func)
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
                else:
                    @api_router.get(str) # TODO: Perhaps we want POST instead?
                    @wraps(func)
                    #@register
                    async def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
            return wrapper

        def includeUIAPI(str, func):
            # based on UIExport decorator, only get is supported


            if hasattr(func, '_UIExport') and func._UIExport:
                @app.get(str)
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
            else:
                @app.get(str)
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
            return wrapper

        # add APIExport decorated functions to the fastAPI
        for f in functions:
            func = api_dict[f]
            if hasattr(func, 'module'):
                module = func.module
            else:
                module = func.__module__.split('.')[-1]
            self.func = includeAPI("/"+module+"/"+f, func)

        # add UIExport decorated functions to the fastAPI under /externUI
        if self._uiapi is None: return # we are on QT mode
        uiapi_dict = self._uiapi._asdict()
        functions = uiapi_dict.keys()
        for f in functions:
            func = uiapi_dict[f]
            if hasattr(func, 'module'):
                module = func.module
            else:
                module = func.__module__.split('.')[-1]
            meta = getattr(func, "_ui_meta", None)
            mount = f"/plugin/{meta['name']}"
            # self.func = includeUIAPI(mount, func)
            _ui_manifests.append({
                "name": meta["name"],
                "icon": meta["icon"],
                "path": meta["path"],
                "exposed": "Widget",
                "scope": "lightsheet_plugin",
                "url": os.path.join(mount,"index.html"),
                "remote": os.path.join(mount,"remoteEntry.js")
            })
            # only if the mount exists:
            self.__logger.debug(f"Mounting {mount} to {os.path.join(meta['path'])}")
            if os.path.exists(os.path.join(meta["path"])):
                app.mount(
                    mount,
                    StaticFiles(directory=os.path.join(meta["path"])),
                    name=meta["name"],
                )

    # The reason why it's still called UC2ConfigController is because we don't want to change the API
    @api_router.get("/UC2ConfigController/returnAvailableSetups")
    def returnAvailableSetups():
        """
        Returns a list of available setups in the config file.
        """
        _, _ = configfiletools.loadOptions()
        setup_list = configfiletools.getSetupList()
        # sort list alphabetically
        setup_list.sort()
        return {"available_setups": setup_list}

    @api_router.get("/UC2ConfigController/getCurrentSetupFilename")
    def getCurrentSetupFilename() -> Dict[str, str]:
        """
        Returns the current setup filename.
        """
        options = imswitch.DEFAULT_SETUP_FILE # configfiletools.loadOptions()
        return {"current_setup": options}


    @api_router.get("/UC2ConfigController/readSetupFile")
    def readSetupFile(setupFileName: str=None) -> dict:
        '''Reads the setup file. If setupFileName is None, reads the current setup file.'''
        if setupFileName is None:
            # get current setup file name
            options, _ = configfiletools.loadOptions()
            setupFileName = options.setupFileName
        if setupFileName.split("/")[-1] not in configfiletools.getSetupList():
            print(f"Setup file {setupFileName} does not exist.")
            return f"Setup file {setupFileName} does not exist."
        mOptions = Options(setupFileName=setupFileName)
        setup_dict = configfiletools.loadSetupInfo(mOptions, ViewSetupInfo)
        return setup_dict.to_dict()

    @api_router.post("/UC2ConfigController/writeNewSetupFile")
    def writeNewSetupFile(setupFileName: str, setupDict: dict, setAsCurrentConfig: bool = True, restart: bool = False, overwrite: bool = False) -> str:
        '''Writes a new setup file. and set as new setup file if needed on next boot.'''
        if setupFileName is None:
            return "No setup file name provided."
        if setupFileName in configfiletools.getSetupList() and not overwrite:
            print(f"Setup file {setupFileName} already exists.")
            return f"Setup file {setupFileName} already exists."
        mOptions = Options(
            setupFileName=setupFileName
        )
        setupInfo = ViewSetupInfo.from_dict(setupDict)
        configfiletools.saveSetupInfo(mOptions, setupInfo)
        if setAsCurrentConfig:
            options, _ = configfiletools.loadOptions()
            options = dataclasses.replace(options, setupFileName=setupFileName)
            configfiletools.saveOptions(options)
        if restart:
            ostools.restartSoftware()

        if restart:
            ostools.restartSoftware(forceConfigFile=setAsCurrentConfig)
        return f"Setup file {setupFileName} written successfully."

    @api_router.get("/UC2ConfigController/setSetupFileName")
    def setSetupFileName(setupFileName: str, restartSoftware: bool=False) -> str:
        '''Sets the setup file name in the options file.'''
        if setupFileName is  None:
            return "No setup file name provided."
        if setupFileName not in configfiletools.getSetupList():
            print(f"Setup file {setupFileName} does not exist.")
            return f"Setup file {setupFileName} does not exist."
        options, _ = configfiletools.loadOptions()
        options = dataclasses.replace(options, setupFileName=setupFileName)
        configfiletools.saveOptions(options)
        if restartSoftware:
            ostools.restartSoftware()
        return f"Setup file {setupFileName} set successfully."

    # Log file management endpoints
    @api_router.get("/LogController/listLogFiles")
    def list_log_files() -> Dict[str, List[Dict[str, str]]]:
        """
        Returns a list of available log files in alphanumerical order.
        """
        from imswitch.imcommon.model import get_log_folder
        log_folder = get_log_folder()

        if not os.path.exists(log_folder):
            return {"log_files": []}

        log_files = []
        for filename in os.listdir(log_folder):
            if filename.endswith('.log'):
                file_path = os.path.join(log_folder, filename)
                file_stat = os.stat(file_path)
                log_files.append({
                    'filename': filename,
                    'size': str(file_stat.st_size),
                    'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    'path': file_path
                })

        # Sort alphanumerically by filename
        log_files.sort(key=lambda x: x['filename'])

        return {"log_files": log_files}

    @api_router.get("/LogController/downloadLogFile")
    def download_log_file(filename: str):
        """
        Download a specific log file.
        """
        from imswitch.imcommon.model import get_log_folder
        log_folder = get_log_folder()

        # Security: prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = os.path.join(log_folder, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Log file not found")

        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail="Not a file")

        return FileResponse(file_path, filename=filename, media_type='text/plain')


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
