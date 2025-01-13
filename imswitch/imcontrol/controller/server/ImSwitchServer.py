import threading
import Pyro5.server
from Pyro5.api import expose
from imswitch.imcommon.framework import Worker
from imswitch.imcommon.model import dirtools, initLogger
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from pydantic import BaseModel
from typing import List
import os
import shutil
from fastapi.responses import FileResponse
import zipfile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import imswitch
import uvicorn
from functools import wraps
import os
import socket
import os

from imswitch import IS_HEADLESS, __ssl__, __httpport__

import socket
from fastapi.middleware.cors import CORSMiddleware
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import threading
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

try:
    pass
#    from arkitekt_next import easy
except ImportError:
    print("Arkitekt not found")

 
PORT = __httpport__
IS_SSL = __ssl__

_baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(imswitch.__file__)), '_data')
static_dir = os.path.join(_baseDataFilesDir,  'static')
imswitchapp_dir = os.path.join(_baseDataFilesDir,  'static', 'imswitch')
images_dir =  os.path.join(_baseDataFilesDir, 'images')
app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=static_dir), name="static")  # serve static files such as the swagger UI
app.mount("/imswitch", StaticFiles(directory=imswitchapp_dir), name="imswitch") # serve react app
app.mount("/images", StaticFiles(directory=images_dir), name="images") # serve images for GUI


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

# Base upload directory
BASE_DIR = dirtools.UserFileDirs.Data
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# Pydantic Model for folder creation
class CreateFolderRequest(BaseModel):
    name: str
    parentId: str | None = None  # Optional parent folder

# 📁 Create a Folder
@app.post("/folder")
def create_folder(request: CreateFolderRequest):
    """
    Create a folder using JSON payload.
    """
    # Resolve folder path
    parent_path = request.parentId or ""
    folder_path = os.path.join(BASE_DIR, parent_path, request.name)

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
import os

def list_items(base_path: str) -> List[dict]:
    """
    List files and directories under the given base path.
    Appends '/' at the start of paths and generates preview URLs.
    """
    items = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs + files:
            # Full path and relative path
            full_path = os.path.join(root, name)
            relative_path = os.path.relpath(full_path, BASE_DIR)

            # Add '/' at the start of the path
            formatted_path = f"/{relative_path.replace(os.path.sep, '/')}"  

            # Generate file preview URL if it's a file
            preview_url = None
            if os.path.isfile(full_path):  # Skip unsupported files
                preview_url = "/preview"+formatted_path

            # Append file/folder details to the list
            items.append({
                "name": name,
                "isDirectory": os.path.isdir(full_path),
                "path": formatted_path,  # Path starts with '/'
                "size": os.path.getsize(full_path) if os.path.isfile(full_path) else None,
                "filePreviewPath": preview_url  # Optional preview URL for supported files
            })
    return items


@app.get("/preview/{file_path:path}")
def preview_file(file_path: str):
    """
    Provides file previews by serving the file from disk.
    - `file_path` is the relative path to the file within BASE_DIR.
    """
    # Resolve the absolute file path
    absolute_path = BASE_DIR / file_path

    # Check if the file exists and is a file
    if not absolute_path.exists() or not absolute_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Optionally: Prevent serving unsupported file types
    unsupported_extensions = [".js"]
    if absolute_path.suffix in unsupported_extensions:
        raise HTTPException(status_code=400, detail="File type not supported for preview")

    # Serve the file
    return FileResponse(absolute_path, filename=absolute_path.name)

# 📂 Get All Files/Folders
@app.get("/")
def get_items(path: str = ""):
    directory = os.path.join(BASE_DIR, path)
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail="Path not found")
    return list_items(directory)


# ⬆️ Upload a File

@app.post("/upload")
def upload_file(file: UploadFile = File(...), target_path: Optional[str] = Form("")):
    """
    Upload a file to the specified target directory.
    - `file`: The file being uploaded.
    - `target_path`: The relative path where the file should be uploaded.
    """
    # Resolve target directory
    upload_dir = BASE_DIR / target_path
    upload_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Save the uploaded file
    file_location = upload_dir / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"File '{file.filename}' uploaded successfully", "path": str(file_location)}

# 📋 Copy File(s) or Folder(s)
@app.post("/copy")
def copy_item(source: str = Form(...), destination: str = Form(...)):
    src = BASE_DIR / source
    dest = BASE_DIR / destination / src.name
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
@app.put("/move")
def move_item(source: str = Form(...), destination: str = Form(...)):
    src = BASE_DIR / source
    dest = BASE_DIR / destination / src.name
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    shutil.move(src, dest)
    return {"message": "Item moved successfully", "destination": str(dest)}


# ✏️ Rename a File or Folder
@app.patch("/rename")
def rename_item(source: str = Form(...), new_name: str = Form(...)):
    src = BASE_DIR / source
    new_path = src.parent / new_name
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    src.rename(new_path)
    return {"message": "Item renamed successfully", "new_path": str(new_path)}


# 🗑️ Delete File(s) or Folder(s)
@app.delete("/")
def delete_item(paths: List[str]):
    for path in paths:
        target = BASE_DIR / path
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path '{path}' not found")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return {"message": "Item(s) deleted successfully"}


# ⬇️ Download File(s) or Folder(s)
@app.get("/download//{path:path}")
def download_file(path: str):
    target = os.path.join(BASE_DIR, path.lstrip("/"))
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




class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.server = None

    def run(self):
        try:
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=PORT,
                ssl_keyfile=os.path.join(_baseDataFilesDir, "ssl", "key.pem") if IS_SSL else None,
                ssl_certfile=os.path.join(_baseDataFilesDir, "ssl", "cert.pem") if IS_SSL else None
            )
            self.server = uvicorn.Server(config)
            self.server.run()
        except Exception as e:
            print(f"Couldn't start server: {e}")

    def stop(self):
        if self.server:
            self.server.should_exit = True
            self.server.lifespan.shutdown()
            print("Server is stopping...")
class ImSwitchServer(Worker):

    def __init__(self, api, setupInfo):
        super().__init__()

        self._api = api
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

        # To operate remotely we need to provide https
        # openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
        # uvicorn your_fastapi_app:app --host 0.0.0.0 --port 8001 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem

        # Create and start the server thread
        self.server_thread = ServerThread()
        self.server_thread.start()
        self.__logger.debug("Started server with URI -> PYRO:" + self._name + "@" + self._host + ":" + str(self._port))


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

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ImSwitch Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )
        

    @app.get("/")
    def createAPI(self):
        api_dict = self._api._asdict()
        functions = api_dict.keys()

        def includeAPI(str, func):
            if hasattr(func, '_APIAsyncExecution') and func._APIAsyncExecution:
                if hasattr(func, '_APIRequestType') and func._APIRequestType == "POST":
                    @app.post(str)
                    @wraps(func)
                    async def wrapper(*args, **kwargs):
                        return await func(*args, **kwargs)
                else:
                    @app.get(str) # TODO: Perhaps we want POST instead?
                    @wraps(func)
                    async def wrapper(*args, **kwargs):
                        import importlib #importlib.reload(my_module)
                        return await func(*args, **kwargs) # sometimes we need to return a future 
            else:
                if hasattr(func, '_APIRequestType') and func._APIRequestType == "POST":
                    @app.post(str)
                    @wraps(func)
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
                else:
                    @app.get(str) # TODO: Perhaps we want POST instead?
                    @wraps(func)
                    #@register
                    async def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
            return wrapper

        def includePyro(func):
            @expose
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        for f in functions:
            func = api_dict[f]
            if hasattr(func, 'module'):
                module = func.module
            else:
                module = func.__module__.split('.')[-1]
            self.func = includePyro(includeAPI("/"+module+"/"+f, func))



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
