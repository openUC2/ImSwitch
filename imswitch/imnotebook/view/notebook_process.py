
import os
import subprocess
import threading
import signal
import time
import socket
from imswitch.config import get_config
from .logger import log

_process = None
_monitor = None
_webaddr = None


def get_server_ip():
    """Get the server's IP address for accessing Jupyter notebook."""
    try:
        # Create a socket to get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def testnotebook(notebook_executable="jupyter-notebook"):
    # if the notebook executable is not found, return False
    return 0 == os.system("%s --version" % notebook_executable)

def startnotebook(notebook_executable="jupyter-lab", port=None, directory='',
                  configfile=os.path.join(os.path.dirname(__file__), 'jupyter_notebook_config.py')):
    global _process, _monitor, _webaddr
    
    # Get config and use jupyter_port if port not specified
    config = get_config()
    if port is None:
        port = config.jupyter_port
    
    if _process is not None:
        raise ValueError("Cannot start jupyter lab: one is already running in this module")
    print("Starting Jupyter lab process")
    print("Notebook executable: %s" % notebook_executable)
    print("Jupyter port: %s" % port)
    
    # check if the notebook executable is available
    if not testnotebook(notebook_executable):
        print("Notebook executable not found")
    # it is necessary to redirect all 3 outputs or .app does not open
    if 0 < 1:
        if config.with_kernel:
            notebookp = subprocess.Popen([notebook_executable,
                                    "--port=%s" % port,
                                    "--allow-root",
                                    "--IdentityProvider.token=",
                                    "--ServerApp.base_url=/jupyter/",
                                    "--ServerApp.password=",
                                    "--no-browser",
                                    "--ip=0.0.0.0",
                                    "--config=%s" % configfile,
                                    "--notebook-dir=%s" % directory,
                                    "--KernelProvisionerFactory.default_provisioner_name=imswitch-provisioner"
                                    ], bufsize=1, stderr=subprocess.PIPE)
        else:
            notebookp = subprocess.Popen([notebook_executable,
                                    "--port=%s" % port,
                                    "--allow-root",
                                    "--IdentityProvider.token=",
                                    "--ServerApp.base_url=/jupyter/",
                                    "--ServerApp.password=",
                                    "--no-browser",
                                    "--ip=0.0.0.0",
                                    "--config=%s" % configfile,
                                    "--notebook-dir=%s" % directory,
                                    ], bufsize=1, stderr=subprocess.PIPE)

        print("Starting jupyter with: %s" % " ".join(notebookp.args))
        print("Waiting for server to start...")
        
        time0 = time.time()
        server_started = False
        
        while not server_started:
            line = notebookp.stderr.readline().decode('utf-8').strip()
            log(line)
            
            # Check if server has started by looking for the startup message
            if "http://" in line or "is running at" in line.lower():
                server_started = True
                break

            if time.time() - time0 > 10:
                print("Timeout waiting for server to start")
                return None
        
        # Construct the proper web address using server IP and configured port
        server_ip = get_server_ip()
        webaddr = f"http://{server_ip}:{port}/jupyter/"
        print(f"Jupyter Lab server started at {webaddr}")

        # pass monitoring over to child thread
        def process_thread_pipe(process):
            while process.poll() is None:  # while process is still alive
                log(process.stderr.readline().decode('utf-8').strip())
        notebookmonitor = threading.Thread(name="Notebook Monitor", target=process_thread_pipe,
                                        args=(notebookp,), daemon=True)
        notebookmonitor.start()
        _process = notebookp
        _monitor = notebookmonitor
        _webaddr = webaddr
        return webaddr
    else:
        print("Starting jupyter with: %s" % " ".join([notebook_executable,
                                    "--port=%s" % port,
                                    "--allow-root",
                                    "--ServerApp.base_url=/jupyter/",
                                    "--no-browser",
                                    "--ip=0.0.0.0",
                                    "--config=%s" % configfile,
                                    "--notebook-dir=%s" % directory,
                                    ]))
    return f"http://{get_server_ip()}:{port}/jupyter/"


def stopnotebook():
    global _process, _monitor, _webaddr
    if _process is None:
        return
    log("Sending interrupt signal to jupyter-notebook")
    _process.send_signal(signal.SIGINT)
    try:
        log("Waiting for jupyter to exit...")
        time.sleep(1)
        _process.send_signal(signal.SIGINT)
        _process.wait(10)
        log("Final output:")
        log(_process.communicate())

    except subprocess.TimeoutExpired:
        log("control c timed out, killing")
        _process.kill()

    _process = None
    _monitor = None
    _webaddr = None
