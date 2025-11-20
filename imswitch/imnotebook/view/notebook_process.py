
import os
import subprocess
import threading
import signal
import time
from imswitch import __jupyter_port__, WITH_KERNEL
from .logger import log

_process = None
_monitor = None
_webaddr = None


def testnotebook(notebook_executable="jupyter-notebook"):
    # if the notebook executable is not found, return False
    return 0 == os.system("%s --version" % notebook_executable)

def startnotebook(notebook_executable="jupyter-lab", port=__jupyter_port__, directory='',
                  configfile=os.path.join(os.path.dirname(__file__), 'jupyter_notebook_config.py')):
    global _process, _monitor, _webaddr
    if _process is not None:
        raise ValueError("Cannot start jupyter lab: one is already running in this module")
    print("Starting Jupyter lab process")
    print("Notebook executable: %s" % notebook_executable)
    # check if the notebook executable is available
    if not testnotebook(notebook_executable):
        print("Notebook executable not found")
    # it is necessary to redirect all 3 outputs or .app does not open
    if 0 < 1:
        if WITH_KERNEL:
            notebookp = subprocess.Popen([notebook_executable,
                                    "--port=%s" % port,
                                    "--allow-root",
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
                                    "--no-browser",
                                    "--ip=0.0.0.0",
                                    "--config=%s" % configfile,
                                    "--notebook-dir=%s" % directory,
                                    ], bufsize=1, stderr=subprocess.PIPE)
            
        print("Starting jupyter with: %s" % " ".join(notebookp.args))
        # concat the string to have the full terminal command 
        print("Waiting for server to start...")
        webaddr = None
        time0 = time.time()
        while webaddr is None:
            line = notebookp.stderr.readline().decode('utf-8').strip()
            log(line)
            # FIXME(ethan): this logic is totally broken and doesn't work at all in a Docker
            # container! If we decide to start the Jupyter Lab server in a separate Docker
            # container, then it may be better to just pass in the address for accessing the server
            # via an env var from the Compose file in the rpi-imswitch-os pallet. In the meantime,
            # if we could start the Jupyter Lab server via our uvicorn server, that would be a
            # viable way to always get a correct web address. Because the ImSwitch server may be
            # accessed via multiple different domain names (e.g. a tailscale domain name, a mDNS
            # name, etc.), we cannot statically know a URL for accessing Jupyter Lab which would
            # work for everyone accessing the ImSwitch server's backend API. Instead, at best the
            # server could return a relative path (relative to the URL used for accessing the API).
            # For example, if the API endpoint for querying the Jupyter Lab server's URL is accessed
            # at `foobar.local:8001/jupyternotebookurl`, then the API should return something like
            # `/jupyter-lab` as the URL which the client can use to access Jupyter Lab; then the
            # client should interpret that as the path under the same protocol+hostname+port
            # combination as what it used to access the API.
            if "http://" in line:
                start = line.find("http://")
                # end = line.find("/", start+len("http://")) new notebook
                # needs a token which is at the end of the line
                # replace hostname.local with localhost
                webaddr = line[start:]
                if ".local" in webaddr:
                    # replace hostname.local with localhost # TODO: Not good!
                    webaddr = "http://localhost"+webaddr.split(".local")[1]
                    break

            if time.time() - time0 > 10:
                print("Timeout waiting for server to start")
                return None
        print("Server found at %s, migrating monitoring to listener thread" % webaddr)

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
                                    "--no-browser",
                                    "--ip=0.0.0.0",
                                    "--config=%s" % configfile,
                                    "--notebook-dir=%s" % directory,
                                    ]))
    return "http://localhost:%s" % port


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

