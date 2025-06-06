
import os
import subprocess
import threading
import signal
import time
from imswitch import __jupyter_port__
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
    notebookp = subprocess.Popen([notebook_executable,
                            "--port=%s" % port,
                            "--allow-root",
                            "--no-browser",
                            "--ip=0.0.0.0",
                            "--config=%s" % configfile,
                            "--notebook-dir=%s" % directory
                            ], bufsize=1, stderr=subprocess.PIPE)
    print("STarting jupyter with: %s" % notebookp.args)
    print("Waiting for server to start...")
    webaddr = None
    time0 = time.time()
    while webaddr is None:
        line = notebookp.stderr.readline().decode('utf-8').strip()
        log(line)
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

