
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
    
    # Check for embedded kernel and install kernel spec if available
    embedded_kernel_info = None
    try:
        import sys
        imswitch_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if imswitch_path not in sys.path:
            sys.path.insert(0, imswitch_path)
        
        from imswitch.imcommon.kernel_state import get_embedded_kernel_connection_file, is_embedded_kernel_running
        from imswitch.imcommon.embedded_kernel_spec import ensure_kernel_spec_available
        
        if is_embedded_kernel_running():
            connection_file = get_embedded_kernel_connection_file()
            if connection_file and os.path.exists(connection_file):
                embedded_kernel_info = {
                    'connection_file': connection_file,
                    'filename': os.path.basename(connection_file)
                }
                print("=" * 60)
                print("EMBEDDED KERNEL DETECTED")
                print("=" * 60)
                print(f"ImSwitch embedded kernel is running")
                print(f"Connection file: {connection_file}")
                print("")
                
                # Try to install kernel spec for notebook interface
                if ensure_kernel_spec_available():
                    print("✅ ImSwitch kernel installed successfully!")
                    print("   Look for 'ImSwitch (Live Connection)' in the notebook kernel list")
                    print("")
                
                print("Alternative connection methods:")
                print("1. From Jupyter console (terminal):")
                print(f"   jupyter console --existing {embedded_kernel_info['filename']}")
                print("")
                print("2. From notebook cells:")
                print("   Use the helper notebook: connect_to_imswitch_kernel.ipynb")
                print("")
                print("Available ImSwitch objects in embedded kernel:")
                print("   - detectorsManager, lasersManager, stageManager, etc.")
                print("   - master_controller, app, mainView, config")
                print("=" * 60)
                
    except Exception as e:
        print(f"Warning: Could not check for embedded kernel: {e}")
        import traceback
        traceback.print_exc()
    
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

