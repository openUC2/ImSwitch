#!/usr/bin/env python3
"""
ImSwitch Embedded Kernel Proxy

This script creates a Jupyter kernel that connects to the running ImSwitch embedded kernel.
It can be used as a kernel spec to make the embedded kernel available in Jupyter notebooks.
"""

import sys
import os
import json
import time
from jupyter_client import BlockingKernelClient
from ipykernel.kernelbase import Kernel
from ipykernel.kernelapp import IPKernelApp

class ImSwitchKernelProxy(Kernel):
    implementation = 'imswitch_proxy'
    implementation_version = '1.0'
    language = 'python'
    language_version = '3.x'
    banner = "ImSwitch Embedded Kernel Proxy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedded_client = None
        self.connect_to_embedded_kernel()

    def connect_to_embedded_kernel(self):
        """Connect to the running ImSwitch embedded kernel"""
        try:
            # Find the connection file
            connection_file = self.find_embedded_kernel_connection()
            if not connection_file:
                self.log.error("No embedded kernel connection file found")
                return False

            # Create client
            self.embedded_client = BlockingKernelClient()
            self.embedded_client.load_connection_file(connection_file)
            self.embedded_client.start_channels()
            
            self.log.info(f"Connected to ImSwitch embedded kernel: {connection_file}")
            return True

        except Exception as e:
            self.log.error(f"Failed to connect to embedded kernel: {e}")
            return False

    def find_embedded_kernel_connection(self):
        """Find the embedded kernel connection file"""
        try:
            import jupyter_core.paths
            runtime_dir = jupyter_core.paths.jupyter_runtime_dir()
        except ImportError:
            runtime_dir = os.path.expanduser('~/.local/share/jupyter/runtime')

        if not os.path.exists(runtime_dir):
            return None

        import glob
        kernel_files = glob.glob(os.path.join(runtime_dir, 'kernel-*.json'))
        
        if not kernel_files:
            return None

        # Return the most recently modified one
        return max(kernel_files, key=os.path.getmtime)

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        """Execute code in the embedded kernel"""
        if not self.embedded_client:
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'ConnectionError',
                'evalue': 'Not connected to ImSwitch embedded kernel',
                'traceback': ['Not connected to ImSwitch embedded kernel. Make sure ImSwitch is running with --with-kernel']
            }

        try:
            # Execute code in embedded kernel
            msg_id = self.embedded_client.execute(code, silent=silent, store_history=store_history)
            
            # Get execution result
            reply = self.embedded_client.get_shell_msg(timeout=10)
            
            # Forward any output
            while True:
                try:
                    msg = self.embedded_client.get_iopub_msg(timeout=0.1)
                    if msg['msg_type'] == 'stream':
                        self.send_response(self.iopub_socket, 'stream', {
                            'name': msg['content']['name'],
                            'text': msg['content']['text']
                        })
                    elif msg['msg_type'] == 'execute_result':
                        self.send_response(self.iopub_socket, 'execute_result', {
                            'execution_count': msg['content']['execution_count'],
                            'data': msg['content']['data'],
                            'metadata': msg['content'].get('metadata', {})
                        })
                    elif msg['msg_type'] == 'error':
                        self.send_response(self.iopub_socket, 'error', {
                            'ename': msg['content']['ename'],
                            'evalue': msg['content']['evalue'],
                            'traceback': msg['content']['traceback']
                        })
                except:
                    break

            return {
                'status': reply['content']['status'],
                'execution_count': self.execution_count,
            }

        except Exception as e:
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'ExecutionError',
                'evalue': str(e),
                'traceback': [str(e)]
            }

    def do_shutdown(self, restart):
        """Shutdown the proxy kernel"""
        if self.embedded_client:
            self.embedded_client.stop_channels()
        return {'status': 'ok', 'restart': restart}

if __name__ == '__main__':
    # Launch the kernel
    IPKernelApp.launch_instance(kernel_class=ImSwitchKernelProxy)