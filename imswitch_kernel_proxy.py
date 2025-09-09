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
import argparse
from jupyter_client import BlockingKernelClient
from ipykernel.kernelbase import Kernel
from ipykernel.kernelapp import IPKernelApp

class ImSwitchKernelProxy(Kernel):
    implementation = 'imswitch_proxy'
    implementation_version = '1.0'
    language = 'python'
    language_version = '3.x'
    banner = "ImSwitch Embedded Kernel Proxy - Connected to Live ImSwitch Instance"

    def __init__(self, connection_file=None, **kwargs):
        super().__init__(**kwargs)
        self.connection_file = connection_file
        self.embedded_client = None
        self.connect_to_embedded_kernel()

    def connect_to_embedded_kernel(self):
        """Connect to the running ImSwitch embedded kernel"""
        try:
            # Use provided connection file or find one
            connection_file = self.connection_file or self.find_embedded_kernel_connection()
            if not connection_file:
                self.log.error("No embedded kernel connection file found")
                return False

            if not os.path.exists(connection_file):
                self.log.error(f"Connection file does not exist: {connection_file}")
                return False

            # Create client
            self.embedded_client = BlockingKernelClient()
            self.embedded_client.load_connection_file(connection_file)
            self.embedded_client.start_channels()
            
            # Test the connection
            self.embedded_client.wait_for_ready(timeout=5)
            
            self.log.info(f"Connected to ImSwitch embedded kernel: {connection_file}")
            return True

        except Exception as e:
            self.log.error(f"Failed to connect to embedded kernel: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            return False

    def find_embedded_kernel_connection(self):
        """Find the embedded kernel connection file"""
        try:
            import jupyter_core.paths
            runtime_dir = jupyter_core.paths.jupyter_runtime_dir()
        except ImportError:
            # Fallback for different platforms
            runtime_dirs = [
                os.path.expanduser('~/.local/share/jupyter/runtime'),
                os.path.expanduser('~/Library/Jupyter/runtime'),  # macOS
                os.path.expanduser('~/.jupyter/runtime')
            ]
            runtime_dir = None
            for d in runtime_dirs:
                if os.path.exists(d):
                    runtime_dir = d
                    break

        if not runtime_dir or not os.path.exists(runtime_dir):
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
            error_msg = ('Not connected to ImSwitch embedded kernel. '
                        'Make sure ImSwitch is running with --with-kernel flag.')
            
            self.send_response(self.iopub_socket, 'stream', {
                'name': 'stderr',
                'text': error_msg + '\n'
            })
            
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'ConnectionError',
                'evalue': 'Not connected to ImSwitch embedded kernel',
                'traceback': [error_msg]
            }

        try:
            # Check if client is still alive
            if not self.embedded_client.is_alive():
                self.log.warning("Embedded kernel client is not alive, attempting reconnect...")
                self.connect_to_embedded_kernel()
                if not self.embedded_client or not self.embedded_client.is_alive():
                    raise Exception("Failed to reconnect to embedded kernel")

            # Execute code in embedded kernel
            msg_id = self.embedded_client.execute(code, silent=silent, store_history=store_history)
            
            # Get execution result with timeout
            reply = self.embedded_client.get_shell_msg(timeout=30)
            
            # Forward any output messages
            while True:
                try:
                    msg = self.embedded_client.get_iopub_msg(timeout=0.1)
                    msg_type = msg['msg_type']
                    content = msg['content']
                    
                    if msg_type == 'stream':
                        self.send_response(self.iopub_socket, 'stream', {
                            'name': content['name'],
                            'text': content['text']
                        })
                    elif msg_type == 'execute_result':
                        self.send_response(self.iopub_socket, 'execute_result', {
                            'execution_count': content['execution_count'],
                            'data': content['data'],
                            'metadata': content.get('metadata', {})
                        })
                    elif msg_type == 'display_data':
                        self.send_response(self.iopub_socket, 'display_data', {
                            'data': content['data'],
                            'metadata': content.get('metadata', {})
                        })
                    elif msg_type == 'error':
                        self.send_response(self.iopub_socket, 'error', {
                            'ename': content['ename'],
                            'evalue': content['evalue'],
                            'traceback': content['traceback']
                        })
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        # Execution finished
                        break
                        
                except Exception:
                    # No more messages or timeout
                    break

            return {
                'status': reply['content']['status'],
                'execution_count': self.execution_count,
            }

        except Exception as e:
            error_msg = f"Error executing code in embedded kernel: {str(e)}"
            self.log.error(error_msg)
            
            self.send_response(self.iopub_socket, 'stream', {
                'name': 'stderr',
                'text': error_msg + '\n'
            })
            
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
            try:
                self.embedded_client.stop_channels()
            except Exception as e:
                self.log.warning(f"Error stopping embedded client: {e}")
        return {'status': 'ok', 'restart': restart}

    def do_complete(self, code, cursor_pos):
        """Handle code completion"""
        if not self.embedded_client:
            return {'matches': [], 'cursor_start': cursor_pos, 'cursor_end': cursor_pos, 'status': 'ok'}
        
        try:
            # Request completion from embedded kernel
            msg_id = self.embedded_client.complete(code, cursor_pos)
            reply = self.embedded_client.get_shell_msg(timeout=5)
            return reply['content']
        except Exception as e:
            self.log.warning(f"Completion failed: {e}")
            return {'matches': [], 'cursor_start': cursor_pos, 'cursor_end': cursor_pos, 'status': 'ok'}

# Custom kernel app to handle connection file argument
class ImSwitchProxyKernelApp(IPKernelApp):
    
    def initialize(self, argv=None):
        # Parse our custom arguments first
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--connection-file', type=str, help='Path to ImSwitch kernel connection file')
        args, remaining = parser.parse_known_args(argv)
        
        # Store connection file for kernel
        self.connection_file = args.connection_file
        
        # Initialize with remaining arguments
        super().initialize(remaining)
    
    def init_kernel(self):
        # Pass connection file to kernel
        self.kernel = ImSwitchKernelProxy(
            parent=self,
            session=self.session,
            iopub_socket=self.iopub_socket,
            stdin_socket=self.stdin_socket,
            control_socket=self.control_socket,
            shell_socket=self.shell_socket,
            connection_file=getattr(self, 'connection_file', None)
        )

if __name__ == '__main__':
    # Launch the kernel
    ImSwitchProxyKernelApp.launch_instance()