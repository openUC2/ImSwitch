#!/usr/bin/env python
#
# Original espota.py by Ivan Grokhotkov:
# https://gist.github.com/igrr/d35ab8446922179dc58c
#
# Modified since 2015-09-18 from Pascal Gollor (https://github.com/pgollor)
# Modified since 2015-11-09 from Hristo Gochkov (https://github.com/me-no-dev)
# Modified since 2016-01-03 from Matthew O'Gorman (https://githumb.com/mogorman)
# Modified 2025 to be used as a Python module in ImSwitch
#
# This module handles OTA (Over-The-Air) updates for ESP32 devices
#

from __future__ import print_function
import socket
import sys
import os
import logging
import hashlib
import random

# Commands
FLASH = 0
SPIFFS = 100
AUTH = 200


def update_progress(progress, show_progress_bar=False, logger=None):
    """
    Displays or updates a console progress bar.
    
    :param progress: Float between 0 and 1. Any int will be converted to a float.
                     A value under 0 represents a 'halt'.
                     A value at 1 or bigger represents 100%
    :param show_progress_bar: If True, shows a progress bar, otherwise shows dots
    :param logger: Optional logger instance for logging progress
    """
    if show_progress_bar:
        barLength = 60  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rUploading: [{0}] {1}% {2}".format(
            "=" * block + " " * (barLength - block), int(progress * 100), status
        )
        sys.stderr.write(text)
        sys.stderr.flush()
    else:
        sys.stderr.write('.')
        sys.stderr.flush()
    
    if logger:
        logger.debug(f"Upload progress: {int(progress * 100)}%")


def upload_ota(
    esp_ip,
    firmware_path,
    esp_port=3232,
    host_ip="0.0.0.0",
    host_port=3333,
    password="",
    spiffs=False,
    timeout=10,
    show_progress=False,
    logger=None,
    progress_callback=None
):
    """
    Upload firmware to ESP32 via OTA.
    
    :param esp_ip: IP address of the ESP32 device
    :param firmware_path: Path to the firmware binary file
    :param esp_port: ESP32 OTA port (default: 3232)
    :param host_ip: Host IP address (default: 0.0.0.0)
    :param host_port: Host server OTA port (default: random 10000-60000)
    :param password: Authentication password (optional)
    :param spiffs: True if uploading SPIFFS image (default: False)
    :param timeout: Timeout in seconds to wait for ESP32 response (default: 10)
    :param show_progress: Show progress bar (default: False)
    :param logger: Optional logger instance
    :param progress_callback: Optional callback function(percent: int) for progress updates
    :return: 0 on success, 1 on failure
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if host_port is None:
        host_port = random.randint(10000, 60000)
    
    command = SPIFFS if spiffs else FLASH
    
    return serve(
        remoteAddr=esp_ip,
        localAddr=host_ip,
        remotePort=esp_port,
        localPort=host_port,
        password=password,
        filename=firmware_path,
        command=command,
        timeout=timeout,
        show_progress=show_progress,
        logger=logger,
        progress_callback=progress_callback
    )


def serve(remoteAddr, localAddr, remotePort, localPort, password, filename, 
          command=FLASH, timeout=10, show_progress=False, logger=None, 
          progress_callback=None):
    """
    Internal function to handle the OTA upload process.
    
    :param remoteAddr: Remote ESP32 IP address
    :param localAddr: Local host IP address
    :param remotePort: Remote ESP32 port
    :param localPort: Local host port
    :param password: Authentication password
    :param filename: Path to firmware file
    :param command: FLASH or SPIFFS command
    :param timeout: Timeout in seconds
    :param show_progress: Show progress bar
    :param logger: Logger instance
    :param progress_callback: Callback function for progress updates
    :return: 0 on success, 1 on failure
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (localAddr, localPort)
    logger.info(f'Starting on {server_address[0]}:{server_address[1]}')
    
    try:
        sock.bind(server_address)
        sock.listen(1)
    except Exception as e:
        logger.error(f"Listen Failed: {e}")
        return 1

    content_size = os.path.getsize(filename)
    f = open(filename, 'rb')
    file_md5 = hashlib.md5(f.read()).hexdigest()
    f.close()
    logger.info(f'Upload size: {content_size}')
    message = '%d %d %d %s\n' % (command, localPort, content_size, file_md5)

    # Wait for a connection
    inv_trys = 0
    data = ''
    msg = f'Sending invitation to {remoteAddr} '
    sys.stderr.write(msg)
    sys.stderr.flush()
    
    while inv_trys < 10:
        inv_trys += 1
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        remote_address = (remoteAddr, int(remotePort))
        
        try:
            sent = sock2.sendto(message.encode(), remote_address)
        except Exception as e:
            sys.stderr.write('failed\n')
            sys.stderr.flush()
            sock2.close()
            logger.error(f'Host {remoteAddr} Not Found: {e}')
            return 1
        
        sock2.settimeout(timeout)
        try:
            data = sock2.recv(37).decode()
            break
        except:
            sys.stderr.write('.')
            sys.stderr.flush()
            sock2.close()
    
    sys.stderr.write('\n')
    sys.stderr.flush()
    
    if inv_trys == 10:
        logger.error('No response from the ESP')
        return 1
    
    if data != "OK":
        if data.startswith('AUTH'):
            nonce = data.split()[1]
            cnonce_text = '%s%u%s%s' % (filename, content_size, file_md5, remoteAddr)
            cnonce = hashlib.md5(cnonce_text.encode()).hexdigest()
            passmd5 = hashlib.md5(password.encode()).hexdigest()
            result_text = '%s:%s:%s' % (passmd5, nonce, cnonce)
            result = hashlib.md5(result_text.encode()).hexdigest()
            sys.stderr.write('Authenticating...')
            sys.stderr.flush()
            message = '%d %s %s\n' % (AUTH, cnonce, result)
            sock2.sendto(message.encode(), remote_address)
            sock2.settimeout(10)
            
            try:
                data = sock2.recv(32).decode()
            except:
                sys.stderr.write('FAIL\n')
                logger.error('No Answer to our Authentication')
                sock2.close()
                return 1
            
            if data != "OK":
                sys.stderr.write('FAIL\n')
                logger.error(f'Authentication failed: {data}')
                sock2.close()
                return 1
            sys.stderr.write('OK\n')
        else:
            logger.error(f'Bad Answer: {data}')
            sock2.close()
            return 1
    
    sock2.close()

    logger.info('Waiting for device...')
    try:
        sock.settimeout(10)
        connection, client_address = sock.accept()
        sock.settimeout(None)
        connection.settimeout(None)
    except Exception as e:
        logger.error(f'No response from device: {e}')
        sock.close()
        return 1
    
    try:
        f = open(filename, "rb")
        if show_progress:
            update_progress(0, show_progress, logger)
        else:
            sys.stderr.write('Uploading')
            sys.stderr.flush()
        
        offset = 0
        last_progress = 0
        
        while True:
            chunk = f.read(1024)
            if not chunk:
                break
            offset += len(chunk)
            progress = offset / float(content_size)
            
            update_progress(progress, show_progress, logger)
            
            # Call progress callback if provided
            if progress_callback:
                current_percent = int(progress * 100)
                if current_percent >= last_progress + 10:  # Report every 10%
                    try:
                        progress_callback(current_percent)
                        last_progress = current_percent
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
            
            connection.settimeout(10)
            try:
                connection.sendall(chunk)
                res = connection.recv(10)
                lastResponseContainedOK = 'OK' in res.decode()
            except Exception as e:
                sys.stderr.write('\n')
                logger.error(f'Error Uploading: {e}')
                connection.close()
                f.close()
                sock.close()
                return 1

        if lastResponseContainedOK:
            logger.info('Success')
            connection.close()
            f.close()
            sock.close()
            return 0

        sys.stderr.write('\n')
        logger.info('Waiting for result...')
        try:
            count = 0
            while True:
                count = count + 1
                connection.settimeout(60)
                data = connection.recv(32).decode()
                logger.info(f'Result: {data}')

                if "OK" in data:
                    logger.info('Success')
                    connection.close()
                    f.close()
                    sock.close()
                    return 0
                if count == 5:
                    logger.error('Error response from device')
                    connection.close()
                    f.close()
                    sock.close()
                    return 1
        except Exception as e:
            logger.error(f'No Result: {e}')
            connection.close()
            f.close()
            sock.close()
            return 1

    finally:
        connection.close()
        f.close()

    sock.close()
    return 1
