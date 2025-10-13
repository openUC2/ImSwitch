import inspect
import logging
import logging.handlers
import weakref
import time
import os
from datetime import datetime
from typing import Optional

try:
    import coloredlogs
    _coloredlogs_available = True
except ImportError:
    _coloredlogs_available = False


baseLogger = logging.getLogger('imswitch')
_file_handler = None
_signal_handler = None
objLoggers = {}


def setup_logging(log_level: str = "INFO", log_to_file: bool = True, 
                 log_folder: Optional[str] = None, config_folder: Optional[str] = None):
    """
    Set up the logging system with configurable log level and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_folder: Path to log folder (if None, uses config_folder/logs)
        config_folder: Path to config folder (used as fallback for log_folder)
    """
    global _file_handler, baseLogger
    
    # Convert log level string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    baseLogger.setLevel(numeric_level)
    
    # Setup console logging with coloredlogs if available
    if _coloredlogs_available:
        coloredlogs.install(level=numeric_level, logger=baseLogger,
                            fmt='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        # Remove existing handlers to avoid duplicate logs
        baseLogger.handlers.clear()
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        baseLogger.addHandler(console_handler)
    
    # Setup file logging if requested
    if log_to_file:
        # Determine log folder
        if log_folder is None:
            if config_folder is None:
                # Use default location - avoid circular import by importing directly
                from imswitch.imcommon.model.dirtools import UserFileDirs
                config_folder = UserFileDirs.Root
            log_folder = os.path.join(config_folder, 'logs')
        
        # Create log folder if it doesn't exist
        os.makedirs(log_folder, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = os.path.join(log_folder, f'imswitch_{timestamp}.log')
        
        # Remove old file handler if it exists
        if _file_handler is not None:
            baseLogger.removeHandler(_file_handler)
            _file_handler.close()
        
        # Create new file handler
        _file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        _file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _file_handler.setFormatter(file_formatter)
        baseLogger.addHandler(_file_handler)
        
        baseLogger.info(f"Logging to file: {log_filename}")
    
    return baseLogger


def get_log_folder(config_folder: Optional[str] = None) -> str:
    """Get the log folder path."""
    if config_folder is None:
        try:
            from imswitch.imcommon.model.dirtools import UserFileDirs
            config_folder = UserFileDirs.Root
        except ImportError:
            # Fallback to home directory if dirtools cannot be imported
            config_folder = os.path.expanduser('~/ImSwitchConfig')
    return os.path.join(config_folder, 'logs')


# Initialize logging with default settings
# This will be reconfigured when the application starts with proper config
if _coloredlogs_available:
    coloredlogs.install(level='INFO', logger=baseLogger,
                        fmt='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
else:
    # Fallback to basic logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


class SignalEmittingHandler(logging.Handler):
    """Custom logging handler that emits log messages via signals."""
    
    def __init__(self, signal=None):
        super().__init__()
        self._signal = signal
    
    def set_signal(self, signal):
        """Set or update the signal to emit log messages to."""
        self._signal = signal
    
    def emit(self, record):
        """Emit a log record via signal."""
        if self._signal is None:
            return
        
        try:
            log_entry = self.format(record)
            # Emit signal with log data
            self._signal.emit({
                'timestamp': record.created,
                'level': record.levelname,
                'name': record.name,
                'message': record.getMessage(),
                'formatted': log_entry
            })
        except Exception:
            self.handleError(record)


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, prefixes, objRef, include_traceback=False, custom_format=None):
        super().__init__(logger, {})
        self.prefixes = prefixes
        self.objRef = objRef
        self.include_traceback = include_traceback
        self.custom_format = custom_format

    def process(self, msg, kwargs):
        processedPrefixes = []
        for prefix in self.prefixes:
            if callable(prefix):
                try:
                    processedPrefixes.append(prefix(self.objRef() if self.objRef else None))
                except Exception:
                    pass
            else:
                processedPrefixes.append(prefix)

        # Apply custom formatting if specified
        if self.custom_format:
            processedMsg = self.custom_format.format(
                prefixes=" -> ".join(processedPrefixes),
                message=msg
            )
        else:
            processedMsg = f'[{" -> ".join(processedPrefixes)}] {msg}'
            
        return processedMsg, kwargs

    def error(self, msg, *args, **kwargs):
        """Enhanced error method with automatic traceback inclusion."""
        if self.include_traceback and 'exc_info' not in kwargs:
            kwargs['exc_info'] = True
        super().error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Exception method that always includes traceback."""
        kwargs['exc_info'] = True
        super().error(msg, *args, **kwargs)

    def debug_with_context(self, msg, context=None, *args, **kwargs):
        """Debug logging with additional context information."""
        if context:
            msg = f"{msg} | Context: {context}"
        super().debug(msg, *args, **kwargs)

    def info_with_timing(self, msg, start_time=None, *args, **kwargs):
        """Info logging with optional timing information."""
        if start_time:
            elapsed = time.time() - start_time
            msg = f"{msg} (took {elapsed:.3f}s)"
        super().info(msg, *args, **kwargs)


def enable_signal_emission(signal):
    """
    Enable emitting log messages via a signal.
    
    Args:
        signal: A signal object with an emit method that accepts a dict
    """
    global _signal_handler, baseLogger
    
    # Remove old signal handler if it exists
    if _signal_handler is not None:
        baseLogger.removeHandler(_signal_handler)
    
    # Create and add new signal handler
    _signal_handler = SignalEmittingHandler(signal)
    _signal_handler.setLevel(logging.DEBUG)  # Emit all levels
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _signal_handler.setFormatter(formatter)
    baseLogger.addHandler(_signal_handler)
    
    baseLogger.debug("Signal emission enabled for logging")


def initLogger(obj, *, instanceName=None, tryInheritParent=False, level=None, 
               format_string=None, include_traceback=False, extra_prefixes=None,
               logger_name_override=None):
    """ 
    Initializes a logger for the specified object with enhanced configuration options.
    
    Args:
        obj: Class, object or string to create logger for
        instanceName: Optional instance name for the logger prefix
        tryInheritParent: Try to inherit logger from parent in call stack
        level: Optional logging level override (e.g., logging.DEBUG)
        format_string: Custom format string for this logger
        include_traceback: Automatically include traceback in error logs
        extra_prefixes: Additional prefixes to include in log messages
        logger_name_override: Override the automatic logger name detection
    """

    logger = None

    if tryInheritParent:
        # Use logger from first parent in stack that has one
        for frameInfo in inspect.stack():
            frameLocals = frameInfo[0].f_locals
            if 'self' not in frameLocals:
                continue

            parent = frameLocals['self']
            parentRef = weakref.ref(parent)
            if parentRef not in objLoggers:
                continue

            logger = objLoggers[parentRef]
            break

    if logger is None:
        # Create logger with enhanced naming
        if logger_name_override:
            objName = logger_name_override
        elif inspect.isclass(obj):
            objName = obj.__name__
        elif isinstance(obj, str):
            objName = obj
        else:
            objName = obj.__class__.__name__

        # Build prefix list
        prefixes = []
        prefixes.append(objName)
        
        if instanceName:
            prefixes.append(instanceName)
            
        if extra_prefixes:
            if isinstance(extra_prefixes, str):
                prefixes.append(extra_prefixes)
            elif isinstance(extra_prefixes, list):
                prefixes.extend(extra_prefixes)

        # Create object reference for weak reference tracking
        if inspect.isclass(obj):
            objRef = weakref.ref(obj)
        elif isinstance(obj, str):
            objRef = None
        else:
            objRef = weakref.ref(obj)

        logger = LoggerAdapter(baseLogger,
                               prefixes,
                               objRef,
                               include_traceback=include_traceback,
                               custom_format=format_string)

        # Set custom level if specified
        if level is not None:
            logger.setLevel(level)

        # Save logger so it can be used by tryInheritParent requesters later
        if objRef is not None:
            objLoggers[objRef] = logger

    return logger