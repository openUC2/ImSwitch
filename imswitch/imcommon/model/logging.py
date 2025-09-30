import inspect
import logging
import weakref
import time

try:
    import coloredlogs
    _coloredlogs_available = True
except ImportError:
    _coloredlogs_available = False


baseLogger = logging.getLogger('imswitch')
if _coloredlogs_available:
    coloredlogs.install(level='DEBUG', logger=baseLogger,
                        fmt='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    coloredlogs.install(level='INFO', logger=baseLogger,
                        fmt='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
else:
    # Fallback to basic logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
objLoggers = {}


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