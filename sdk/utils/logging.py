"""
Logging Utilities
=================

Structured logging for AIPlatform SDK.
"""

import logging
import sys
import json
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class LogFormat(Enum):
    """Log output formats."""
    TEXT = "text"
    JSON = "json"
    COLORED = "colored"


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Build message
        msg = f"{color}[{timestamp}] {record.levelname:8}{reset} "
        msg += f"\033[1m{record.name}\033[0m: {record.getMessage()}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info',
                          'exc_info', 'exc_text', 'message']:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logging(level: str = "INFO",
                  format: LogFormat = LogFormat.COLORED,
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup SDK logging.
    
    Args:
        level: Log level
        format: Output format
        log_file: Optional file path
        
    Returns:
        Root logger
    """
    # Get root SDK logger
    root_logger = logging.getLogger("sdk")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if format == LogFormat.JSON:
        console_handler.setFormatter(JSONFormatter())
    elif format == LogFormat.COLORED:
        console_handler.setFormatter(ColoredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"sdk.{name}")


class LogContext:
    """
    Context manager for structured logging.
    
    Example:
        >>> with LogContext(operation="train", model="vqe"):
        ...     logger.info("Starting training")
    """
    
    _context: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        self._added = kwargs
    
    def __enter__(self):
        LogContext._context.update(self._added)
        return self
    
    def __exit__(self, *args):
        for key in self._added:
            LogContext._context.pop(key, None)
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls._context.get(key, default)
    
    @classmethod
    def all(cls) -> Dict[str, Any]:
        return cls._context.copy()


class ContextLogger:
    """Logger with automatic context injection."""
    
    def __init__(self, name: str):
        self._logger = get_logger(name)
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        # Add context to extra
        extra = kwargs.pop('extra', {})
        extra.update(LogContext.all())
        kwargs['extra'] = extra
        
        self._logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)
