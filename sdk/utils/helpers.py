"""
Helper Utilities
================

Decorators and helper functions.
"""

from typing import Callable, Any, Optional, TypeVar, Dict
from functools import wraps
import asyncio
import time
import threading
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def retry(max_attempts: int = 3,
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: tuple = (Exception,)) -> Callable[[F], F]:
    """
    Retry decorator for synchronous functions.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
        
    Example:
        >>> @retry(max_attempts=3, delay=1.0)
        ... def unstable_operation():
        ...     # may fail sometimes
        ...     pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(max_attempts: int = 3,
                delay: float = 1.0,
                backoff: float = 2.0,
                exceptions: tuple = (Exception,)) -> Callable[[F], F]:
    """
    Retry decorator for async functions.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Timeout decorator for synchronous functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


def async_timeout(seconds: float) -> Callable[[F], F]:
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
        
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter for function calls.
    
    Example:
        >>> limiter = RateLimiter(calls=10, period=60)
        >>> @limiter.limit
        ... def api_call():
        ...     pass
    """
    
    def __init__(self, calls: int, period: float):
        """
        Initialize rate limiter.
        
        Args:
            calls: Maximum calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self._timestamps: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
    
    def _cleanup(self, key: str):
        """Remove old timestamps."""
        cutoff = time.time() - self.period
        self._timestamps[key] = [
            ts for ts in self._timestamps[key] if ts > cutoff
        ]
    
    def is_allowed(self, key: str = "default") -> bool:
        """Check if call is allowed."""
        with self._lock:
            self._cleanup(key)
            return len(self._timestamps[key]) < self.calls
    
    def record(self, key: str = "default"):
        """Record a call."""
        with self._lock:
            self._timestamps[key].append(time.time())
    
    def wait_time(self, key: str = "default") -> float:
        """Get time to wait before next call."""
        with self._lock:
            self._cleanup(key)
            
            if len(self._timestamps[key]) < self.calls:
                return 0.0
            
            oldest = min(self._timestamps[key])
            return max(0, oldest + self.period - time.time())
    
    def limit(self, func: F) -> F:
        """Decorator to rate limit a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = func.__name__
            
            wait = self.wait_time(key)
            if wait > 0:
                logger.debug(f"Rate limited, waiting {wait:.2f}s")
                time.sleep(wait)
            
            self.record(key)
            return func(*args, **kwargs)
        
        return wrapper
    
    def async_limit(self, func: F) -> F:
        """Decorator to rate limit an async function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = func.__name__
            
            wait = self.wait_time(key)
            if wait > 0:
                logger.debug(f"Rate limited, waiting {wait:.2f}s")
                await asyncio.sleep(wait)
            
            self.record(key)
            return await func(*args, **kwargs)
        
        return wrapper


def rate_limit(calls: int, period: float) -> Callable[[F], F]:
    """
    Rate limit decorator factory.
    
    Args:
        calls: Maximum calls allowed
        period: Time period in seconds
    """
    limiter = RateLimiter(calls, period)
    return limiter.limit


def cache(ttl: float = 300, maxsize: int = 128) -> Callable[[F], F]:
    """
    Simple caching decorator with TTL.
    
    Args:
        ttl: Time to live in seconds
        maxsize: Maximum cache size
    """
    def decorator(func: F) -> F:
        _cache: Dict[str, tuple] = {}
        _lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str((args, tuple(sorted(kwargs.items()))))
            
            with _lock:
                # Check cache
                if key in _cache:
                    value, timestamp = _cache[key]
                    if time.time() - timestamp < ttl:
                        return value
                
                # Compute value
                value = func(*args, **kwargs)
                
                # Store in cache
                _cache[key] = (value, time.time())
                
                # Cleanup old entries
                if len(_cache) > maxsize:
                    oldest = min(_cache.items(), key=lambda x: x[1][1])
                    del _cache[oldest[0]]
                
                return value
        
        wrapper.cache_clear = lambda: _cache.clear()
        return wrapper
    
    return decorator


def deprecated(message: str = "") -> Callable[[F], F]:
    """
    Mark a function as deprecated.
    
    Args:
        message: Deprecation message
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(
                f"Function {func.__name__} is deprecated. {message}"
            )
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls):
    """
    Singleton decorator for classes.
    
    Example:
        >>> @singleton
        ... class MyClass:
        ...     pass
    """
    instances = {}
    lock = threading.Lock()
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
    
    return get_instance


class LazyProperty:
    """
    Lazy property descriptor.
    
    Example:
        >>> class MyClass:
        ...     @LazyProperty
        ...     def expensive_computation(self):
        ...         return compute_something()
    """
    
    def __init__(self, func: Callable):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.func(obj))
        
        return getattr(obj, self.attr_name)


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """Unflatten dictionary."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result
