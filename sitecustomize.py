import asyncio
import inspect


_original_is_coroutine_function = asyncio.iscoroutinefunction


def _patched_is_coroutine_function(obj):
    """Compatibility shim for Python 3.14+ where asyncio.iscoroutinefunction is deprecated."""
    try:
        return inspect.iscoroutinefunction(obj)
    except Exception:
        return _original_is_coroutine_function(obj)


asyncio.iscoroutinefunction = _patched_is_coroutine_function
