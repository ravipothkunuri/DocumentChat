"""Error-handling utilities shared across modules.

Includes `handle_errors` decorator to log and re-raise exceptions with
operation context.
"""
from functools import wraps
import logging


logger = logging.getLogger(__name__)


def handle_errors(operation_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                raise
        return wrapper
    return decorator


