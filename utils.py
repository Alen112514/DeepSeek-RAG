import logging
from functools import wraps
import time
from typing import Callable, Any, List, Dict
import aiofiles
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

async def save_uploaded_image(image_bytes: bytes, filename: str, directory: Path) -> Path:
    """Save uploaded image to directory and return the path."""
    filepath = directory / filename
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(image_bytes)
    return filepath