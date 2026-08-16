import os
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)

_log_file = os.environ.get("GENRECON_PIPELINE_LOG")
if _log_file:
    Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        _log_file,
        level="DEBUG",
        mode="a",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
    )

__all__ = ["logger"]
