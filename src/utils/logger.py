"""Logging setup for the IELTS Writing Task 2 application."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "ielts_writing",
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a configured logger.

    Args:
        name: Logger name.
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
        log_format: Custom log format string.

    Returns:
        Configured logger instance.
    """
    if level is None:
        level = "INFO"

    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
