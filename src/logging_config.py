"""Logging configuration for Autopilot project."""

import logging
import logging.config
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} | {asctime} | {name}:{funcName}:{lineno} | {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "{levelname} | {name} | {message}",
            "style": "{",
        },
        "colored": {
            "format": "%(levelname)-8s | %(asctime)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "verbose",
            "filename": str(LOG_DIR / "autopilot.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "verbose",
            "filename": str(LOG_DIR / "errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "autopilot": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}


def setup_logging(log_level=logging.INFO):
    """
    Configure logging for the entire project.

    Args:
        log_level: Logging level (default: INFO)
    """
    logging.config.dictConfig(LOGGING_CONFIG)

    # Set root logger level
    logging.getLogger().setLevel(log_level)

    # Get project logger
    logger = logging.getLogger("autopilot")
    logger.setLevel(log_level)

    return logger


def get_logger(name):
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"autopilot.{name}")
