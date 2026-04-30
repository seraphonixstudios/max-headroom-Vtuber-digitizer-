#!/usr/bin/env python3
"""
Max Headroom - Centralized Logging Utility
Provides structured logging across all modules with console + file output
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "max_headroom.log")

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logger(name="MaxHeadroom", level=logging.INFO, log_to_file=True):
    """Setup and return a configured logger."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    
    if sys.platform != "win32" or os.environ.get("TERM"):
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    else:
        console_formatter = logging.Formatter(console_format, datefmt="%H:%M:%S")
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Main application logger
LOG = setup_logger("MaxHeadroom")

# Module-specific loggers
def get_logger(module_name):
    """Get a logger for a specific module."""
    return setup_logger(f"MaxHeadroom.{module_name}")

# Test
if __name__ == "__main__":
    LOG.debug("Debug message test")
    LOG.info("Info message test")
    LOG.warning("Warning message test")
    LOG.error("Error message test")
    LOG.critical("Critical message test")
    print(f"Log file: {LOG_FILE}")