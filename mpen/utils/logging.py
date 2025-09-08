"""
Logging utilities for the MPEN system.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_system_logger(level: str = "INFO") -> logging.Logger:
    """Get the main system logger."""
    return setup_logger("mpen.system", level=level)


def get_agent_logger(agent_id: str, level: str = "INFO") -> logging.Logger:
    """Get a logger for a specific agent."""
    return setup_logger(f"mpen.agent.{agent_id}", level=level)


def configure_logging_for_experiments(
    experiment_name: str,
    log_dir: str = "logs",
    level: str = "INFO"
) -> logging.Logger:
    """
    Configure logging for experimental runs.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Configured experiment logger
    """
    log_file = f"{log_dir}/{experiment_name}.log"
    
    return setup_logger(
        f"mpen.experiment.{experiment_name}",
        level=level,
        log_file=log_file
    )
