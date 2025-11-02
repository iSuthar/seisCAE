"""Logging setup utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None, 
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Log file path. If None, only logs to console.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    format_str : str, optional
        Custom format string
    
    Examples
    --------
    >>> setup_logging('seiscae.log', level=logging.DEBUG)
    >>> setup_logging()  # Console only
    """
    if format_str is None:
        format_str = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
