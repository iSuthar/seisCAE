"""Core functionality for seisCAE."""

from .detection import EventDetector
from .preprocessing import SpectrogramGenerator, EventExtractor
from .catalog import EventCatalog

__all__ = [
    'EventDetector',
    'SpectrogramGenerator',
    'EventExtractor',
    'EventCatalog',
]
