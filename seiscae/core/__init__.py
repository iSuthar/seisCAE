"""Core detection and preprocessing modules."""

from .detection import EventDetector, multi_sta_lta
from .preprocessing import SpectrogramGenerator, EventExtractor
from .catalog import EventCatalog

__all__ = [
    'EventDetector',
    'multi_sta_lta',
    'SpectrogramGenerator',
    'EventExtractor',
    'EventCatalog',
]
