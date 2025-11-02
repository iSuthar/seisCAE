"""
seisCAE: Clustering Seismic Events with Convolutional Autoencoders
====================================================================

A modular Python package for detecting, analyzing, and clustering seismic events
using deep learning (convolutional autoencoders) and unsupervised learning (GMM).

Basic Usage
-----------
>>> from seiscae import Pipeline
>>> from seiscae.config import load_config
>>>
>>> config = load_config("configs/default.yaml")
>>> pipeline = Pipeline(config)
>>> results = pipeline.run(data_path="/path/to/data")
>>> print(f"Found {results.n_clusters} clusters")

Modular Usage
-------------
>>> from seiscae.core import EventDetector
>>> from seiscae.models import ConvAutoencoder
>>> from seiscae.clustering import GMMClusterer
>>>
>>> detector = EventDetector()
>>> events = detector.detect_directory("/path/to/data")
>>>
>>> model = ConvAutoencoder(latent_dim=16)
>>> # ... train model ...
>>>
>>> clusterer = GMMClusterer()
>>> labels = clusterer.fit_predict(features)
"""

__version__ = "0.1.0"
__author__ = "Ankit Suthar"
__email__ = "ankit.suthar@example.com"

# Core imports
from .pipeline import Pipeline, PipelineResults
from .config import load_config, ConfigManager

# Component imports (for modular usage)
from .core import EventDetector, SpectrogramGenerator, EventCatalog
from .models import get_model, ConvAutoencoder
from .training import AutoencoderTrainer
from .clustering import get_clusterer, GMMClusterer
from .visualization import Visualizer

# Utilities
from .utils.logging import setup_logging
from .utils.device import get_device

__all__ = [
    # Main API
    'Pipeline',
    'PipelineResults',
    'load_config',
    'ConfigManager',
    
    # Core components
    'EventDetector',
    'SpectrogramGenerator',
    'EventCatalog',
    
    # Models
    'get_model',
    'ConvAutoencoder',
    
    # Training
    'AutoencoderTrainer',
    
    # Clustering
    'get_clusterer',
    'GMMClusterer',
    
    # Visualization
    'Visualizer',
    
    # Utils
    'setup_logging',
    'get_device',
]
