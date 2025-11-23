"""Clustering algorithms for seisCAE."""

from .base import BaseClusterer
from .gmm import GMMClusterer
from .metrics_phase3 import SpectrogramMetrics, ClusterAnalyzer

# Clusterer registry
CLUSTERERS = {
    'gmm': GMMClusterer,
    # Future: 'dbscan': DBSCANClusterer,
    # Future: 'hdbscan': HDBSCANClusterer,
}


def get_clusterer(name: str, **kwargs) -> BaseClusterer:
    """
    Factory function to get clusterer by name.
    
    Parameters
    ----------
    name : str
        Clusterer name (e.g., 'gmm')
    **kwargs
        Clusterer-specific parameters
        
    Returns
    -------
    clusterer : BaseClusterer
        Initialized clusterer
    
    Examples
    --------
    >>> clusterer = get_clusterer('gmm', n_clusters=5)
    """
    if name not in CLUSTERERS:
        raise ValueError(f"Unknown clusterer: {name}. Available: {list(CLUSTERERS.keys())}")
    return CLUSTERERS[name](**kwargs)


__all__ = [
    'BaseClusterer',
    'GMMClusterer',
    'SpectrogramMetrics',
    'ClusterAnalyzer',
    'get_clusterer',
]
