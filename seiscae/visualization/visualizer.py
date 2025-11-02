"""Main visualizer class that orchestrates all visualizations."""

import numpy as np
from pathlib import Path
from typing import Optional
import logging

from .spectrograms import plot_detection_summary
from .diagnostics import plot_training_history, plot_reconstruction_grid, plot_loss_distribution
from .clusters import plot_cluster_examples, plot_cluster_centers, plot_cluster_sizes
from .latent_space import plot_latent_space_umap, plot_latent_space_tsne, plot_gmm_selection_metrics

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Main visualizer class for seisCAE.
    
    Parameters
    ----------
    config : ConfigManager
        Configuration manager
    
    Examples
    --------
    >>> from seiscae.config import load_config
    >>> config = load_config()
    >>> viz = Visualizer(config)
    >>> viz.plot_training_history(history, 'history.png')
    """
    
    def __init__(self, config):
        self.config = config
        self.dpi = config.get('visualization.dpi', 300)
    
    def plot_detection_summary(self, trace, cft, triggers, save_path):
        """Plot detection summary."""
        return plot_detection_summary(
            trace, cft, triggers, save_path,
            self.config.get('detection.threshold_on'),
            self.config.get('detection.threshold_off'),
            self.dpi
        )
    
    def plot_training_history(self, history, save_path):
        """Plot training history."""
        return plot_training_history(history, save_path, self.dpi)
    
    def plot_reconstruction_grid(self, original, reconstructed, save_path, n_examples=8):
        """Plot reconstruction grid."""
        return plot_reconstruction_grid(original, reconstructed, n_examples, save_path, self.dpi)
    
    def plot_cluster_examples(self, spectrograms, labels, cluster_id, save_path, n_examples=5):
        """Plot cluster examples."""
        return plot_cluster_examples(spectrograms, labels, cluster_id, n_examples, save_path, self.dpi)
    
    def plot_cluster_centers(self, centers, labels, save_path):
        """Plot cluster centers."""
        return plot_cluster_centers(centers, labels, save_path, self.dpi)
    
    def plot_cluster_sizes(self, labels, save_path):
        """Plot cluster sizes."""
        return plot_cluster_sizes(labels, save_path, self.dpi)
    
    def plot_latent_space(self, features, labels, save_path, method='umap'):
        """Plot latent space projection."""
        if method == 'umap':
            return plot_latent_space_umap(features, labels, save_path, self.dpi)
        elif method == 'tsne':
            return plot_latent_space_tsne(features, labels, save_path, self.dpi)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def plot_gmm_metrics(self, metrics, save_path):
        """Plot GMM selection metrics."""
        return plot_gmm_selection_metrics(metrics, save_path, self.dpi)
    
    def plot_all(self, catalog, features, labels, output_dir):
        """
        Generate all visualizations.
        
        Parameters
        ----------
        catalog : EventCatalog
            Event catalog
        features : np.ndarray
            Latent features
        labels : np.ndarray
            Cluster labels
        output_dir : Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating visualizations...")
        
        # Cluster visualizations
        if labels is not None:
            logger.info("  - Cluster sizes")
            self.plot_cluster_sizes(labels, output_dir / 'cluster_sizes.png')
            
            logger.info("  - Latent space (UMAP)")
            self.plot_latent_space(features, labels, output_dir / 'latent_umap.png', 'umap')
            
            logger.info("  - Latent space (t-SNE)")
            self.plot_latent_space(features, labels, output_dir / 'latent_tsne.png', 'tsne')
            
            # Examples per cluster
            spectrograms = np.array([e['spectrogram'] for e in catalog.events])
            for cluster_id in np.unique(labels):
                logger.info(f"  - Cluster {cluster_id} examples")
                self.plot_cluster_examples(
                    spectrograms, labels, cluster_id,
                    output_dir / f'cluster_{cluster_id}_examples.png',
                    n_examples=5
                )
        
        logger.info("Visualizations complete!")
