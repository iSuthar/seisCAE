"""Visualization utilities for seisCAE."""

from .spectrograms import plot_spectrogram, plot_waveform_spectrogram
from .diagnostics import plot_training_history, plot_reconstruction_grid
from .clusters import plot_cluster_examples, plot_cluster_centers
from .latent_space import plot_latent_space_umap, plot_latent_space_tsne
from .visualizer import Visualizer
from .training_diagnostics import (
    plot_loss_curves_detailed,
    plot_reconstruction_comparison,
    plot_latent_space_evolution,
    plot_per_sample_loss_histogram,
    plot_cluster_grid,
    plot_cluster_feature_heatmap,
    plot_distance_distributions,
)

__all__ = [
    'plot_spectrogram',
    'plot_waveform_spectrogram',
    'plot_training_history',
    'plot_reconstruction_grid',
    'plot_cluster_examples',
    'plot_cluster_centers',
    'plot_latent_space_umap',
    'plot_latent_space_tsne',
    'Visualizer',
    # Training diagnostics
    'plot_loss_curves_detailed',
    'plot_reconstruction_comparison',
    'plot_latent_space_evolution',
    'plot_per_sample_loss_histogram',
    # Clustering diagnostics
    'plot_cluster_grid',
    'plot_cluster_feature_heatmap',
    'plot_distance_distributions',
]
