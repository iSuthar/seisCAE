"""Comprehensive training and clustering diagnostic visualizations.

This module provides advanced visualization tools for:
- Training diagnostics (loss curves, reconstructions, latent space evolution)
- Clustering diagnostics (cluster grids, feature heatmaps, distance distributions)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TRAINING DIAGNOSTICS
# ============================================================================

def plot_loss_curves_detailed(
    history: Dict[str, Any],
    save_path: Optional[str] = None,
    log_scale: bool = False,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot detailed training history with log scale option.
    
    Parameters
    ----------
    history : dict
        Training history with 'train_losses' and 'val_losses'
    save_path : str, optional
        Path to save figure
    log_scale : bool
        Use log scale for y-axis
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # Linear scale plot
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.7)
    
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.5, 
                label=f'Best Epoch ({best_epoch})')
    ax1.plot(best_epoch, best_val_loss, 'g*', markersize=15)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training History - Linear Scale', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    if log_scale:
        ax2.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        ax2.semilogy(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.7)
    else:
        ax2.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        ax2.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.7)
    
    ax2.axvline(best_epoch, color='green', linestyle='--', alpha=0.5,
                label=f'Best Epoch ({best_epoch})')
    ax2.plot(best_epoch, best_val_loss, 'g*', markersize=15)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)' + (' [log scale]' if log_scale else ''), fontsize=12)
    ax2.set_title('Training History - ' + ('Log Scale' if log_scale else 'Zoomed'), fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics box
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    stats_text = f'Final Train: {final_train:.6f}\nFinal Val: {final_val:.6f}\nBest Val: {best_val_loss:.6f}'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        plt.close(fig)
    
    return fig


def plot_reconstruction_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    n_examples: int = 8,
    save_path: Optional[str] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot side-by-side comparison of original and reconstructed spectrograms.
    
    Parameters
    ----------
    original : torch.Tensor
        Original spectrograms, shape (N, 1, H, W)
    reconstructed : torch.Tensor
        Reconstructed spectrograms, shape (N, 1, H, W)
    n_examples : int
        Number of examples to show
    save_path : str, optional
        Path to save figure
    freq_range : tuple, optional
        (min_freq, max_freq) for axis labels
    time_range : tuple, optional
        (min_time, max_time) for axis labels
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    n_examples = min(n_examples, original.shape[0])
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 2.5*n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        orig = original[i, 0].cpu().detach().numpy()
        recon = reconstructed[i, 0].cpu().detach().numpy()
        diff = np.abs(orig - recon)
        
        # Original
        im0 = axes[i, 0].imshow(orig, aspect='auto', origin='lower', 
                                cmap='OrRd', vmin=0, vmax=1)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].set_ylabel(f'Sample {i+1}\nFrequency')
        
        # Reconstructed
        im1 = axes[i, 1].imshow(recon, aspect='auto', origin='lower',
                                cmap='OrRd', vmin=0, vmax=1)
        axes[i, 1].set_title('Reconstructed' if i == 0 else '')
        axes[i, 1].set_ylabel('')
        
        # Difference
        im2 = axes[i, 2].imshow(diff, aspect='auto', origin='lower',
                                cmap='viridis', vmin=0, vmax=0.5)
        axes[i, 2].set_title('|Difference|' if i == 0 else '')
        axes[i, 2].set_ylabel('')
        
        # Add MSE text
        mse = np.mean((orig - recon) ** 2)
        axes[i, 2].text(0.98, 0.02, f'MSE: {mse:.4f}', 
                       transform=axes[i, 2].transAxes,
                       fontsize=9, color='white', ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Set x-labels for bottom row
        if i == n_examples - 1:
            axes[i, 0].set_xlabel('Time')
            axes[i, 1].set_xlabel('Time')
            axes[i, 2].set_xlabel('Time')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved reconstruction comparison to {save_path}")
        plt.close(fig)
    
    return fig


def plot_latent_space_evolution(
    features_dict: Dict[int, np.ndarray],
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot t-SNE visualization of latent space at multiple epochs.
    
    Parameters
    ----------
    features_dict : dict
        Dictionary mapping epoch number to feature arrays
        e.g., {10: features_epoch10, 50: features_epoch50, ...}
    labels : np.ndarray, optional
        Cluster labels for coloring points
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    n_epochs = len(features_dict)
    epochs = sorted(features_dict.keys())
    
    fig, axes = plt.subplots(1, n_epochs, figsize=(5*n_epochs, 5))
    
    if n_epochs == 1:
        axes = [axes]
    
    for idx, epoch in enumerate(epochs):
        features = features_dict[epoch]
        
        # Compute t-SNE
        logger.info(f"Computing t-SNE for epoch {epoch}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        # Plot
        if labels is not None:
            scatter = axes[idx].scatter(features_2d[:, 0], features_2d[:, 1],
                                       c=labels, cmap='tab20', alpha=0.6, s=20)
            if idx == n_epochs - 1:
                plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        else:
            axes[idx].scatter(features_2d[:, 0], features_2d[:, 1],
                            alpha=0.6, s=20, c='steelblue')
        
        axes[idx].set_title(f'Epoch {epoch}', fontsize=12)
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2' if idx == 0 else '')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved latent space evolution to {save_path}")
        plt.close(fig)
    
    return fig


def plot_per_sample_loss_histogram(
    losses: np.ndarray,
    threshold_percentile: float = 95,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot histogram of per-sample reconstruction losses.
    
    Helps identify outliers and poor reconstructions.
    
    Parameters
    ----------
    losses : np.ndarray
        Per-sample reconstruction losses
    threshold_percentile : float
        Percentile for marking outliers
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(losses, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    threshold = np.percentile(losses, threshold_percentile)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'{threshold_percentile}th percentile')
    
    ax1.set_xlabel('Reconstruction Loss (MSE)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Per-Sample Losses', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2.boxplot(losses, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax2.set_title('Loss Distribution (Box Plot)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = (f'Mean: {np.mean(losses):.6f}\n'
                 f'Median: {np.median(losses):.6f}\n'
                 f'Std: {np.std(losses):.6f}\n'
                 f'Min: {np.min(losses):.6f}\n'
                 f'Max: {np.max(losses):.6f}')
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved loss histogram to {save_path}")
        plt.close(fig)
    
    return fig


# ============================================================================
# CLUSTERING DIAGNOSTICS
# ============================================================================

def plot_cluster_grid(
    cluster_id: int,
    spectrograms: np.ndarray,
    waveforms: np.ndarray,
    labels: np.ndarray,
    reconstructed: Optional[np.ndarray] = None,
    n_examples: int = 5,
    freq_bins: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 100,
) -> plt.Figure:
    """
    Plot multi-panel layout for cluster examples.
    
    Matches research paper visualization: waveform | original spec | 
    reconstructed spec | frequency profile for each example.
    
    Parameters
    ----------
    cluster_id : int
        Cluster ID to visualize
    spectrograms : np.ndarray
        Original spectrograms
    waveforms : np.ndarray
        Waveform data
    labels : np.ndarray
        Cluster labels
    reconstructed : np.ndarray, optional
        Reconstructed spectrograms
    n_examples : int
        Number of examples to show
    freq_bins : np.ndarray, optional
        Frequency bin values
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    cluster_indices = np.where(labels == cluster_id)[0]
    n_examples = min(n_examples, len(cluster_indices))
    
    if n_examples == 0:
        logger.warning(f"No samples in cluster {cluster_id}")
        return None
    
    # Select evenly spaced examples
    idx = np.linspace(0, len(cluster_indices)-1, n_examples, dtype=int)
    selected_indices = cluster_indices[idx]
    
    # Create figure
    n_cols = 4 if reconstructed is not None else 3
    fig = plt.figure(figsize=(10, 4*n_examples), dpi=dpi)
    grid = plt.GridSpec(n_examples, 7, hspace=0.2, wspace=0.1)
    
    for i, sample_idx in enumerate(selected_indices):
        spec = spectrograms[sample_idx]
        if spec.ndim == 3:  # (C, H, W)
            spec = spec[0]
        
        waveform = waveforms[sample_idx] if waveforms is not None else None
        
        # Frequency profile (sum over time)
        freq_profile = np.sum(spec, axis=-1)
        
        # Panel 1: Frequency profile
        ax1 = fig.add_subplot(grid[i, 0])
        if freq_bins is not None:
            ax1.plot(freq_profile, freq_bins)
        else:
            ax1.plot(freq_profile, range(len(freq_profile)))
        ax1.invert_xaxis()
        ax1.set_xticks([])
        if i == 0:
            ax1.set_ylabel('Frequency [Hz]', fontsize=10)
        
        # Panel 2: Original spectrogram
        ax2 = fig.add_subplot(grid[i, 1:4])
        ax2.pcolormesh(spec, cmap='Reds', vmin=0, vmax=1, shading='auto')
        ax2.tick_params(labelleft=False)
        ax2.set_yticks([])
        if i == 0:
            ax2.set_title('STFT (original)', fontsize=10)
        
        # Panel 3: Reconstructed spectrogram (if available)
        if reconstructed is not None:
            recon_spec = reconstructed[sample_idx]
            if recon_spec.ndim == 3:
                recon_spec = recon_spec[0]
            
            ax3 = fig.add_subplot(grid[i, 4:])
            im = ax3.pcolormesh(recon_spec, cmap='Reds', vmin=0, vmax=1, shading='auto')
            ax3.tick_params(labelleft=False)
            ax3.set_yticks([])
            if i == 0:
                ax3.set_title('STFT (reconstructed)', fontsize=10)
            
            # Add colorbar
            if i == n_examples - 1:
                fig.colorbar(im, ax=(ax2, ax3), fraction=0.05)
        
        # Add xlabel for bottom row
        if i == n_examples - 1:
            ax2.set_xlabel('Time [sec]', fontsize=12)
            if reconstructed is not None:
                ax3.set_xlabel('Time [sec]', fontsize=12)
    
    plt.suptitle(f'Cluster {cluster_id+1} ({len(cluster_indices)} samples)', 
                 fontsize=15, y=0.92)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved cluster grid to {save_path}")
        plt.close(fig)
    
    return fig


def plot_cluster_feature_heatmap(
    cluster_stats: pd.DataFrame,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot heatmap of Phase 3 metrics across clusters.
    
    Parameters
    ----------
    cluster_stats : pd.DataFrame
        Cluster statistics with Phase 3 metrics
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    # Extract mean metrics
    metric_cols = [col for col in cluster_stats.columns if col.endswith('_mean')]
    
    data = cluster_stats[metric_cols].values
    cluster_labels = [f"C{int(c)}" for c in cluster_stats['cluster']]
    feature_labels = [col.replace('_mean', '').replace('_', ' ').title() 
                     for col in metric_cols]
    
    fig, ax = plt.subplots(figsize=(max(10, len(metric_cols)*1.5), 
                                    max(6, len(cluster_labels)*0.4)))
    
    # Normalize data for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    sns.heatmap(data_normalized, annot=True, fmt='.2f', cmap='RdYlBu_r',
                yticklabels=cluster_labels, xticklabels=feature_labels,
                cbar_kws={'label': 'Normalized Value'}, ax=ax)
    
    ax.set_title('Phase 3 Metrics Heatmap (Standardized)', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Clusters', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved feature heatmap to {save_path}")
        plt.close(fig)
    
    return fig


def plot_distance_distributions(
    distances: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot per-cluster distance to center distributions.
    
    Parameters
    ----------
    distances : np.ndarray
        Distance to cluster center for each sample
    labels : np.ndarray
        Cluster labels
    n_clusters : int, optional
        Number of clusters to plot (None = all)
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
    """
    unique_labels = np.unique(labels)
    if n_clusters is not None:
        unique_labels = unique_labels[:n_clusters]
    
    n = len(unique_labels)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_distances = distances[mask]
        
        axes[idx].hist(cluster_distances, bins=30, color='steelblue',
                      edgecolor='black', alpha=0.7)
        axes[idx].axvline(np.mean(cluster_distances), color='red',
                         linestyle='--', linewidth=2, label='Mean')
        axes[idx].axvline(np.median(cluster_distances), color='orange',
                         linestyle='--', linewidth=2, label='Median')
        
        axes[idx].set_title(f'Cluster {cluster_id} (n={np.sum(mask)})', fontsize=11)
        axes[idx].set_xlabel('Distance to Center')
        axes[idx].set_ylabel('Count')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(unique_labels), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Distance to Cluster Center Distributions', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved distance distributions to {save_path}")
        plt.close(fig)
    
    return fig
