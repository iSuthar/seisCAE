"""Cluster visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def plot_cluster_examples(
    spectrograms: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    n_examples: int = 5,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot examples from a specific cluster.
    
    Parameters
    ----------
    spectrograms : np.ndarray
        All spectrograms
    labels : np.ndarray
        Cluster labels
    cluster_id : int
        Cluster ID to visualize
    n_examples : int
        Number of examples to show
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    cluster_indices = np.where(labels == cluster_id)[0]
    n_examples = min(n_examples, len(cluster_indices))
    
    if n_examples == 0:
        logger.warning(f"No samples in cluster {cluster_id}")
        return None
    
    indices = np.random.choice(cluster_indices, n_examples, replace=False)
    
    fig, axes = plt.subplots(1, n_examples, figsize=(3*n_examples, 4))
    
    if n_examples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        spec = spectrograms[idx]
        if spec.ndim == 3:  # (C, H, W)
            spec = spec[0]
        
        axes[i].imshow(spec, aspect='auto', origin='lower', cmap='OrRd')
        axes[i].set_title(f'Sample {idx}')
        axes[i].set_xlabel('Time')
        if i == 0:
            axes[i].set_ylabel('Frequency')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.suptitle(f'Cluster {cluster_id} ({len(cluster_indices)} samples)', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved cluster examples to {save_path}")
        plt.close(fig)
    
    return fig


def plot_cluster_centers(
    centers: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot cluster centers as heatmap.
    
    Parameters
    ----------
    centers : np.ndarray
        Cluster centers, shape (n_clusters, n_features)
    labels : np.ndarray
        Cluster labels for counting
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    # Count samples per cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Create labels with counts
    ylabels = [f"Cluster {i} (n={cluster_counts.get(i, 0)})" for i in range(len(centers))]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(centers) * 0.5)))
    
    sns.heatmap(
        centers,
        cmap='viridis',
        yticklabels=ylabels,
        xticklabels=[f'F{i+1}' for i in range(centers.shape[1])],
        cbar_kws={'label': 'Feature Value'},
        ax=ax,
    )
    
    ax.set_title('Cluster Centers (Feature Space)', fontsize=14)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Clusters', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved cluster centers to {save_path}")
        plt.close(fig)
    
    return fig


def plot_cluster_sizes(
    labels: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot cluster size distribution.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add count labels on bars
    for i, (cluster, count) in enumerate(zip(unique, counts)):
        ax.text(cluster, count + max(counts)*0.01, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=14)
    ax.set_xticks(unique)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved cluster sizes to {save_path}")
        plt.close(fig)
    
    return fig
