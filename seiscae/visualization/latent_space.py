"""Latent space visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def plot_latent_space_umap(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot UMAP projection of latent space.
    
    Parameters
    ----------
    features : np.ndarray
        Latent features
    labels : np.ndarray, optional
        Cluster labels for coloring
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    try:
        import umap
    except ImportError:
        logger.error("UMAP not installed. Install with: pip install umap-learn")
        return None
    
    # Compute UMAP embedding
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='tab10', s=20, alpha=0.6, edgecolors='k', linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP Projection of Latent Space', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved UMAP plot to {save_path}")
        plt.close(fig)
    
    return fig


def plot_latent_space_tsne(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot t-SNE projection of latent space.
    
    Parameters
    ----------
    features : np.ndarray
        Latent features
    labels : np.ndarray, optional
        Cluster labels for coloring
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    
    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='tab10', s=20, alpha=0.6, edgecolors='k', linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Projection of Latent Space', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved t-SNE plot to {save_path}")
        plt.close(fig)
    
    return fig


def plot_gmm_selection_metrics(
    metrics: dict,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot GMM cluster selection metrics (BIC, AIC, Silhouette).
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from GMMClusterer
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    k_range = metrics['k_range']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BIC and AIC
    ax = axes[0]
    ax.plot(k_range, metrics['bic'], 'o-', label='BIC', linewidth=2, markersize=6)
    ax.plot(k_range, metrics['aic'], 's-', label='AIC', linewidth=2, markersize=6)
    ax.axvline(metrics['best_k_bic'], color='red', linestyle='--', 
               label=f'Best k={metrics["best_k_bic"]}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Score (lower is better)', fontsize=12)
    ax.set_title('BIC and AIC', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Silhouette
    ax = axes[1]
    ax.plot(k_range, metrics['silhouette'], 'o-', color='green', linewidth=2, markersize=6)
    if metrics['best_k_silhouette']:
        ax.axvline(metrics['best_k_silhouette'], color='red', linestyle='--',
                   label=f'Best k={metrics["best_k_silhouette"]}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score (higher is better)', fontsize=12)
    ax.set_title('Silhouette Score', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved GMM selection metrics to {save_path}")
        plt.close(fig)
    
    return fig
