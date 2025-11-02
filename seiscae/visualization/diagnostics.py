"""Diagnostic visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, Any],
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot training history (loss curves).
    
    Parameters
    ----------
    history : dict
        Training history with 'train_losses' and 'val_losses'
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    ax.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_losses'], 'r-', linewidth=2, label='Validation Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_losses']) + 1
    best_val_loss = min(history['val_losses'])
    ax.plot(best_epoch, best_val_loss, 'r*', markersize=15, 
            label=f'Best (Epoch {best_epoch})')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
        plt.close(fig)
    
    return fig


def plot_reconstruction_grid(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    n_examples: int = 8,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot grid of original and reconstructed spectrograms.
    
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
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_examples = min(n_examples, original.shape[0])
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 2*n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        orig = original[i, 0].cpu().detach().numpy()
        recon = reconstructed[i, 0].cpu().detach().numpy()
        
        # Original
        axes[i, 0].imshow(orig, aspect='auto', origin='lower', cmap='OrRd')
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].set_ylabel(f'Sample {i+1}')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Reconstructed
        axes[i, 1].imshow(recon, aspect='auto', origin='lower', cmap='OrRd')
        axes[i, 1].set_title('Reconstructed' if i == 0 else '')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved reconstruction grid to {save_path}")
        plt.close(fig)
    
    return fig


def plot_loss_distribution(
    losses: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot distribution of reconstruction losses.
    
    Parameters
    ----------
    losses : np.ndarray
        Per-sample reconstruction losses
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(losses, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(losses), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(losses):.4f}')
    ax.set_xlabel('Reconstruction Loss (MSE)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Reconstruction Losses', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved loss distribution to {save_path}")
        plt.close(fig)
    
    return fig
