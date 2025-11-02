"""Training callbacks."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping
    delta : float
        Minimum change to qualify as improvement
    save_path : Path, optional
        Path to save best model
    
    Examples
    --------
    >>> callback = EarlyStopping(patience=20, save_path='best_model.pt')
    >>> result = callback(epoch, val_loss, model)
    >>> if result['stop_training']:
    ...     break
    """
    
    def __init__(
        self,
        patience: int = 7,
        delta: float = 0.0,
        save_path: Optional[Path] = None,
    ):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        
        self.counter = 0
        self.best_score = None
        self.best_loss = np.inf
    
    def __call__(self, epoch: int, val_loss: float, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Check if training should stop.
        
        Returns
        -------
        dict : {'stop_training': bool}
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return {'stop_training': True}
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0
        
        return {'stop_training': False}
    
    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Save model checkpoint."""
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            logger.info(f"Validation loss improved ({self.best_loss:.6f} → {val_loss:.6f}). Model saved.")
        self.best_loss = val_loss


class ModelCheckpoint:
    """
    Save model checkpoints periodically.
    
    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints
    save_freq : int
        Save every N epochs
    
    Examples
    --------
    >>> callback = ModelCheckpoint(save_dir='./checkpoints', save_freq=10)
    >>> callback(epoch, val_loss, model)
    """
    
    def __init__(self, save_dir: str, save_freq: int = 10):
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, epoch: int, val_loss: float, model: torch.nn.Module) -> Dict[str, Any]:
        """Save checkpoint if needed."""
        if epoch % self.save_freq == 0:
            save_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            logger.info(f"Checkpoint saved: {save_path}")
        
        return {'stop_training': False}
