"""Model training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import logging

from ..models.base import BaseAutoencoder
from .callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    Trainer for autoencoder models.
    
    Parameters
    ----------
    model : BaseAutoencoder
        Model to train
    device : torch.device
        Device to train on
    learning_rate : float
        Learning rate
    
    Examples
    --------
    >>> model = ConvAutoencoder(latent_dim=16)
    >>> trainer = AutoencoderTrainer(model, torch.device('cuda:0'))
    >>> history = trainer.train(spectrograms, epochs=300)
    """
    
    def __init__(
        self,
        model: BaseAutoencoder,
        device: torch.device,
        learning_rate: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
    
    def train(
        self,
        spectrograms: np.ndarray,
        epochs: int = 300,
        batch_size: int = 128,
        validation_split: float = 0.3,
        patience: int = 20,
        num_workers: int = 4,
        save_dir: Optional[str] = None,
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Train the autoencoder.
        
        Parameters
        ----------
        spectrograms : np.ndarray
            Training spectrograms, shape (N, H, W) or (N, C, H, W)
        epochs : int
            Maximum number of epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        patience : int
            Early stopping patience
        num_workers : int
            DataLoader workers
        save_dir : str, optional
            Directory to save checkpoints
        callbacks : list, optional
            Additional callbacks
            
        Returns
        -------
        history : dict
            Training history
        """
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(
            spectrograms, batch_size, validation_split, num_workers
        )
        
        # Setup callbacks
        callbacks = callbacks or []
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                EarlyStopping(patience=patience, save_path=Path(save_dir) / "best_model.pt")
            )
            callbacks.append(
                ModelCheckpoint(save_dir=save_dir, save_freq=10)
            )
        else:
            callbacks.append(EarlyStopping(patience=patience))
        
        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            val_loss = self._validate_epoch(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )
            
            # Callback execution
            stop_training = False
            for callback in callbacks:
                if hasattr(callback, '__call__'):
                    result = callback(epoch, val_loss, self.model)
                    if result and result.get('stop_training'):
                        stop_training = True
                        break
            
            if stop_training:
                logger.info("Early stopping triggered")
                break
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': epoch,
        }
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        for batch_idx, (data,) in enumerate(pbar):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            _, recon = self.model(data)
            loss = self.criterion(data, recon)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(self.device)
                _, recon = self.model(data)
                loss = self.criterion(data, recon)
                epoch_loss += loss.item()
        
        return epoch_loss / len(val_loader)
    
    def _prepare_data(
        self,
        spectrograms: np.ndarray,
        batch_size: int,
        validation_split: float,
        num_workers: int,
    ) -> tuple:
        """Prepare train and validation dataloaders."""
        # Convert to tensor and add channel dimension if needed
        if spectrograms.ndim == 3:
            spectrograms = spectrograms[:, np.newaxis, :, :]
        
        dataset = TensorDataset(torch.tensor(spectrograms).float())
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        return train_loader, val_loader
    
    def extract_features(self, spectrograms: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract latent features from spectrograms.
        
        Parameters
        ----------
        spectrograms : np.ndarray
            Input spectrograms
        batch_size : int
            Batch size for inference
            
        Returns
        -------
        features : np.ndarray
            Latent features
        """
        self.model.eval()
        
        if spectrograms.ndim == 3:
            spectrograms = spectrograms[:, np.newaxis, :, :]
        
        dataset = TensorDataset(torch.tensor(spectrograms).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        features = []
        with torch.no_grad():
            for data, in tqdm(loader, desc="Extracting features"):
                data = data.to(self.device)
                z, _ = self.model(data)
                features.append(z.cpu().numpy())
        
        return np.vstack(features)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
