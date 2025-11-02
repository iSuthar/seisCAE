"""Convolutional autoencoder implementation."""

import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseAutoencoder


class ConvAutoencoder(BaseAutoencoder):
    """
    Convolutional Autoencoder for seismic spectrograms.
    
    Architecture:
        Encoder: Conv2D -> Conv2D -> FC -> FC (to latent)
        Decoder: FC -> FC -> ConvTranspose2D -> ConvTranspose2D
    
    Input shape: (batch, 1, 256, 40)
    Output shape: (batch, 1, 256, 40)
    Latent shape: (batch, latent_dim)
    
    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent space (default: 16)
    dropout : float
        Dropout probability (default: 0.1)
    input_channels : int
        Number of input channels (default: 1)
    
    Examples
    --------
    >>> model = ConvAutoencoder(latent_dim=16)
    >>> x = torch.randn(32, 1, 256, 40)
    >>> z, x_recon = model(x)
    >>> print(f"Latent shape: {z.shape}")  # (32, 16)
    >>> print(f"Reconstruction shape: {x_recon.shape}")  # (32, 1, 256, 40)
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        dropout: float = 0.1,
        input_channels: int = 1,
    ):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        
        # Encoder
        # Input: (1, 256, 40)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Output: (8, 85, 13)
            
            nn.Conv2d(8, 16, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Output: (16, 28, 4)
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(16 * 28 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        
        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 16 * 28 * 4),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # Input: (16, 28, 4)
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Output: (8, 85, 13)
            
            nn.ConvTranspose2d(8, input_channels, kernel_size=5, stride=3, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            # Output: (1, 256, 40)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        z = self.fc_encoder(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        x = self.fc_decoder(z)
        x = self.dropout(x)
        x = x.view(-1, 16, 28, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon
    
    def get_latent_dim(self) -> int:
        """Get latent space dimensionality."""
        return self.latent_dim


# Model registry for easy access
MODELS = {
    'conv_ae': ConvAutoencoder,
    # Future models can be added here:
    # 'resnet_ae': ResNetAutoencoder,
    # 'vae': VariationalAutoencoder,
}


def get_model(name: str, **kwargs) -> BaseAutoencoder:
    """
    Factory function to get model by name.
    
    Parameters
    ----------
    name : str
        Model name (e.g., 'conv_ae')
    **kwargs
        Model-specific parameters
        
    Returns
    -------
    model : BaseAutoencoder
        Initialized model
    
    Examples
    --------
    >>> model = get_model('conv_ae', latent_dim=32, dropout=0.2)
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)
