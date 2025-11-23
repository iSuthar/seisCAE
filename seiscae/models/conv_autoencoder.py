"""Convolutional autoencoder implementation."""

import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseAutoencoder


class ConvAutoencoder(BaseAutoencoder):
    """
    Convolutional Autoencoder for seismic spectrograms.
    
    This architecture EXACTLY matches the research paper implementation
    for reproducibility and comparison with published results.
    
    Architecture:
        Encoder: Conv2D -> Conv2D -> FC -> FC (to latent)
        Decoder: FC -> FC -> ConvTranspose2D -> ConvTranspose2D
    
    Input shape: (batch, 1, 256, 40)
    Output shape: (batch, 1, 256, 40)
    Latent shape: (batch, latent_dim)
    
    Intermediate dimensions after conv layers: 16 * 28 * 4 = 1792
    This is hardcoded to match the research paper's exact architecture.
    
    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent space (default: 16, matching research paper)
    dropout : float
        Dropout probability (default: 0.1, matching research paper)
    input_channels : int
        Number of input channels (default: 1)
    
    Examples
    --------
    >>> model = ConvAutoencoder(latent_dim=16)
    >>> x = torch.randn(32, 1, 256, 40)
    >>> z, x_recon = model(x)
    >>> print(f"Latent shape: {z.shape}")  # (32, 16)
    >>> print(f"Reconstruction shape: {x_recon.shape}")  # (32, 1, 256, 40)
    
    Notes
    -----
    BatchNorm2d Placement and Purpose:
    
    BatchNorm2d layers are placed after each convolutional layer (both Conv2d 
    and ConvTranspose2d) and normalize activations across the batch dimension.
    
    **Why BatchNorm is critical:**
    1. Stabilizes training by reducing internal covariate shift
    2. Allows higher learning rates (typically 10x higher)
    3. Acts as regularization, reducing need for dropout in some cases
    4. Accelerates convergence (fewer epochs to reach optimal loss)
    5. Reduces sensitivity to weight initialization
    
    **What happens if you REMOVE BatchNorm:**
    1. **Slower convergence**: Training requires 2-3x more epochs
    2. **Lower learning rates needed**: Must reduce LR by ~10x (e.g., 1e-4 -> 1e-5)
    3. **Gradient instability**: Gradients can explode/vanish more easily
    4. **Initialization sensitivity**: Results highly dependent on weight init
    5. **Worse generalization**: Higher validation loss, more overfitting
    6. **Training instability**: Loss curves show more oscillation/noise
    
    The research paper uses BatchNorm2d after conv layers but NOT after
    fully-connected layers (only dropout is used there).
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
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=3, padding=1),
            # BatchNorm2d: Normalizes across batch for each of 8 channels
            # Maintains running mean/var during training for inference stability
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Output: (8, 85, 13)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=3, padding=1),
            # BatchNorm2d: Critical for stable gradient flow through deeper layers
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Output: (16, 28, 4) -> Flattened: 16*28*4 = 1792
        )
        
        # Fully-connected encoder layers
        # Note: NO BatchNorm here, only dropout for regularization
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 28 * 4, 128),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        
        # Decoder
        # Fully-connected decoder layers
        self.fc3 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(128, 16 * 28 * 4),
            nn.ReLU(),
        )
        
        # Transposed convolutions for upsampling
        self.conv3 = nn.Sequential(
            # Input: (16, 28, 4)
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1, output_padding=(1, 1)),
            # BatchNorm2d: Stabilizes reconstruction process
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Output: (8, 85, 13)
        )
        
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(8, input_channels, kernel_size=5, stride=3, padding=1, output_padding=(1, 1)),
            # BatchNorm2d: Even on final layer for consistent activation scale
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            # Output: (1, 256, 40)
        )
        
        # Dropout: Applied after flattening and between FC layers
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Follows research paper's exact forward pass through encoder.
        """
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Flatten: (batch, 16, 28, 4) -> (batch, 1, 1, 1792)
        x = x.view(x.size()[0], 1, 1, -1)
        x = self.dropout(x)
        
        # Fully-connected compression
        x = self.fc1(x)
        x = self.dropout(x)
        z = self.fc2(x)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation.
        
        Follows research paper's exact forward pass through decoder.
        """
        # Fully-connected expansion
        x = self.dropout(z)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        
        # Reshape: (batch, 1, 1, 1792) -> (batch, 16, 28, 4)
        x = x.view(-1, 16, 28, 4)
        
        # Transposed convolutions for upsampling
        x = self.conv3(x)
        x = self.conv4(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode."""
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
