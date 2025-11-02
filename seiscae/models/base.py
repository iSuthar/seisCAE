"""Base autoencoder interface."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple


class BaseAutoencoder(ABC, nn.Module):
    """
    Abstract base class for autoencoder models.
    
    All autoencoder models should inherit from this class and implement
    the abstract methods.
    """
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        z : torch.Tensor
            Latent representation
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation
            
        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: return (latent, reconstruction).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        z : torch.Tensor
            Latent representation
        x_recon : torch.Tensor
            Reconstructed input
        """
        pass
    
    def get_latent_dim(self) -> int:
        """
        Get latent space dimensionality.
        
        Returns
        -------
        latent_dim : int
            Dimensionality of latent space
        """
        raise NotImplementedError("Subclass must implement get_latent_dim()")
