"""Tests for model architectures."""

import pytest
import torch
import numpy as np

from seiscae.models import ConvAutoencoder, get_model


class TestConvAutoencoder:
    """Test suite for ConvAutoencoder."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ConvAutoencoder(latent_dim=16, dropout=0.1)
        
        assert model.latent_dim == 16
        assert model.dropout_p == 0.1
    
    def test_forward_pass(self):
        """Test forward pass with correct shapes."""
        model = ConvAutoencoder(latent_dim=16)
        
        # Input: (batch, channels, height, width)
        x = torch.randn(8, 1, 256, 40)
        
        z, x_recon = model(x)
        
        # Check latent shape
        assert z.shape == (8, 16)
        
        # Check reconstruction shape
        assert x_recon.shape == x.shape
    
    def test_encode_decode(self):
        """Test encode and decode separately."""
        model = ConvAutoencoder(latent_dim=16)
        
        x = torch.randn(4, 1, 256, 40)
        
        # Encode
        z = model.encode(x)
        assert z.shape == (4, 16)
        
        # Decode
        x_recon = model.decode(z)
        assert x_recon.shape == x.shape
    
    def test_get_model_factory(self):
        """Test model factory function."""
        model = get_model('conv_ae', latent_dim=32)
        
        assert isinstance(model, ConvAutoencoder)
        assert model.latent_dim == 32
    
    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            get_model('invalid_model')


class TestTraining:
    """Test suite for training utilities."""
    
    def test_trainer_initialization(self):
        """Test AutoencoderTrainer initialization."""
        from seiscae.training import AutoencoderTrainer
        
        model = ConvAutoencoder(latent_dim=16)
        device = torch.device('cpu')
        
        trainer = AutoencoderTrainer(model, device, learning_rate=1e-4)
        
        assert trainer.device == device
        assert trainer.learning_rate == 1e-4
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        from seiscae.training import AutoencoderTrainer
        
        model = ConvAutoencoder(latent_dim=16)
        device = torch.device('cpu')
        trainer = AutoencoderTrainer(model, device)
        
        # Create synthetic spectrograms
        spectrograms = np.random.rand(10, 256, 40)
        
        # Extract features
        features = trainer.extract_features(spectrograms, batch_size=4)
        
        assert features.shape == (10, 16)
