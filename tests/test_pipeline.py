"""Tests for pipeline orchestration."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from seiscae import Pipeline
from seiscae.config import ConfigManager


class TestPipeline:
    """Test suite for Pipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = ConfigManager()
        pipeline = Pipeline(config)
        
        assert pipeline.config is not None
        assert pipeline.detector is not None
        assert pipeline.spec_generator is not None
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigManager()
        
        # Valid config should not raise
        config.validate()
        
        # Invalid config should raise
        config.set('detection.sta_seconds', 50.0)
        config.set('detection.lta_seconds', 10.0)
        
        with pytest.raises(ValueError):
            config.validate()
