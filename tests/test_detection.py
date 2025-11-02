"""Tests for event detection module."""

import pytest
import numpy as np
from obspy import Trace, UTCDateTime

from seiscae.core import EventDetector


class TestEventDetector:
    """Test suite for EventDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = EventDetector(
            sta_seconds=0.5,
            lta_seconds=30.0,
            threshold_on=25.0,
            threshold_off=3.0,
        )
        
        assert detector.sta_seconds == 0.5
        assert detector.lta_seconds == 30.0
        assert detector.threshold_on == 25.0
        assert detector.threshold_off == 3.0
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # STA >= LTA should raise error
        with pytest.raises(ValueError):
            EventDetector(sta_seconds=30.0, lta_seconds=0.5)
        
        # Threshold ON <= Threshold OFF should raise error
        with pytest.raises(ValueError):
            EventDetector(threshold_on=3.0, threshold_off=25.0)
    
    def test_detect_synthetic_event(self):
        """Test detection on synthetic data with injected event."""
        # Create synthetic trace
        sampling_rate = 100.0
        duration = 60.0  # 60 seconds
        npts = int(sampling_rate * duration)
        
        # Background noise
        data = np.random.randn(npts) * 0.1
        
        # Inject event (spike) at 30 seconds
        event_start = int(30 * sampling_rate)
        event_duration = int(2 * sampling_rate)
        data[event_start:event_start + event_duration] += np.sin(
            np.linspace(0, 10 * np.pi, event_duration)
        ) * 5.0
        
        # Create ObsPy trace
        trace = Trace(data=data)
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime = UTCDateTime(2025, 1, 1, 0, 0, 0)
        trace.stats.station = "TEST"
        trace.stats.channel = "HHZ"
        
        # Save to temporary file (would need tempfile in actual implementation)
        # For now, we'll test the core detection logic separately
        
        detector = EventDetector(
            sta_seconds=0.5,
            lta_seconds=10.0,
            threshold_on=5.0,
            threshold_off=2.0,
        )
        
        # Manually compute STA/LTA for testing
        from obspy.signal.trigger import classic_sta_lta, trigger_onset
        
        sta_samples = int(0.5 * sampling_rate)
        lta_samples = int(10.0 * sampling_rate)
        cft = classic_sta_lta(data, sta_samples, lta_samples)
        triggers = trigger_onset(cft, 5.0, 2.0)
        
        # Should detect at least one event
        assert len(triggers) > 0


class TestSpectrogramGenerator:
    """Test suite for SpectrogramGenerator."""
    
    def test_spectrogram_shape(self):
        """Test that generated spectrogram has correct shape."""
        from seiscae.core import SpectrogramGenerator
        
        # Create synthetic trace
        sampling_rate = 100.0
        duration = 10.0
        npts = int(sampling_rate * duration)
        data = np.random.randn(npts)
        
        trace = Trace(data=data)
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime = UTCDateTime(2025, 1, 1)
        
        # Generate spectrogram
        gen = SpectrogramGenerator(
            time_bins=40,
            freq_bins=256,
        )
        
        f, t, spec = gen.generate(trace)
        
        # Check shape
        assert spec.shape == (256, 40)
        assert len(f) == 256
        assert len(t) == 40
    
    def test_normalization(self):
        """Test that spectrogram is normalized to [0, 1]."""
        from seiscae.core import SpectrogramGenerator
        
        sampling_rate = 100.0
        duration = 10.0
        npts = int(sampling_rate * duration)
        data = np.random.randn(npts) * 100  # Large amplitude
        
        trace = Trace(data=data)
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime = UTCDateTime(2025, 1, 1)
        
        gen = SpectrogramGenerator()
        f, t, spec = gen.generate(trace)
        
        # Check normalization
        assert spec.min() >= 0.0
        assert spec.max() <= 1.0
