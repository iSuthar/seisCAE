"""Input validation utilities."""

import numpy as np
from typing import Tuple


def validate_spectrogram_shape(
    spectrograms: np.ndarray, 
    expected_shape: Tuple[int, int] = (256, 40)
) -> None:
    """
    Validate spectrogram array shape.
    
    Parameters
    ----------
    spectrograms : np.ndarray
        Spectrogram array
    expected_shape : tuple
        Expected (height, width) shape
    
    Raises
    ------
    ValueError
        If shape is invalid
    
    Examples
    --------
    >>> validate_spectrogram_shape(spectrograms, (256, 40))
    """
    if spectrograms.ndim not in [3, 4]:
        raise ValueError(
            f"Spectrograms must be 3D (N, H, W) or 4D (N, C, H, W), "
            f"got shape {spectrograms.shape}"
        )
    
    if spectrograms.ndim == 3:
        _, h, w = spectrograms.shape
    else:
        _, _, h, w = spectrograms.shape
    
    if (h, w) != expected_shape:
        raise ValueError(
            f"Expected spectrogram shape {expected_shape}, got ({h}, {w})"
        )


def validate_features_shape(features: np.ndarray, min_samples: int = 10) -> None:
    """
    Validate feature array shape.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array
    min_samples : int
        Minimum number of samples required
    
    Raises
    ------
    ValueError
        If shape is invalid
    
    Examples
    --------
    >>> validate_features_shape(features, min_samples=10)
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D (N, D), got shape {features.shape}")
    
    n_samples, n_features = features.shape
    
    if n_samples < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples, got {n_samples}"
        )
    
    if n_features == 0:
        raise ValueError("Feature dimension cannot be 0")
