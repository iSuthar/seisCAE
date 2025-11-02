"""Default configuration values."""

DEFAULT_CONFIG = {
    # Detection parameters
    "detection": {
        "sta_seconds": 0.5,
        "lta_seconds": 30.0,
        "threshold_on": 25.0,
        "threshold_off": 3.0,
        "sampling_rate": 100.0,
    },
    
    # Spectrogram parameters
    "spectrogram": {
        "nperseg": 128,
        "noverlap_ratio": 0.9,
        "nfft": 512,
        "freq_min": 1.0,
        "freq_max": 50.0,
        "time_bins": 40,
        "freq_bins": 256,
    },
    
    # Model parameters
    "model": {
        "latent_dim": 16,
        "dropout": 0.1,
        "architecture": "conv_ae",
    },
    
    # Training parameters
    "training": {
        "epochs": 300,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "patience": 20,
        "validation_split": 0.3,
        "num_workers": 4,
    },
    
    # Clustering parameters
    "clustering": {
        "algorithm": "gmm",
        "n_clusters": None,  # None = auto-select
        "covariance_type": "full",
        "max_clusters": 20,
    },
    
    # Visualization parameters
    "visualization": {
        "dpi": 300,
        "format": "png",
        "examples_per_cluster": 5,
    },
    
    # I/O parameters
    "io": {
        "output_base": "./results",
        "save_waveforms": True,
        "save_spectrograms": True,
        "save_diagnostics": True,
    },
    
    # Component parameters
    "components": ["EHZ"],
    
    # Hardware parameters
    "hardware": {
        "gpu": -1,  # -1 for CPU, 0+ for GPU ID
        "mixed_precision": False,
    },
}
