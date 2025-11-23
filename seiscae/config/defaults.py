"""Default configuration values."""

DEFAULT_CONFIG = {
    # Detection parameters (matching research paper)
    "detection": {
        "sta_seconds": 0.5,
        "lta_seconds": 30.0,
        "threshold_on": 10.0,  # Research paper uses 10
        "threshold_off": 3.0,
        "sampling_rate": 100.0,
        "highpass_freq": 1.0,
        "method": "classic",
        "delta_sta": 20.0,
        "delta_lta": 20.0,
        "epsilon": 2.0,
        "min_event_duration": 0.0,
        "dead_time": 0.0,
    },
    
    # Spectrogram parameters (matching research paper)
    "spectrogram": {
        "nperseg": 128,  # ~1-second segment (adjust based on sampling rate: 512 for 500Hz, 128 for 100Hz)
        "noverlap_ratio": 0.9,  # 90% overlap
        "nfft": 512,  # Keep FFT size for frequency resolution
        "freq_min": 1.0,
        "freq_max": 50.0,
        "time_bins": 40,  # Research paper uses 40
        "freq_bins": 256,  # Research paper uses 256
        "window_seconds": 4.0,  # 4-second window centered on max energy
    },
    
    # Model parameters (matching research paper)
    "model": {
        "latent_dim": 16,  # Research paper uses 16
        "dropout": 0.1,  # Research paper uses 0.1
        "architecture": "conv_ae",
    },
    
    # Training parameters (matching research paper)
    "training": {
        "epochs": 300,
        "batch_size": 128,
        "learning_rate": 1e-4,  # Research paper uses 1e-4
        "patience": 20,  # Early stopping patience
        "validation_split": 0.3,  # 70/30 train/val split
        "num_workers": 4,
    },
    
    # Clustering parameters (matching research paper)
    "clustering": {
        "algorithm": "gmm",
        "n_clusters": 50,  # Research paper uses 50 for fine-grain clustering
        "covariance_type": "full",  # Research paper uses full covariance
    },
    
    # Phase 3 post-clustering analysis
    "phase3": {
        "compute_metrics": True,  # Compute peak_freq, freq_dominance, etc.
        "export_excel": True,  # Export results to Excel
        "suggest_merges": True,  # Suggest cluster merges based on similarity
        "merge_threshold": 0.2,  # Similarity threshold for merge suggestions
    },
    
    # Visualization parameters
    "visualization": {
        "dpi": 300,
        "format": "png",
        "examples_per_cluster": 5,
        # Training diagnostics
        "training_diagnostics": True,
        "plot_loss_curves": True,
        "plot_reconstructions": True,
        "plot_latent_evolution": False,  # Requires saving features at multiple epochs
        # Clustering diagnostics
        "clustering_diagnostics": True,
        "plot_cluster_grids": True,
        "plot_feature_heatmap": True,
        "plot_distance_distributions": True,
    },
    
    # I/O parameters
    "io": {
        "output_base": "./results",
        "save_waveforms": True,
        "save_spectrograms": True,
        "save_diagnostics": True,
        "file_pattern": "*",  # File pattern for detection (e.g., "*.mseed", "*.sac", "*")
        "recursive_search": False,  # Recursively search subdirectories
    },
    
    # Component parameters
    "components": ["EHZ"],
    
    # Hardware parameters
    "hardware": {
        "gpu": -1,  # -1 for CPU, 0+ for GPU ID
        "mixed_precision": False,
    },
}
