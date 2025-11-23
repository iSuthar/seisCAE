"""Main pipeline orchestrator for seisCAE."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from tqdm import tqdm

from .core.detection import EventDetector
from .core.preprocessing import SpectrogramGenerator, EventExtractor
from .core.catalog import EventCatalog
from .models import get_model
from .training.trainer import AutoencoderTrainer
from .clustering import get_clusterer
from .clustering.metrics_phase3 import SpectrogramMetrics, ClusterAnalyzer
from .visualization import Visualizer
from .visualization.training_diagnostics import (
    plot_loss_curves_detailed,
    plot_reconstruction_comparison,
    plot_cluster_grid,
    plot_cluster_feature_heatmap,
    plot_distance_distributions,
)
from .config import ConfigManager
from .utils.device import get_device

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main seisCAE pipeline orchestrator.
    
    This class coordinates the entire workflow:
    1. Event detection (STA/LTA)
    2. Spectrogram generation
    3. Autoencoder training
    4. Feature extraction
    5. Clustering
    6. Visualization
    
    Parameters
    ----------
    config : ConfigManager
        Configuration manager
    
    Examples
    --------
    >>> from seiscae import Pipeline, load_config
    >>> config = load_config("configs/default.yaml")
    >>> pipeline = Pipeline(config)
    >>> results = pipeline.run(data_path="/path/to/data")
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = get_device(config.get('hardware.gpu', -1))
        
        # Initialize components
        self.detector = None
        self.spec_generator = None
        self.event_extractor = None
        self.model = None
        self.trainer = None
        self.clusterer = None
        self.visualizer = None
        
        # Results
        self.catalog = None
        self.features = None
        self.labels = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components from config."""
        # Event detector
        self.detector = EventDetector(
            sta_seconds=self.config.get('detection.sta_seconds'),
            lta_seconds=self.config.get('detection.lta_seconds'),
            threshold_on=self.config.get('detection.threshold_on'),
            threshold_off=self.config.get('detection.threshold_off'),
            highpass_freq=self.config.get('detection.highpass_freq'),
            method=self.config.get('detection.method'),
            delta_sta=self.config.get('detection.delta_sta'),
            delta_lta=self.config.get('detection.delta_lta'),
            epsilon=self.config.get('detection.epsilon'),
            min_event_duration=self.config.get('detection.min_event_duration'),
            dead_time=self.config.get('detection.dead_time'),
        )
        
        # Spectrogram generator
        self.spec_generator = SpectrogramGenerator(
            nperseg=self.config.get('spectrogram.nperseg'),
            noverlap_ratio=self.config.get('spectrogram.noverlap_ratio'),
            nfft=self.config.get('spectrogram.nfft'),
            freq_min=self.config.get('spectrogram.freq_min'),
            freq_max=self.config.get('spectrogram.freq_max'),
            time_bins=self.config.get('spectrogram.time_bins'),
            freq_bins=self.config.get('spectrogram.freq_bins'),
            window_seconds=self.config.get('spectrogram.window_seconds'),
        )
        
        # Event extractor
        self.event_extractor = EventExtractor(self.spec_generator)
        
        # Visualizer
        self.visualizer = Visualizer(self.config)
        
        logger.info("Pipeline components initialized")
    
    def run(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        skip_detection: bool = False,
        skip_training: bool = False,
        skip_clustering: bool = False,
    ) -> "PipelineResults":
        """
        Run the complete pipeline.
        
        Parameters
        ----------
        data_path : str
            Path to seismic data directory or catalog file
        output_dir : str, optional
            Output directory (uses config if not specified)
        skip_detection : bool
            Skip detection if catalog already exists
        skip_training : bool
            Skip training if model already exists
        skip_clustering : bool
            Skip clustering
            
        Returns
        -------
        results : PipelineResults
            Pipeline results object
        """
        output_dir = output_dir or self.config.get('io.output_base')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("seisCAE Pipeline Starting")
        logger.info("=" * 80)
        
        # Stage 1: Event Detection
        if not skip_detection:
            logger.info("\n[Stage 1/5] Event Detection")
            self.detect_events(data_path, output_path)
        else:
            logger.info("\n[Stage 1/5] Skipping detection, loading existing catalog")
            self.catalog = EventCatalog.load(output_path / "catalog.pkl")
        
        # Stage 2: Spectrogram Generation (already done in detection)
        logger.info(f"\n[Stage 2/5] Spectrograms Generated: {len(self.catalog)} events")
        
        # Stage 3: Model Training
        if not skip_training:
            logger.info("\n[Stage 3/5] Model Training")
            self.train_model(output_path)
        else:
            logger.info("\n[Stage 3/5] Skipping training, loading existing model")
            self.load_model(output_path / "models" / "best_model.pt")
        
        # Stage 4: Feature Extraction
        logger.info("\n[Stage 4/5] Feature Extraction")
        self.extract_features()
        
        # Stage 5: Clustering
        if not skip_clustering:
            logger.info("\n[Stage 5/5] Clustering")
            self.cluster_events()
            
            # Phase 3: Post-clustering analysis
            if self.config.get('phase3.compute_metrics', False):
                logger.info("\n[Phase 3] Post-Clustering Analysis")
                self.run_phase3_analysis(output_path)
        
        # Save results
        logger.info("\nSaving results...")
        self.save_results(output_path)
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        self.generate_visualizations(output_path)
        
        logger.info("=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        
        return PipelineResults(
            catalog=self.catalog,
            features=self.features,
            labels=self.labels,
            model=self.model,
            clusterer=self.clusterer,
            output_dir=output_path,
        )
    
    def detect_events(self, data_path: str, output_path: Path) -> None:
        """
        Stage 1: Detect events and generate spectrograms.
        
        Supports flexible directory structures:
        - Flat: all files in data_path
        - Component-based: files organized by component subdirectories
        - Mixed: combination of both
        
        The method will attempt to process files based on the 'components'
        configuration. If component subdirectories don't exist, it falls back
        to processing all files in the main directory.
        """
        components = self.config.get('components', ['EHZ'])
        file_pattern = self.config.get('io.file_pattern', '*')
        recursive_search = self.config.get('io.recursive_search', False)
        
        all_events = []
        data_path_obj = Path(data_path)
        
        # Check if component-based structure exists
        component_dirs_exist = all(
            (data_path_obj / comp).exists() for comp in components
        )
        
        if component_dirs_exist:
            # Process by component subdirectories
            logger.info("Processing component-based directory structure")
            for component in components:
                logger.info(f"Processing component: {component}")
                
                results = self.detector.detect_directory(
                    directory=data_path,
                    component=component,
                    pattern=file_pattern,
                    recursive=recursive_search,
                )
                
                # Extract events and generate spectrograms
                for filepath, stream, cft, triggers in tqdm(results, desc=f"{component} events"):
                    trace = stream[0]
                    events = self.event_extractor.extract_events(trace, triggers, component)
                    all_events.extend(events)
                    
                    # Optional: Save diagnostic plots
                    if self.config.get('io.save_diagnostics'):
                        diag_path = output_path / "diagnostics" / component
                        diag_path.mkdir(parents=True, exist_ok=True)
                        self.visualizer.plot_detection_summary(
                            trace, cft, triggers,
                            save_path=diag_path / f"{Path(filepath).stem}.png"
                        )
        else:
            # Process flat directory structure - all files in one directory
            logger.info(
                "Component subdirectories not found. "
                "Processing all files in flat directory structure."
            )
            
            results = self.detector.detect_directory(
                directory=data_path,
                component=None,
                pattern=file_pattern,
                recursive=recursive_search,
            )
            
            # Extract events without component-specific labeling
            # Try to infer component from channel name if possible
            for filepath, stream, cft, triggers in tqdm(results, desc="Processing files"):
                trace = stream[0]
                # Try to extract component from channel name (e.g., BHZ -> Z, EHN -> N)
                channel = trace.stats.channel
                component = channel[-1] if channel else "Z"  # Default to Z
                
                events = self.event_extractor.extract_events(trace, triggers, component)
                all_events.extend(events)
                
                # Optional: Save diagnostic plots
                if self.config.get('io.save_diagnostics'):
                    diag_path = output_path / "diagnostics"
                    diag_path.mkdir(parents=True, exist_ok=True)
                    self.visualizer.plot_detection_summary(
                        trace, cft, triggers,
                        save_path=diag_path / f"{Path(filepath).stem}.png"
                    )
        
        self.catalog = EventCatalog(all_events)
        logger.info(f"Total events detected: {len(self.catalog)}")
    
    def train_model(self, output_path: Path) -> None:
        """Stage 3: Train autoencoder model."""
        # Initialize model
        self.model = get_model(
            self.config.get('model.architecture'),
            latent_dim=self.config.get('model.latent_dim'),
            dropout=self.config.get('model.dropout'),
        )
        
        # Initialize trainer
        self.trainer = AutoencoderTrainer(
            model=self.model,
            device=self.device,
            learning_rate=self.config.get('training.learning_rate'),
        )
        
        # Get spectrograms from catalog
        spectrograms = np.array([event['spectrogram'] for event in self.catalog.events])
        
        # Train
        history = self.trainer.train(
            spectrograms=spectrograms,
            epochs=self.config.get('training.epochs'),
            batch_size=self.config.get('training.batch_size'),
            validation_split=self.config.get('training.validation_split'),
            patience=self.config.get('training.patience'),
            num_workers=self.config.get('training.num_workers'),
            save_dir=str(output_path / "models"),
        )
        
        # Plot training history
        self.visualizer.plot_training_history(
            history,
            save_path=output_path / "visualizations" / "training_history.png"
        )
    
    def extract_features(self) -> None:
        """Stage 4: Extract latent features."""
        spectrograms = np.array([event['spectrogram'] for event in self.catalog.events])
        self.features = self.trainer.extract_features(spectrograms)
        logger.info(f"Extracted features: {self.features.shape}")
    
    def cluster_events(self) -> None:
        """Stage 5: Cluster events."""
        self.clusterer = get_clusterer(
            self.config.get('clustering.algorithm'),
            n_clusters=self.config.get('clustering.n_clusters'),
            covariance_type=self.config.get('clustering.covariance_type'),
        )
        
        self.labels = self.clusterer.fit_predict(self.features)
        logger.info(f"Clustering complete: {self.clusterer.n_clusters} clusters")
    
    def run_phase3_analysis(self, output_path: Path) -> None:
        """
        Phase 3: Post-clustering analysis with physical metrics.
        
        Computes spectral and temporal metrics from spectrograms,
        calculates distances to cluster centers, suggests cluster
        merges, and exports complete analysis to Excel.
        """
        logger.info("Computing Phase 3 metrics...")
        
        # Get cluster centers
        cluster_centers = self.clusterer.get_cluster_centers(original_scale=False)
        
        # Initialize analyzer
        analyzer = ClusterAnalyzer(
            labels=self.labels,
            features=self.features,
            cluster_centers=cluster_centers,
        )
        
        # Get spectrograms and frequency bins
        spectrograms = np.array([event['spectrogram'] for event in self.catalog.events])
        
        # Generate frequency bins for metrics
        freq_min = self.config.get('spectrogram.freq_min')
        freq_max = self.config.get('spectrogram.freq_max')
        freq_bins_count = self.config.get('spectrogram.freq_bins')
        freq_bins = np.linspace(freq_min, freq_max, freq_bins_count)
        
        # Compute metrics for all spectrograms
        metrics_df = analyzer.compute_cluster_metrics(spectrograms, freq_bins)
        
        # Compute distances to centers
        distances = analyzer.compute_distances_to_centers()
        
        # Get cluster statistics
        cluster_stats = analyzer.get_cluster_statistics(metrics_df)
        logger.info(f"Cluster statistics computed for {len(cluster_stats)} clusters")
        
        # Suggest merges if enabled
        if self.config.get('phase3.suggest_merges', False):
            threshold = self.config.get('phase3.merge_threshold', 0.2)
            suggestions = analyzer.suggest_merges(cluster_stats, threshold=threshold)
            
            if suggestions:
                logger.info(f"Suggested {len(suggestions)} cluster merges:")
                for c1, c2, sim in suggestions[:10]:  # Show top 10
                    logger.info(f"  Clusters {c1} <-> {c2} (similarity: {sim:.4f})")
                
                # Save suggestions
                suggestions_df = pd.DataFrame(
                    suggestions,
                    columns=['cluster_1', 'cluster_2', 'similarity']
                )
                suggestions_path = output_path / "cluster_merge_suggestions.csv"
                suggestions_df.to_csv(suggestions_path, index=False)
                logger.info(f"Merge suggestions saved to {suggestions_path}")
        
        # Export to Excel if enabled
        if self.config.get('phase3.export_excel', False):
            catalog_df = self.catalog.to_dataframe(include_arrays=False)
            excel_path = output_path / "phase3_analysis.xlsx"
            
            analyzer.export_to_excel(
                metrics_df=metrics_df,
                catalog_df=catalog_df,
                output_path=str(excel_path),
                distances=distances,
            )
            logger.info(f"Phase 3 analysis exported to {excel_path}")
        
        # Store for visualization
        self.phase3_metrics = metrics_df
        self.phase3_cluster_stats = cluster_stats
        self.phase3_distances = distances
        
        logger.info("Phase 3 analysis complete")
    
    def generate_visualizations(self, output_path: Path) -> None:
        """Generate all diagnostic visualizations."""
        viz_path = output_path / "visualizations"
        viz_path.mkdir(parents=True, exist_ok=True)
        
        # Standard visualizations
        self.visualizer.plot_all(
            catalog=self.catalog,
            features=self.features,
            labels=self.labels,
            output_dir=viz_path,
        )
        
        # Training diagnostics
        if self.config.get('visualization.training_diagnostics', False):
            logger.info("Generating training diagnostics...")
            
            if hasattr(self.trainer, 'train_losses'):
                history = {
                    'train_losses': self.trainer.train_losses,
                    'val_losses': self.trainer.val_losses,
                }
                
                if self.config.get('visualization.plot_loss_curves', True):
                    plot_loss_curves_detailed(
                        history,
                        save_path=viz_path / "training_loss_detailed.png",
                        log_scale=True,
                    )
                
                if self.config.get('visualization.plot_reconstructions', True):
                    # Get sample spectrograms
                    spectrograms = np.array([event['spectrogram'] 
                                            for event in self.catalog.events[:8]])
                    spectrograms = torch.tensor(spectrograms[:, np.newaxis, :, :]).float()
                    
                    self.model.eval()
                    with torch.no_grad():
                        spectrograms = spectrograms.to(self.device)
                        _, reconstructed = self.model(spectrograms)
                    
                    plot_reconstruction_comparison(
                        spectrograms,
                        reconstructed,
                        n_examples=8,
                        save_path=viz_path / "reconstructions.png",
                    )
        
        # Clustering diagnostics
        if self.config.get('visualization.clustering_diagnostics', False):
            logger.info("Generating clustering diagnostics...")
            
            spectrograms = np.array([event['spectrogram'] for event in self.catalog.events])
            waveforms = np.array([event['waveform'] for event in self.catalog.events])
            
            # Cluster grids
            if self.config.get('visualization.plot_cluster_grids', True):
                n_clusters_to_plot = min(10, self.clusterer.n_clusters)
                for cluster_id in range(n_clusters_to_plot):
                    plot_cluster_grid(
                        cluster_id=cluster_id,
                        spectrograms=spectrograms,
                        waveforms=waveforms,
                        labels=self.labels,
                        reconstructed=None,
                        n_examples=5,
                        save_path=viz_path / f"cluster_{cluster_id}_grid.png",
                    )
            
            # Phase 3 visualizations
            if hasattr(self, 'phase3_cluster_stats'):
                if self.config.get('visualization.plot_feature_heatmap', True):
                    plot_cluster_feature_heatmap(
                        self.phase3_cluster_stats,
                        save_path=viz_path / "phase3_metrics_heatmap.png",
                    )
                
                if self.config.get('visualization.plot_distance_distributions', True):
                    plot_distance_distributions(
                        self.phase3_distances,
                        self.labels,
                        n_clusters=min(20, self.clusterer.n_clusters),
                        save_path=viz_path / "distance_distributions.png",
                    )
        
        logger.info(f"Visualizations saved to {viz_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model."""
        self.model = get_model(
            self.config.get('model.architecture'),
            latent_dim=self.config.get('model.latent_dim'),
            dropout=self.config.get('model.dropout'),
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.trainer = AutoencoderTrainer(
            model=self.model,
            device=self.device,
            learning_rate=self.config.get('training.learning_rate'),
        )
        
        logger.info(f"Model loaded from {model_path}")
    
    def save_results(self, output_path: Path) -> None:
        """Save pipeline results."""
        # Save catalog
        self.catalog.save(output_path / "catalog.pkl")
        self.catalog.save(output_path / "catalog.csv")
        
        # Save features
        np.save(output_path / "features.npy", self.features)
        
        # Save labels
        if self.labels is not None:
            np.save(output_path / "labels.npy", self.labels)
            
            # Save cluster summary
            df = self.catalog.to_dataframe()
            df['cluster'] = self.labels
            df.to_csv(output_path / "results_with_clusters.csv", index=False)
            
            # Save cluster centers
            if hasattr(self.clusterer, 'get_cluster_centers'):
                centers = self.clusterer.get_cluster_centers()
                np.save(output_path / "cluster_centers.npy", centers)
        
        # Save config
        self.config.save(str(output_path / "config_used.yaml"))
        
        logger.info(f"Results saved to {output_path}")


class PipelineResults:
    """
    Container for pipeline results.
    
    Attributes
    ----------
    catalog : EventCatalog
        Event catalog
    features : np.ndarray
        Latent features
    labels : np.ndarray
        Cluster labels
    model : torch.nn.Module
        Trained model
    clusterer : BaseClusterer
        Fitted clusterer
    output_dir : Path
        Output directory
    n_events : int
        Number of events
    n_clusters : int
        Number of clusters
    """
    
    def __init__(
        self,
        catalog: EventCatalog,
        features: np.ndarray,
        labels: Optional[np.ndarray],
        model: torch.nn.Module,
        clusterer: Any,
        output_dir: Path,
    ):
        self.catalog = catalog
        self.features = features
        self.labels = labels
        self.model = model
        self.clusterer = clusterer
        self.output_dir = output_dir
        self.n_events = len(catalog)
        self.n_clusters = len(np.unique(labels)) if labels is not None else None
    
    def get_cluster(self, cluster_id: int) -> EventCatalog:
        """
        Get events for a specific cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID
            
        Returns
        -------
        catalog : EventCatalog
            Catalog containing only events from specified cluster
        """
        if self.labels is None:
            raise ValueError("No clustering labels available")
        
        mask = self.labels == cluster_id
        cluster_events = [e for i, e in enumerate(self.catalog.events) if mask[i]]
        return EventCatalog(cluster_events)
    
    def __repr__(self) -> str:
        return (
            f"PipelineResults(\n"
            f"  events={self.n_events},\n"
            f"  clusters={self.n_clusters},\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )
