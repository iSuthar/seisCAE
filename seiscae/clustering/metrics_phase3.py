"""Phase 3 post-clustering metrics and analysis.

This module implements the Phase 3 workflow from the research paper:
computing physical metrics from spectrograms for cluster refinement
and manual merging.
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpectrogramMetrics:
    """
    Calculate physical metrics from spectrograms for cluster refinement.
    
    These metrics capture the spectral and temporal characteristics of
    seismic events and are used for manual cluster merging based on
    physical similarity (research paper Phase 3).
    
    Examples
    --------
    >>> metrics = SpectrogramMetrics()
    >>> spec = np.random.rand(256, 40)  # freq x time
    >>> freq_bins = np.linspace(1, 50, 256)
    >>> result = metrics.compute_all(spec, freq_bins)
    >>> print(result['peak_freq'], result['freq_dominance'])
    """
    
    @staticmethod
    def peak_frequency(
        spectrogram: np.ndarray,
        freq_bins: np.ndarray
    ) -> float:
        """
        Get frequency with highest energy.
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Spectrogram array, shape (freq_bins, time_bins)
        freq_bins : np.ndarray
            Frequency values for each frequency bin
            
        Returns
        -------
        peak_freq : float
            Frequency (in Hz) with maximum total energy
        """
        # Sum over time axis to get energy per frequency
        freq_energy = np.sum(spectrogram, axis=-1)
        
        # Find peak
        peak_idx = np.argmax(freq_energy)
        
        return freq_bins[peak_idx]
    
    @staticmethod
    def peak_frequencies_multi(
        spectrogram: np.ndarray,
        freq_bins: np.ndarray,
        n_peaks: int = 3,
        prominence: float = 2.0
    ) -> List[Tuple[float, float]]:
        """
        Get multiple peak frequencies with their amplitudes.
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Spectrogram array, shape (freq_bins, time_bins)
        freq_bins : np.ndarray
            Frequency values for each frequency bin
        n_peaks : int
            Number of peaks to return
        prominence : float
            Minimum prominence for peak detection
            
        Returns
        -------
        peaks : list of (freq, amplitude) tuples
            Top n_peaks sorted by amplitude
        """
        freq_energy = np.sum(spectrogram, axis=-1)
        
        # Find peaks
        peak_indices, properties = find_peaks(
            freq_energy,
            prominence=prominence
        )
        
        if len(peak_indices) == 0:
            # No peaks found, return global maximum
            peak_idx = np.argmax(freq_energy)
            return [(freq_bins[peak_idx], freq_energy[peak_idx])]
        
        # Sort peaks by amplitude
        peak_amps = freq_energy[peak_indices]
        sorted_indices = np.argsort(peak_amps)[::-1]
        
        peaks = []
        for i in sorted_indices[:n_peaks]:
            idx = peak_indices[i]
            peaks.append((freq_bins[idx], freq_energy[idx]))
        
        return peaks
    
    @staticmethod
    def frequency_dominance(spectrogram: np.ndarray) -> float:
        """
        Calculate spectral maximum to mean ratio.
        
        Measures how concentrated the spectral energy is at specific
        frequencies (higher = more tonal, lower = more broadband).
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Spectrogram array, shape (freq_bins, time_bins)
            
        Returns
        -------
        dominance : float
            Ratio of max to mean spectral energy
        """
        freq_energy = np.sum(spectrogram, axis=-1)
        
        mean_energy = np.mean(freq_energy)
        if mean_energy < 1e-10:
            return 0.0
        
        return np.max(freq_energy) / mean_energy
    
    @staticmethod
    def temporal_concentration(spectrogram: np.ndarray) -> float:
        """
        Calculate maximum to mean energy ratio along time window.
        
        Measures how impulsive the signal is in time (higher = more
        impulsive, lower = more sustained).
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Spectrogram array, shape (freq_bins, time_bins)
            
        Returns
        -------
        concentration : float
            Ratio of max to mean temporal energy
        """
        time_energy = np.sum(spectrogram, axis=0)  # Sum over frequency
        
        mean_energy = np.mean(time_energy)
        if mean_energy < 1e-10:
            return 0.0
        
        return np.max(time_energy) / mean_energy
    
    @staticmethod
    def temporal_mad(spectrogram: np.ndarray) -> float:
        """
        Calculate Median Absolute Deviation of energy along time axis.
        
        Robust measure of temporal dispersion (higher = more variable
        energy over time).
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Spectrogram array, shape (freq_bins, time_bins)
            
        Returns
        -------
        mad : float
            Median Absolute Deviation of temporal energy
        """
        time_energy = np.sum(spectrogram, axis=0)
        return median_abs_deviation(time_energy)
    
    @classmethod
    def compute_all(
        cls,
        spectrogram: np.ndarray,
        freq_bins: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all Phase 3 metrics for a single spectrogram.
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Spectrogram array, shape (freq_bins, time_bins)
        freq_bins : np.ndarray
            Frequency values for each frequency bin
            
        Returns
        -------
        metrics : dict
            Dictionary with all metric values
        """
        return {
            'peak_freq': cls.peak_frequency(spectrogram, freq_bins),
            'freq_dominance': cls.frequency_dominance(spectrogram),
            'temporal_concentration': cls.temporal_concentration(spectrogram),
            'temporal_mad': cls.temporal_mad(spectrogram),
        }


class ClusterAnalyzer:
    """
    Analyze clusters using Phase 3 metrics and latent features.
    
    Provides tools for:
    - Computing aggregate metrics per cluster
    - Calculating distances to cluster centers
    - Exporting analysis results to Excel
    - Suggesting cluster merges based on metric similarity
    
    Examples
    --------
    >>> analyzer = ClusterAnalyzer(labels, features, cluster_centers)
    >>> df = analyzer.compute_cluster_metrics(spectrograms, freq_bins)
    >>> analyzer.export_to_excel(df, catalog, "results.xlsx")
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        cluster_centers: np.ndarray,
    ):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        labels : np.ndarray
            Cluster labels for each sample
        features : np.ndarray
            Latent features, shape (n_samples, n_features)
        cluster_centers : np.ndarray
            Cluster centers, shape (n_clusters, n_features)
        """
        self.labels = labels
        self.features = features
        self.cluster_centers = cluster_centers
        self.n_clusters = len(np.unique(labels))
    
    def compute_distances_to_centers(self) -> np.ndarray:
        """
        Compute Euclidean distance from each sample to its cluster center.
        
        Returns
        -------
        distances : np.ndarray
            Distance to assigned cluster center for each sample
        """
        distances = np.zeros(len(self.labels))
        
        for i, (label, feature) in enumerate(zip(self.labels, self.features)):
            center = self.cluster_centers[label]
            distances[i] = np.linalg.norm(feature - center)
        
        return distances
    
    def compute_cluster_metrics(
        self,
        spectrograms: np.ndarray,
        freq_bins: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute Phase 3 metrics for all spectrograms.
        
        Parameters
        ----------
        spectrograms : np.ndarray
            All spectrograms, shape (n_samples, freq_bins, time_bins)
        freq_bins : np.ndarray
            Frequency values
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with metrics for each sample
        """
        logger.info("Computing Phase 3 metrics for all samples...")
        
        metrics_list = []
        for i, spec in enumerate(spectrograms):
            metrics = SpectrogramMetrics.compute_all(spec, freq_bins)
            metrics['sample_idx'] = i
            metrics_list.append(metrics)
        
        df = pd.DataFrame(metrics_list)
        logger.info("Phase 3 metrics computed")
        
        return df
    
    def get_cluster_statistics(
        self,
        metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute aggregate statistics for each cluster.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            DataFrame with Phase 3 metrics
            
        Returns
        -------
        stats_df : pd.DataFrame
            Cluster-level statistics
        """
        # Add labels to metrics
        metrics_df['cluster'] = self.labels
        
        # Aggregate by cluster
        metric_cols = ['peak_freq', 'freq_dominance', 
                      'temporal_concentration', 'temporal_mad']
        
        stats = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_data = metrics_df[cluster_mask]
            
            stat = {
                'cluster': cluster_id,
                'n_samples': int(np.sum(cluster_mask)),
            }
            
            for col in metric_cols:
                stat[f'{col}_mean'] = cluster_data[col].mean()
                stat[f'{col}_std'] = cluster_data[col].std()
                stat[f'{col}_median'] = cluster_data[col].median()
            
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def suggest_merges(
        self,
        cluster_stats: pd.DataFrame,
        threshold: float = 0.2
    ) -> List[Tuple[int, int, float]]:
        """
        Suggest cluster pairs to merge based on metric similarity.
        
        Parameters
        ----------
        cluster_stats : pd.DataFrame
            Cluster-level statistics
        threshold : float
            Similarity threshold (0-1, lower = more similar)
            
        Returns
        -------
        suggestions : list of (cluster1, cluster2, similarity)
            Pairs of clusters to consider merging
        """
        from scipy.spatial.distance import euclidean
        
        # Extract mean metrics
        metric_cols = [col for col in cluster_stats.columns 
                      if col.endswith('_mean')]
        
        suggestions = []
        n = len(cluster_stats)
        
        for i in range(n):
            for j in range(i + 1, n):
                vec1 = cluster_stats.iloc[i][metric_cols].values
                vec2 = cluster_stats.iloc[j][metric_cols].values
                
                # Normalize and compute distance
                dist = euclidean(vec1, vec2) / np.sqrt(len(metric_cols))
                
                if dist < threshold:
                    suggestions.append((
                        int(cluster_stats.iloc[i]['cluster']),
                        int(cluster_stats.iloc[j]['cluster']),
                        float(dist)
                    ))
        
        # Sort by similarity
        suggestions.sort(key=lambda x: x[2])
        
        return suggestions
    
    def export_to_excel(
        self,
        metrics_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        output_path: str,
        distances: Optional[np.ndarray] = None,
    ) -> None:
        """
        Export complete analysis to Excel file.
        
        Matches research paper format with multiple sheets:
        - Total_Clusters: Complete event catalog with all metrics
        - Cluster_Centers: GMM cluster centers
        - Cluster_Stats: Aggregate statistics per cluster
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            Phase 3 metrics
        catalog_df : pd.DataFrame
            Event catalog
        output_path : str
            Output Excel file path
        distances : np.ndarray, optional
            Distances to cluster centers
        """
        logger.info(f"Exporting analysis to {output_path}")
        
        # Compute distances if not provided
        if distances is None:
            distances = self.compute_distances_to_centers()
        
        # Merge all data
        combined = catalog_df.copy()
        
        # Add latent features
        for i in range(self.features.shape[1]):
            combined[f'feature_{i+1}'] = self.features[:, i]
        
        # Add Phase 3 metrics
        for col in ['peak_freq', 'freq_dominance', 
                   'temporal_concentration', 'temporal_mad']:
            if col in metrics_df.columns:
                combined[col] = metrics_df[col].values
        
        # Add clustering results
        combined['cluster'] = self.labels
        combined['distance_to_center'] = distances
        
        # Create Excel writer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Complete catalog
            combined.to_excel(writer, sheet_name='Total_Clusters', index=False)
            
            # Sheet 2: Cluster centers
            centers_df = pd.DataFrame(
                self.cluster_centers,
                columns=[f'feature_{i+1}' for i in range(self.cluster_centers.shape[1])]
            )
            centers_df.insert(0, 'cluster', range(len(centers_df)))
            
            # Add cluster sizes
            unique, counts = np.unique(self.labels, return_counts=True)
            size_dict = dict(zip(unique, counts))
            centers_df['n_samples'] = centers_df['cluster'].map(size_dict)
            
            centers_df.to_excel(writer, sheet_name='Cluster_Centers', index=False)
            
            # Sheet 3: Cluster statistics
            stats_df = self.get_cluster_statistics(metrics_df)
            stats_df.to_excel(writer, sheet_name='Cluster_Stats', index=False)
        
        logger.info(f"Analysis exported to {output_path}")
