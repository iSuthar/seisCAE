"""Gaussian Mixture Model clustering."""

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Tuple
import logging

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class GMMClusterer(BaseClusterer):
    """
    Gaussian Mixture Model clustering with specified number of clusters.
    
    Updated to match research paper approach: requires explicit n_clusters
    specification (no automatic selection). This ensures reproducibility
    and follows the research methodology of using K=50 for fine-grain
    clustering followed by manual merging based on physical metrics.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters (REQUIRED - no default, no auto-selection)
    covariance_type : str
        Covariance type ('full', 'tied', 'diag', 'spherical')
        Research paper uses 'full' for maximum flexibility
    random_state : int
        Random seed for reproducibility
    
    Examples
    --------
    >>> clusterer = GMMClusterer(n_clusters=50)  # Research paper uses 50
    >>> labels = clusterer.fit_predict(features)
    >>> print(f"Clustered into {clusterer.n_clusters} groups")
    
    Notes
    -----
    The research paper uses n_clusters=50 as an intentional over-division
    strategy to capture fine-grain differences, followed by manual merging
    based on physical metrics (peak frequency, frequency dominance, etc.).
    """
    
    def __init__(
        self,
        n_clusters: int,
        covariance_type: str = 'full',
        random_state: int = 42,
    ):
        if n_clusters is None:
            raise ValueError(
                "n_clusters is required and cannot be None. "
                "Auto-selection has been removed to match research paper methodology. "
                "Specify n_clusters explicitly (research paper uses n_clusters=50)."
            )
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError(
                f"n_clusters must be a positive integer, got {n_clusters}"
            )
        
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
    
    def fit(self, features: np.ndarray) -> "GMMClusterer":
        """
        Fit GMM to features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
            
        Returns
        -------
        self : GMMClusterer
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit GMM with specified number of clusters
        logger.info(f"Fitting GMM with {self.n_clusters} clusters")
        
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            n_init=10,
            random_state=self.random_state,
        )
        self.model.fit(features_scaled)
        self.labels_ = self.model.predict(features_scaled)
        
        logger.info(f"GMM fitted with {self.n_clusters} components")
        self._log_cluster_sizes()
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(features)
        return self.labels_
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster probabilities.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
            
        Returns
        -------
        proba : np.ndarray
            Cluster probabilities, shape (n_samples, n_clusters)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def get_cluster_centers(self, original_scale: bool = True) -> np.ndarray:
        """
        Get cluster centers.
        
        Parameters
        ----------
        original_scale : bool
            Return centers in original feature scale
            
        Returns
        -------
        centers : np.ndarray
            Cluster centers
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        centers = self.model.means_
        
        if original_scale:
            centers = self.scaler.inverse_transform(centers)
        
        return centers
    
    def _log_cluster_sizes(self) -> None:
        """Log cluster sizes."""
        unique, counts = np.unique(self.labels_, return_counts=True)
        for cluster, count in zip(unique, counts):
            pct = 100 * count / len(self.labels_)
            logger.info(f"  Cluster {cluster}: {count} samples ({pct:.1f}%)")
