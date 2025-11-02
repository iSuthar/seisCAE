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
    Gaussian Mixture Model clustering with automatic cluster selection.
    
    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters. If None, automatically determined using BIC.
    covariance_type : str
        Covariance type ('full', 'tied', 'diag', 'spherical')
    max_clusters : int
        Maximum number of clusters to try for auto-selection
    random_state : int
        Random seed
    
    Examples
    --------
    >>> clusterer = GMMClusterer(n_clusters=None)  # Auto-select
    >>> labels = clusterer.fit_predict(features)
    >>> print(f"Found {clusterer.n_clusters} clusters")
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        covariance_type: str = 'full',
        max_clusters: int = 20,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.max_clusters = max_clusters
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.selection_metrics_ = None
    
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
        
        # Auto-select number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters, self.selection_metrics_ = self._select_n_clusters(features_scaled)
            logger.info(f"Auto-selected {self.n_clusters} clusters")
        
        # Fit final model
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
    
    def _select_n_clusters(self, features_scaled: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """
        Automatically select number of clusters using BIC and silhouette.
        
        Returns
        -------
        n_clusters : int
            Selected number of clusters
        metrics : dict
            Selection metrics for all tested cluster numbers
        """
        max_k = min(self.max_clusters, max(2, int(2 * np.sqrt(len(features_scaled)))))
        k_range = range(1, max_k + 1)
        
        bics = []
        aics = []
        silhouettes = []
        
        logger.info(f"Testing k from 1 to {max_k}...")
        
        for k in k_range:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.covariance_type,
                    n_init=5,
                    random_state=self.random_state,
                )
                gmm.fit(features_scaled)
                
                bics.append(gmm.bic(features_scaled))
                aics.append(gmm.aic(features_scaled))
                
                if k >= 2:
                    labels = gmm.predict(features_scaled)
                    try:
                        sil = silhouette_score(features_scaled, labels)
                        silhouettes.append(sil)
                    except:
                        silhouettes.append(np.nan)
                else:
                    silhouettes.append(np.nan)
                    
            except Exception as e:
                logger.warning(f"GMM failed for k={k}: {e}")
                bics.append(np.nan)
                aics.append(np.nan)
                silhouettes.append(np.nan)
        
        # Select best k by BIC (lower is better)
        best_k = k_range[int(np.nanargmin(bics))]
        
        metrics = {
            'k_range': list(k_range),
            'bic': bics,
            'aic': aics,
            'silhouette': silhouettes,
            'best_k_bic': best_k,
            'best_k_silhouette': k_range[int(np.nanargmax(silhouettes))] if len(silhouettes) > 0 else None,
        }
        
        return best_k, metrics
    
    def _log_cluster_sizes(self) -> None:
        """Log cluster sizes."""
        unique, counts = np.unique(self.labels_, return_counts=True)
        for cluster, count in zip(unique, counts):
            pct = 100 * count / len(self.labels_)
            logger.info(f"  Cluster {cluster}: {count} samples ({pct:.1f}%)")
