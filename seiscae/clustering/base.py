"""Abstract base class for clustering algorithms."""

from abc import ABC, abstractmethod
import numpy as np


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.
    
    All clustering algorithms should inherit from this class and implement
    the abstract methods.
    """
    
    @abstractmethod
    def fit(self, features: np.ndarray) -> "BaseClusterer":
        """
        Fit clusterer to features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
            
        Returns
        -------
        self : BaseClusterer
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        pass
