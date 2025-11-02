"""Tests for clustering module."""

import pytest
import numpy as np

from seiscae.clustering import GMMClusterer, get_clusterer


class TestGMMClusterer:
    """Test suite for GMMClusterer."""
    
    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = GMMClusterer(n_clusters=5)
        
        assert clusterer.n_clusters == 5
    
    def test_fit_predict(self):
        """Test fit_predict on synthetic data."""
        # Create synthetic clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(50, 10) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        cluster3 = np.random.randn(50, 10) + np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5])
        
        features = np.vstack([cluster1, cluster2, cluster3])
        
        # Fit and predict
        clusterer = GMMClusterer(n_clusters=3)
        labels = clusterer.fit_predict(features)
        
        # Should have 3 unique labels
        assert len(np.unique(labels)) == 3
        assert len(labels) == 150
    
    def test_auto_selection(self):
        """Test automatic cluster selection."""
        np.random.seed(42)
        
        # Create 4 clear clusters
        data = []
        for i in range(4):
            cluster = np.random.randn(30, 10) + np.ones(10) * i * 5
            data.append(cluster)
        
        features = np.vstack(data)
        
        # Auto-select clusters
        clusterer = GMMClusterer(n_clusters=None, max_clusters=10)
        labels = clusterer.fit_predict(features)
        
        # Should detect around 4 clusters (may vary due to BIC selection)
        n_clusters = len(np.unique(labels))
        assert 2 <= n_clusters <= 6  # Reasonable range
    
    def test_get_cluster_centers(self):
        """Test getting cluster centers."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        
        clusterer = GMMClusterer(n_clusters=3)
        clusterer.fit(features)
        
        centers = clusterer.get_cluster_centers()
        
        assert centers.shape == (3, 10)
    
    def test_clusterer_factory(self):
        """Test clusterer factory function."""
        clusterer = get_clusterer('gmm', n_clusters=5)
        
        assert isinstance(clusterer, GMMClusterer)
        assert clusterer.n_clusters == 5
