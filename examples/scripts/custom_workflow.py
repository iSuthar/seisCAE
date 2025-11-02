"""
Example: Custom Modular Workflow
==================================

This script demonstrates how to use individual seisCAE components
for a custom analysis workflow.
"""

import numpy as np
import torch
from pathlib import Path

from seiscae.core import EventDetector, SpectrogramGenerator, EventExtractor, EventCatalog
from seiscae.models import ConvAutoencoder
from seiscae.training import AutoencoderTrainer
from seiscae.clustering import GMMClusterer
from seiscae.visualization import (
    plot_cluster_examples,
    plot_cluster_sizes,
    plot_latent_space_umap,
)
from seiscae.utils import setup_logging, get_device

# Setup
setup_logging(level=20)
output_dir = Path('./results_custom')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Step 1: Event Detection
# ============================================================================
print("Step 1: Event Detection")

detector = EventDetector(
    sta_seconds=0.5,
    lta_seconds=30.0,
    threshold_on=25.0,
    threshold_off=3.0,
)

spec_gen = SpectrogramGenerator(
    freq_min=1.0,
    freq_max=50.0,
    time_bins=40,
    freq_bins=256,
)

extractor = EventExtractor(spec_gen)

# Detect events
results = detector.detect_directory(
    directory='/path/to/seismic/data',
    component='EHZ',
    pattern='*.mseed',
)

# Extract events
all_events = []
for filepath, stream, cft, triggers in results:
    trace = stream[0]
    events = extractor.extract_events(trace, triggers, 'EHZ')
    all_events.extend(events)

catalog = EventCatalog(all_events)
print(f"Detected {len(catalog)} events")

# ============================================================================
# Step 2: Custom Model Training
# ============================================================================
print("\nStep 2: Training Custom Model")

# Get spectrograms
spectrograms = np.array([e['spectrogram'] for e in catalog.events])

# Create custom model (larger latent space)
device = get_device(0)  # GPU 0
model = ConvAutoencoder(latent_dim=32, dropout=0.2)

# Train with custom parameters
trainer = AutoencoderTrainer(model, device, learning_rate=5e-5)

history = trainer.train(
    spectrograms=spectrograms,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    patience=30,
    save_dir=str(output_dir / 'models'),
)

print(f"Training completed in {history['epochs_trained']} epochs")

# ============================================================================
# Step 3: Feature Extraction
# ============================================================================
print("\nStep 3: Feature Extraction")

features = trainer.extract_features(spectrograms)
print(f"Extracted features: {features.shape}")

# Save features
np.save(output_dir / 'features.npy', features)

# ============================================================================
# Step 4: Custom Clustering
# ============================================================================
print("\nStep 4: Clustering")

# Try different numbers of clusters
for n_clusters in [5, 10, 15]:
    print(f"\n  Testing with {n_clusters} clusters...")
    
    clusterer = GMMClusterer(n_clusters=n_clusters)
    labels = clusterer.fit_predict(features)
    
    # Save results
    cluster_dir = output_dir / f'clusters_{n_clusters}'
    cluster_dir.mkdir(exist_ok=True)
    
    np.save(cluster_dir / 'labels.npy', labels)
    
    # Visualize
    plot_cluster_sizes(labels, cluster_dir / 'sizes.png')
    plot_latent_space_umap(features, labels, cluster_dir / 'umap.png')

# ============================================================================
# Step 5: Analysis
# ============================================================================
print("\nStep 5: Custom Analysis")

# Use auto-selected clustering for final analysis
clusterer_auto = GMMClusterer(n_clusters=None, max_clusters=20)
labels_final = clusterer_auto.fit_predict(features)

print(f"\nAuto-selected {clusterer_auto.n_clusters} clusters")

# Compute cluster statistics
for cluster_id in range(clusterer_auto.n_clusters):
    mask = labels_final == cluster_id
    cluster_features = features[mask]
    cluster_events = [e for i, e in enumerate(catalog.events) if mask[i]]
    
    # Compute average energy
    energies = [e['energy'] for e in cluster_events]
    avg_energy = np.mean(energies)
    
    print(f"  Cluster {cluster_id}: {len(cluster_events)} events, avg energy: {avg_energy:.4f}")

print("\nCustom workflow complete!")
