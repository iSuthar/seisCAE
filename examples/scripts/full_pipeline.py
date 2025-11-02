"""
Example: Full seisCAE Pipeline
================================

This script demonstrates how to run the complete seisCAE pipeline
for seismic event detection, feature extraction, and clustering.
"""

from seiscae import Pipeline, load_config
from seiscae.utils import setup_logging

# Setup logging
setup_logging('seiscae_pipeline.log', level=20)  # INFO level

# Load configuration
config = load_config('configs/default.yaml')

# You can modify config programmatically
config.set('training.epochs', 500)
config.set('clustering.n_clusters', 10)
config.set('hardware.gpu', 0)  # Use GPU 0

# Create pipeline
pipeline = Pipeline(config)

# Run complete pipeline
results = pipeline.run(
    data_path='/path/to/seismic/data',
    output_dir='./results_full_pipeline',
    skip_detection=False,
    skip_training=False,
    skip_clustering=False,
)

# Access results
print(f"\nPipeline Results:")
print(f"  Total events: {results.n_events}")
print(f"  Clusters found: {results.n_clusters}")
print(f"  Output directory: {results.output_dir}")

# Get events from specific cluster
cluster_0 = results.get_cluster(0)
print(f"\nCluster 0 has {len(cluster_0)} events")

# Save cluster-specific catalog
cluster_0.save(results.output_dir / 'cluster_0.csv')

print("\nPipeline complete!")
