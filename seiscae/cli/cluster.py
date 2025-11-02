"""Clustering CLI commands."""

import click
from pathlib import Path
import numpy as np


@click.command('cluster')
@click.option('--features', required=True, type=click.Path(exists=True),
              help='Features file (.npy)')
@click.option('--output', required=True, type=click.Path(), help='Output directory')
@click.option('--n-clusters', type=int, default=None, 
              help='Number of clusters (None for auto-select)')
@click.option('--algorithm', type=str, default='gmm', help='Clustering algorithm')
@click.option('--max-clusters', type=int, default=20, 
              help='Max clusters for auto-selection')
def cluster_cmd(features, output, n_clusters, algorithm, max_clusters):
    """Cluster events based on latent features."""
    from ..clustering import get_clusterer
    from ..visualization import plot_cluster_sizes, plot_latent_space_umap
    
    click.echo("Starting clustering...")
    
    # Load features
    feats = np.load(features)
    click.echo(f"Loaded features: {feats.shape}")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cluster
    clusterer = get_clusterer(
        algorithm,
        n_clusters=n_clusters,
        max_clusters=max_clusters,
    )
    
    labels = clusterer.fit_predict(feats)
    
    # Save results
    np.save(output_path / "labels.npy", labels)
    
    if hasattr(clusterer, 'get_cluster_centers'):
        centers = clusterer.get_cluster_centers()
        np.save(output_path / "cluster_centers.npy", centers)
    
    # Generate visualizations
    click.echo("Generating visualizations...")
    plot_cluster_sizes(labels, save_path=output_path / "cluster_sizes.png")
    plot_latent_space_umap(feats, labels, save_path=output_path / "latent_umap.png")
    
    click.echo(f"\nClustering complete!")
    click.echo(f"Found {len(np.unique(labels))} clusters")
    
    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    click.echo("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        click.echo(f"  Cluster {cluster_id}: {count} samples")
    
    click.echo(f"\nResults saved to: {output_path}")
