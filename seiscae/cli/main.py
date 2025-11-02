"""Main CLI entry point for seisCAE."""

import click
import logging
from pathlib import Path

from ..utils.logging import setup_logging
from .detect import detect_cmd
from .train import train_cmd
from .cluster import cluster_cmd


@click.group()
@click.version_option(version='0.1.0', prog_name='seisCAE')
@click.option('--log-file', type=str, help='Log file path')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def cli(log_file, verbose):
    """
    seisCAE: Clustering Seismic Events with Convolutional Autoencoders
    
    A modular pipeline for seismic event detection, feature extraction,
    and clustering using deep learning.
    
    Examples:
    
        # Run full pipeline
        seiscae run --config config.yaml --data ./data --output ./results
        
        # Individual stages
        seiscae detect --data ./data --output ./results
        seiscae train --spectrograms ./results --output ./models
        seiscae cluster --features ./features.npy --output ./clusters
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_file=log_file, level=level)


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Configuration file (YAML)')
@click.option('--data', required=True, type=click.Path(exists=True), help='Input data directory')
@click.option('--output', required=True, type=click.Path(), help='Output directory')
@click.option('--skip-detection', is_flag=True, help='Skip detection stage')
@click.option('--skip-training', is_flag=True, help='Skip training stage')
@click.option('--skip-clustering', is_flag=True, help='Skip clustering stage')
def run(config, data, output, skip_detection, skip_training, skip_clustering):
    """Run the complete seisCAE pipeline."""
    from ..pipeline import Pipeline
    from ..config import load_config
    
    # Load configuration
    if config:
        cfg = load_config(config)
    else:
        cfg = load_config()
        click.echo("Using default configuration")
    
    # Override output directory
    cfg.set('io.output_base', output)
    
    # Create and run pipeline
    click.echo("=" * 80)
    click.echo("seisCAE Pipeline")
    click.echo("=" * 80)
    
    pipeline = Pipeline(cfg)
    results = pipeline.run(
        data_path=data,
        output_dir=output,
        skip_detection=skip_detection,
        skip_training=skip_training,
        skip_clustering=skip_clustering,
    )
    
    click.echo("\n" + "=" * 80)
    click.echo("Pipeline Complete!")
    click.echo(f"Detected {results.n_events} events")
    click.echo(f"Found {results.n_clusters} clusters")
    click.echo(f"Results saved to: {results.output_dir}")
    click.echo("=" * 80)


# Register subcommands
cli.add_command(detect_cmd)
cli.add_command(train_cmd)
cli.add_command(cluster_cmd)


if __name__ == '__main__':
    cli()
