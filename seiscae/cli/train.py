"""Training CLI commands."""

import click
from pathlib import Path
import numpy as np
import torch


@click.command('train')
@click.option('--spectrograms', required=True, type=click.Path(exists=True), 
              help='Spectrograms file (.npy)')
@click.option('--output', required=True, type=click.Path(), help='Output directory')
@click.option('--latent-dim', type=int, default=16, help='Latent dimension')
@click.option('--epochs', type=int, default=300, help='Number of epochs')
@click.option('--batch-size', type=int, default=128, help='Batch size')
@click.option('--learning-rate', type=float, default=1e-4, help='Learning rate')
@click.option('--patience', type=int, default=20, help='Early stopping patience')
@click.option('--gpu', type=int, default=-1, help='GPU ID (-1 for CPU)')
def train_cmd(spectrograms, output, latent_dim, epochs, batch_size, learning_rate, patience, gpu):
    """Train autoencoder model on spectrograms."""
    from ..models import ConvAutoencoder
    from ..training import AutoencoderTrainer
    from ..utils.device import get_device
    
    click.echo("Starting model training...")
    
    # Load spectrograms
    specs = np.load(spectrograms)
    click.echo(f"Loaded {len(specs)} spectrograms")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and trainer
    device = get_device(gpu)
    model = ConvAutoencoder(latent_dim=latent_dim)
    trainer = AutoencoderTrainer(model, device, learning_rate)
    
    # Train
    click.echo(f"Training on device: {device}")
    history = trainer.train(
        spectrograms=specs,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        save_dir=str(output_path),
    )
    
    # Save final model
    torch.save(model.state_dict(), output_path / "final_model.pt")
    
    # Extract and save features
    click.echo("Extracting features...")
    features = trainer.extract_features(specs)
    np.save(output_path / "features.npy", features)
    
    click.echo(f"\nTraining complete!")
    click.echo(f"Trained for {history['epochs_trained']} epochs")
    click.echo(f"Final train loss: {history['train_losses'][-1]:.6f}")
    click.echo(f"Final val loss: {history['val_losses'][-1]:.6f}")
    click.echo(f"Results saved to: {output_path}")
