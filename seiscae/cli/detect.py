"""Detection CLI commands."""

import click
from pathlib import Path


@click.command('detect')
@click.option('--data', required=True, type=click.Path(exists=True), help='Input data directory')
@click.option('--output', required=True, type=click.Path(), help='Output directory')
@click.option('--component', multiple=True, default=['EHZ'], help='Seismic component(s)')
@click.option('--sta', type=float, default=0.5, help='STA window (seconds)')
@click.option('--lta', type=float, default=30.0, help='LTA window (seconds)')
@click.option('--threshold-on', type=float, default=25.0, help='Trigger ON threshold')
@click.option('--threshold-off', type=float, default=3.0, help='Trigger OFF threshold')
@click.option('--save-diagnostics', is_flag=True, help='Save diagnostic plots')
def detect_cmd(data, output, component, sta, lta, threshold_on, threshold_off, save_diagnostics):
    """Detect seismic events using STA/LTA trigger."""
    from ..core import EventDetector, SpectrogramGenerator, EventExtractor, EventCatalog
    from ..config import ConfigManager
    from ..visualization import Visualizer
    from tqdm import tqdm
    import numpy as np
    
    click.echo("Starting event detection...")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    detector = EventDetector(
        sta_seconds=sta,
        lta_seconds=lta,
        threshold_on=threshold_on,
        threshold_off=threshold_off,
    )
    
    spec_gen = SpectrogramGenerator()
    extractor = EventExtractor(spec_gen)
    
    # Detect events for each component
    all_events = []
    for comp in component:
        click.echo(f"Processing component: {comp}")
        
        results = detector.detect_directory(
            directory=data,
            component=comp,
            pattern="*",
        )
        
        for filepath, stream, cft, triggers in tqdm(results, desc=f"{comp} events"):
            trace = stream[0]
            events = extractor.extract_events(trace, triggers, comp)
            all_events.extend(events)
            
            # Save diagnostics
            if save_diagnostics:
                config = ConfigManager()
                viz = Visualizer(config)
                diag_path = output_path / "diagnostics" / comp
                diag_path.mkdir(parents=True, exist_ok=True)
                viz.plot_detection_summary(
                    trace, cft, triggers,
                    save_path=diag_path / f"{Path(filepath).stem}.png"
                )
    
    # Save catalog
    catalog = EventCatalog(all_events)
    catalog.save(output_path / "catalog.pkl")
    catalog.save(output_path / "catalog.csv")
    
    # Save spectrograms
    spectrograms = np.array([e['spectrogram'] for e in catalog.events])
    np.save(output_path / "spectrograms.npy", spectrograms)
    
    click.echo(f"\nDetection complete!")
    click.echo(f"Total events: {len(catalog)}")
    click.echo(f"Results saved to: {output_path}")
