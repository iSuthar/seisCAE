"""
Generate custom configuration templates for seisCAE.

Usage:
    python template_generator.py --type volcanic --output my_config.yaml
"""

import yaml
import argparse
from pathlib import Path


TEMPLATES = {
    'minimal': {
        'detection': {'sta_seconds': 0.5, 'lta_seconds': 30.0},
        'training': {'epochs': 100, 'batch_size': 128},
        'clustering': {'algorithm': 'gmm', 'n_clusters': None},
    },
    
    'fast': {
        'detection': {'sta_seconds': 0.5, 'lta_seconds': 30.0, 'threshold_on': 25.0},
        'training': {'epochs': 50, 'batch_size': 256, 'num_workers': 8},
        'clustering': {'algorithm': 'gmm', 'n_clusters': 5},
        'hardware': {'gpu': 0, 'mixed_precision': True},
    },
    
    'accurate': {
        'detection': {'sta_seconds': 0.3, 'lta_seconds': 40.0, 'threshold_on': 15.0},
        'training': {'epochs': 1000, 'batch_size': 32, 'patience': 50},
        'clustering': {'algorithm': 'gmm', 'n_clusters': None, 'max_clusters': 30},
        'hardware': {'gpu': 0},
    },
}


def generate_config(template_type, output_path):
    """Generate configuration file from template."""
    if template_type not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_type}")
    
    config = TEMPLATES[template_type]
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated configuration: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate seisCAE configuration')
    parser.add_argument('--type', choices=list(TEMPLATES.keys()), required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    generate_config(args.type, args.output)
