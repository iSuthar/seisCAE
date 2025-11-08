<div align="center">

# seisCAE 🌍

**A modular Python package for clustering seismic events using Convolutional Autoencoders.**

[![PyPI version](https://badge.fury.io/py/seisCAE.svg)](https://badge.fury.io/py/seisCAE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/iSuthar/seisCAE/actions/workflows/test.yml/badge.svg)](https://github.com/iSuthar/seisCAE/actions/workflows/test.yml)

</div>

`seisCAE` provides an end-to-end toolkit for unsupervised seismic event analysis. It automates the process of detecting events from continuous seismic data, extracting meaningful features using a convolutional autoencoder, and grouping similar events with a Gaussian Mixture Model.

---

## 🎯 Core Features

-   **Automated Event Detection**: Employs the classic STA/LTA trigger algorithm to find potential seismic events in raw data streams.
-   **Deep Learning Feature Extraction**: Trains a convolutional autoencoder on event spectrograms to learn a compressed, low-dimensional representation of the data.
-   **Unsupervised Clustering**: Uses a Gaussian Mixture Model (GMM) with automatic cluster number selection (via BIC) to group events.
-   **Rich Visualization**: Generates a suite of diagnostic plots for evaluating detection, training, and clustering results.
-   **Modular & Extensible**: Each component (detector, model, clusterer) can be used independently or as part of the main pipeline.
-   **Dual Interface**: Accessible through both a powerful Command-Line Interface (CLI) and a flexible Python API.

## 📦 Installation

You can install `seisCAE` directly from PyPI or from the source for development.

### From PyPI

```bash
pip install seiscae
```

### From Source

```bash
# Clone the repository
git clone https://github.com/iSuthar/seisCAE.git
cd seisCAE

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

Get started immediately using the CLI for a full pipeline run or use the Python API for more granular control.

### Command-Line Interface (CLI)

The CLI is the quickest way to process your data. Run the full pipeline with a single command, pointing to your data directory and a configuration file.

```bash
seiscae run --config configs/default.yaml --data /path/to/your/data --output ./results
```

You can also run individual stages:

```bash
# 1. Detect events from raw data
seiscae detect --data /path/to/your/data --output ./results/detections

# 2. Train the autoencoder on the resulting spectrograms
seiscae train --spectrograms ./results/detections/spectrograms.npy --output ./models

# 3. Cluster events using the trained model's features
seiscae cluster --features ./models/features.npy --output ./clusters
```

### Python API

The Python API offers maximum flexibility for integration into custom workflows.

#### Full Pipeline Example

```python
from seiscae.pipeline import Pipeline
from seiscae.utils.config import load_config

# Load configuration from a YAML file
config = load_config("configs/default.yaml")

# Initialize and run the full pipeline
pipeline = Pipeline(config)
results = pipeline.run(data_path="/path/to/your/data")

# Inspect the results
print(f"Detected {results.n_events} events.")
print(f"Found {results.n_clusters} distinct clusters.")

# Access events from a specific cluster
cluster_0_events = results.get_cluster(0)
print(f"Cluster 0 contains {len(cluster_0_events)} events.")
```

## 🔧 Configuration

Pipeline behavior is controlled by a single `config.yaml` file. Customize parameters for detection, spectrograms, model architecture, training, and clustering.

```yaml
# configs/default.yaml

# Event detection parameters
detection:
  sta_seconds: 0.5
  lta_seconds: 30.0
  threshold_on: 25.0
  threshold_off: 3.0

# Spectrogram generation settings
spectrogram:
  nperseg: 128
  freq_min: 1.0
  freq_max: 50.0

# Model architecture
model:
  latent_dim: 16
  dropout: 0.1

# Training loop parameters
training:
  epochs: 300
  batch_size: 128
  learning_rate: 0.0001
  patience: 20  # For early stopping

# Clustering algorithm settings
clustering:
  algorithm: gmm
  n_clusters: null  # Set to null for automatic selection via BIC

# Input/Output paths
io:
  output_base: "./results"
```

## 🧪 Testing

We use `pytest` for testing. To run the test suite:

```bash
# Run all tests
pytest

# Run tests with code coverage report
pytest --cov=seiscae --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started with setting up the development environment and our pull request process.

## 📝 Citation

If you use `seisCAE` in your research, please cite it as follows:

```bibtex
@software{seiscae2025,
  author = {Suthar, Ankit},
  title = {seisCAE: A Python Package for Seismic Event Clustering with Convolutional Autoencoders},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/iSuthar/seisCAE}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   Built with [PyTorch](https://pytorch.org/) for deep learning.
-   Seismic data processing powered by [ObsPy](https://obspy.org/).
