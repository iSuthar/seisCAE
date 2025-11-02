# seisCAE 🌍

**Seismic Convolutional AutoEncoder** - A modular Python package for clustering seismic events using deep learning.

[![PyPI version](https://badge.fury.io/py/seisCAE.svg)](https://badge.fury.io/py/seisCAE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Features

- **Event Detection**: STA/LTA trigger algorithm for seismic event detection
- **Deep Learning**: Convolutional autoencoder for feature extraction from spectrograms
- **Clustering**: GMM-based clustering with automatic cluster selection using BIC
- **Visualization**: Comprehensive diagnostic plots and cluster analysis
- **Modular Design**: Use individual components or full pipeline
- **Easy to Extend**: Add custom models and clustering algorithms
- **CLI & API**: Both command-line and Python API interfaces

## 📦 Installation

### From PyPI (when published)

```bash
pip install seisCAE
```

### From source (for development)

```bash
# Clone the repository
git clone https://github.com/iSuthar/seisCAE.git
cd seisCAE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Full pipeline
seiscae run --config configs/default.yaml --data /path/to/data --output ./results

# Individual stages
seiscae detect --data /path/to/data --output ./results
seiscae train --spectrograms ./results/spectrograms --output ./models
seiscae cluster --features ./models/features.npy --output ./clusters
```

### Python API - Full Pipeline

```python
from seiscae import Pipeline, load_config

# Load configuration
config = load_config("configs/default.yaml")

# Run full pipeline
pipeline = Pipeline(config)
results = pipeline.run(data_path="/path/to/seismic/data")

# Access results
print(f"Detected {results.n_events} events")
print(f"Found {results.n_clusters} clusters")

# Get events from a specific cluster
cluster_0 = results.get_cluster(0)
print(f"Cluster 0 has {len(cluster_0)} events")
```

### Python API - Modular Usage

```python
from seiscae.core import EventDetector
from seiscae.models import ConvAutoencoder
from seiscae.clustering import GMMClusterer
from seiscae.training import AutoencoderTrainer
import torch

# Step 1: Detect events
detector = EventDetector(sta_seconds=0.5, lta_seconds=30.0)
results = detector.detect_directory("/path/to/data")

# Step 2: Train autoencoder
model = ConvAutoencoder(latent_dim=16)
trainer = AutoencoderTrainer(model, device=torch.device('cuda:0'))
history = trainer.train(spectrograms, epochs=300)

# Step 3: Extract features
features = trainer.extract_features(spectrograms)

# Step 4: Cluster
clusterer = GMMClusterer(n_clusters=None)  # Auto-select
labels = clusterer.fit_predict(features)
```

## 📊 Workflow

```
Raw Seismic Data
      ↓
[Event Detection] → STA/LTA Trigger
      ↓
[Spectrogram Generation] → STFT
      ↓
[Autoencoder Training] → Feature Extraction
      ↓
[GMM Clustering] → Event Grouping
      ↓
Results & Visualizations
```

## 🔧 Configuration

Create a YAML configuration file:

```yaml
# config.yaml
detection:
  sta_seconds: 0.5
  lta_seconds: 30.0
  threshold_on: 25.0
  threshold_off: 3.0

spectrogram:
  nperseg: 128
  freq_min: 1.0
  freq_max: 50.0

model:
  latent_dim: 16
  dropout: 0.1

training:
  epochs: 300
  batch_size: 128
  learning_rate: 0.0001
  patience: 20

clustering:
  algorithm: gmm
  n_clusters: null  # Auto-select

io:
  output_base: "./results"
```

## 📚 Documentation

- **Quick Start Guide**: See `examples/notebooks/01_quickstart.ipynb`
- **API Reference**: See docstrings in code
- **Custom Models**: See `examples/notebooks/02_custom_models.ipynb`
- **Advanced Clustering**: See `examples/notebooks/03_advanced_clustering.ipynb`

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=seiscae --cov-report=html

# Run specific test
pytest tests/test_detection.py
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black seiscae/
isort seiscae/

# Run linter
flake8 seiscae/

# Type checking
mypy seiscae/
```

## 📝 Citation

If you use seisCAE in your research, please cite:

```bibtex
@software{seiscae2025,
  author = {Suthar, Ankit},
  title = {seisCAE: Clustering Seismic Events with Convolutional Autoencoders},
  year = {2025},
  url = {https://github.com/iSuthar/seisCAE}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Seismic data handling with [ObsPy](https://obspy.org/)
- Inspired by [PyImageSearch AutoEncoder Tutorial](https://pyimagesearch.com/2023/07/17/implementing-a-convolutional-autoencoder-with-pytorch/)

## 📧 Contact

- **Author**: Ankit Suthar
- **GitHub**: [@iSuthar](https://github.com/iSuthar)
- **Issues**: [GitHub Issues](https://github.com/iSuthar/seisCAE/issues)

## 🗺️ Roadmap

- [ ] Add support for more autoencoder architectures (VAE, ResNet-AE)
- [ ] Implement additional clustering algorithms (DBSCAN, HDBSCAN)
- [ ] GPU acceleration for spectrogram generation
- [ ] Real-time event detection
- [ ] Web-based visualization dashboard
- [ ] Pre-trained models for common seismic scenarios

---

**Made with ❤️ for the seismology and machine learning community**
