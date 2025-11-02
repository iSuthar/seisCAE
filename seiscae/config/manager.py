"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
from .defaults import DEFAULT_CONFIG


class ConfigManager:
    """
    Manage configuration loading, merging, and validation.
    
    Parameters
    ----------
    config : dict, optional
        Initial configuration. If None, uses defaults.
    
    Examples
    --------
    >>> config = ConfigManager()
    >>> config.set('training.epochs', 500)
    >>> epochs = config.get('training.epochs')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config : dict, optional
            Initial configuration. If None, uses defaults.
        """
        self.config = deepcopy(DEFAULT_CONFIG)
        if config:
            self._merge_config(config)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ConfigManager":
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        path : str
            Path to YAML configuration file.
            
        Returns
        -------
        ConfigManager
            Initialized configuration manager.
        """
        with open(path, 'r') as f:
            user_config = yaml.safe_load(f)
        return cls(user_config)
    
    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Recursively merge user config into defaults."""
        def merge(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    merge(base[key], value)
                else:
                    base[key] = value
        merge(self.config, user_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., 'detection.sta_seconds')
        default : any
            Default value if key not found
            
        Returns
        -------
        any
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key
        value : any
            Value to set
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def save(self, path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Parameters
        ----------
        path : str
            Output file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # STA/LTA validation
        if self.get('detection.sta_seconds') >= self.get('detection.lta_seconds'):
            raise ValueError("STA window must be smaller than LTA window")
        
        if self.get('detection.threshold_on') <= self.get('detection.threshold_off'):
            raise ValueError("Threshold ON must be greater than threshold OFF")
        
        # Training validation
        if self.get('training.epochs') <= 0:
            raise ValueError("Epochs must be positive")
        
        if self.get('training.batch_size') <= 0:
            raise ValueError("Batch size must be positive")
        
        if not (0 < self.get('training.validation_split') < 1):
            raise ValueError("Validation split must be between 0 and 1")
        
        # Model validation
        if self.get('model.latent_dim') <= 0:
            raise ValueError("Latent dimension must be positive")
    
    def __repr__(self) -> str:
        return f"ConfigManager({self.config})"


def load_config(path: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Parameters
    ----------
    path : str, optional
        Path to YAML config file. If None, uses defaults.
        
    Returns
    -------
    ConfigManager
        Configuration manager instance.
    
    Examples
    --------
    >>> config = load_config("configs/default.yaml")
    >>> config = load_config()  # Use defaults
    """
    if path:
        return ConfigManager.from_yaml(path)
    return ConfigManager()
