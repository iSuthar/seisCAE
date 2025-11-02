"""Event catalog management."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EventCatalog:
    """
    Manage seismic event catalog.
    
    Parameters
    ----------
    events : list of dict, optional
        Initial events
    
    Examples
    --------
    >>> catalog = EventCatalog()
    >>> catalog.add_event({'station': 'ABC', 'duration': 2.5})
    >>> df = catalog.to_dataframe()
    >>> catalog.save('catalog.csv')
    """
    
    def __init__(self, events: Optional[List[Dict[str, Any]]] = None):
        self.events = events or []
        self._df = None
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add a single event to catalog."""
        self.events.append(event)
        self._df = None  # Invalidate cached DataFrame
    
    def add_events(self, events: List[Dict[str, Any]]) -> None:
        """Add multiple events to catalog."""
        self.events.extend(events)
        self._df = None
    
    def to_dataframe(self, include_arrays: bool = False) -> pd.DataFrame:
        """
        Convert catalog to pandas DataFrame.
        
        Parameters
        ----------
        include_arrays : bool
            Include waveform and spectrogram arrays
            
        Returns
        -------
        df : pd.DataFrame
            Catalog as DataFrame
        """
        if self._df is None or include_arrays:
            if not self.events:
                return pd.DataFrame()
            
            # Extract scalar fields
            scalar_fields = [
                'station', 'channel', 'component', 'starttime', 'endtime',
                'duration', 'center_time', 'energy', 'sampling_rate'
            ]
            
            data = {field: [e.get(field) for e in self.events] for field in scalar_fields}
            
            if include_arrays:
                data['waveform'] = [e.get('waveform') for e in self.events]
                data['spectrogram'] = [e.get('spectrogram') for e in self.events]
            
            df = pd.DataFrame(data)
            
            if not include_arrays:
                self._df = df
            
            return df
        
        return self._df
    
    def save(self, filepath: str, include_arrays: bool = True) -> None:
        """
        Save catalog to file.
        
        Parameters
        ----------
        filepath : str
            Output filepath (.csv, .pkl, or .h5)
        include_arrays : bool
            Include waveform/spectrogram arrays (only for .pkl/.h5)
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.csv':
            df = self.to_dataframe(include_arrays=False)
            df.to_csv(filepath, index=False)
        elif path.suffix == '.pkl':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.events, f)
        elif path.suffix == '.h5':
            df = self.to_dataframe(include_arrays=include_arrays)
            df.to_hdf(filepath, key='catalog', mode='w')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Saved catalog to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "EventCatalog":
        """
        Load catalog from file.
        
        Parameters
        ----------
        filepath : str
            Input filepath
            
        Returns
        -------
        catalog : EventCatalog
            Loaded catalog
        """
        path = Path(filepath)
        
        if path.suffix == '.pkl':
            import pickle
            with open(filepath, 'rb') as f:
                events = pickle.load(f)
            return cls(events)
        elif path.suffix == '.h5':
            df = pd.read_hdf(filepath, key='catalog')
            events = df.to_dict('records')
            return cls(events)
        elif path.suffix == '.csv':
            df = pd.read_csv(filepath)
            events = df.to_dict('records')
            return cls(events)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def filter(self, **criteria) -> "EventCatalog":
        """
        Filter events based on criteria.
        
        Parameters
        ----------
        **criteria : keyword arguments
            Filtering criteria (e.g., station='ABC', duration_min=1.0)
            
        Returns
        -------
        catalog : EventCatalog
            Filtered catalog
        
        Examples
        --------
        >>> catalog_filtered = catalog.filter(station='ABC', duration_min=2.0)
        """
        filtered_events = self.events.copy()
        
        for key, value in criteria.items():
            if key.endswith('_min'):
                field = key[:-4]
                filtered_events = [e for e in filtered_events if e.get(field, 0) >= value]
            elif key.endswith('_max'):
                field = key[:-4]
                filtered_events = [e for e in filtered_events if e.get(field, float('inf')) <= value]
            else:
                filtered_events = [e for e in filtered_events if e.get(key) == value]
        
        return EventCatalog(filtered_events)
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __repr__(self) -> str:
        return f"EventCatalog(n_events={len(self.events)})"
