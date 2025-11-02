"""Results writing utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_results(
    output_dir: str,
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    catalog: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save clustering results.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    features : np.ndarray
        Latent features
    labels : np.ndarray, optional
        Cluster labels
    catalog : pd.DataFrame, optional
        Event catalog
    metadata : dict, optional
        Additional metadata
    
    Examples
    --------
    >>> save_results('./results', features, labels, catalog)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save features
    np.save(output_path / 'features.npy', features)
    logger.info(f"Saved features to {output_path / 'features.npy'}")
    
    # Save labels
    if labels is not None:
        np.save(output_path / 'labels.npy', labels)
        logger.info(f"Saved labels to {output_path / 'labels.npy'}")
    
    # Save catalog
    if catalog is not None:
        catalog.to_csv(output_path / 'catalog.csv', index=False)
        logger.info(f"Saved catalog to {output_path / 'catalog.csv'}")
        
        # Also save with labels if available
        if labels is not None:
            catalog_with_labels = catalog.copy()
            catalog_with_labels['cluster'] = labels
            catalog_with_labels.to_csv(
                output_path / 'catalog_with_clusters.csv', 
                index=False
            )
            logger.info(f"Saved catalog with clusters")
    
    # Save metadata
    if metadata:
        import json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata")
