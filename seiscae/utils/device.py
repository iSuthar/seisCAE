"""Device management utilities."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(gpu: int = -1) -> torch.device:
    """
    Get torch device.
    
    Parameters
    ----------
    gpu : int
        GPU ID. Use -1 for CPU, 0+ for specific GPU.
        
    Returns
    -------
    device : torch.device
        PyTorch device
    
    Examples
    --------
    >>> device = get_device(0)  # Use GPU 0
    >>> device = get_device(-1)  # Use CPU
    """
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        logger.info(f"Using GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if gpu >= 0:
            logger.warning("GPU requested but CUDA not available. Using CPU.")
        else:
            logger.info("Using CPU")
    
    return device
