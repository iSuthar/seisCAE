"""Spectrogram visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def plot_spectrogram(
    time: np.ndarray,
    freq: np.ndarray,
    spectrogram: np.ndarray,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
    cmap: str = 'OrRd',
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a spectrogram.
    
    Parameters
    ----------
    time : np.ndarray
        Time array
    freq : np.ndarray
        Frequency array
    spectrogram : np.ndarray
        Spectrogram data
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    cmap : str
        Colormap
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.pcolormesh(time, freq, spectrogram, cmap=cmap, vmin=0, vmax=1, shading='auto')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Amplitude', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved spectrogram plot to {save_path}")
        plt.close(fig)
    
    return fig


def plot_waveform_spectrogram(
    waveform: np.ndarray,
    time: np.ndarray,
    freq: np.ndarray,
    spectrogram: np.ndarray,
    title: str = "Waveform and Spectrogram",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot waveform and spectrogram together.
    
    Parameters
    ----------
    waveform : np.ndarray
        Waveform data
    time : np.ndarray
        Time array for spectrogram
    freq : np.ndarray
        Frequency array
    spectrogram : np.ndarray
        Spectrogram data
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 1, hspace=0.3)
    
    # Waveform plot
    ax1 = plt.subplot(gs[0])
    ax1.plot(waveform, linewidth=0.5, color='k')
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_xlim(0, len(waveform))
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3)
    
    # Spectrogram plot
    ax2 = plt.subplot(gs[1:])
    im = ax2.pcolormesh(time, freq, spectrogram, cmap='OrRd', vmin=0, vmax=1, shading='auto')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Amplitude', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=0.98)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved waveform-spectrogram plot to {save_path}")
        plt.close(fig)
    
    return fig


def plot_detection_summary(
    trace,
    cft: np.ndarray,
    triggers: list,
    save_path: str,
    threshold_on: float,
    threshold_off: float,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot detection summary with waveform and STA/LTA.
    
    Parameters
    ----------
    trace : obspy.Trace
        Seismic trace
    cft : np.ndarray
        Characteristic function (STA/LTA)
    triggers : list
        List of trigger tuples
    save_path : str
        Path to save figure
    threshold_on : float
        Trigger ON threshold
    threshold_off : float
        Trigger OFF threshold
    dpi : int
        DPI for saved figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    data = trace.data.astype(float)
    sr = trace.stats.sampling_rate
    t = np.arange(len(data)) / sr
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Waveform plot
    ax[0].plot(t, data, 'k-', linewidth=0.8, label="Waveform")
    for onset, offset in triggers:
        ax[0].axvspan(t[onset], t[offset], color='red', alpha=0.25)
    ax[0].set_ylabel("Amplitude", fontsize=12)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"{trace.stats.station}.{trace.stats.channel}", fontsize=14)
    
    # STA/LTA plot
    ax[1].plot(t, cft, 'b-', linewidth=0.8, label="STA/LTA Ratio")
    ax[1].axhline(threshold_on, color='red', linestyle='--', linewidth=2, label=f"Trigger On ({threshold_on})")
    ax[1].axhline(threshold_off, color='gray', linestyle='--', linewidth=2, label=f"Trigger Off ({threshold_off})")
    ax[1].set_xlabel("Time (s)", fontsize=12)
    ax[1].set_ylabel("STA/LTA Ratio", fontsize=12)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved detection summary to {save_path}")
    plt.close(fig)
    
    return fig
