"""Spectrogram generation and preprocessing utilities."""

import numpy as np
from scipy.signal import stft
from scipy.interpolate import interp1d
from obspy import Stream, Trace
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """
    Generate spectrograms from seismic waveforms.
    
    Parameters
    ----------
    nperseg : int
        Length of each segment for STFT
    noverlap_ratio : float
        Overlap ratio (0-1)
    nfft : int
        FFT length
    freq_min : float
        Minimum frequency in Hz
    freq_max : float
        Maximum frequency in Hz
    time_bins : int
        Target number of time bins
    freq_bins : int
        Target number of frequency bins
    
    Examples
    --------
    >>> gen = SpectrogramGenerator(freq_min=1.0, freq_max=50.0)
    >>> f, t, spec = gen.generate(trace)
    >>> print(f"Spectrogram shape: {spec.shape}")
    """
    
    def __init__(
        self,
        nperseg: int = 128,
        noverlap_ratio: float = 0.9,
        nfft: int = 512,
        freq_min: float = 1.0,
        freq_max: float = 50.0,
        time_bins: int = 40,
        freq_bins: int = 256,
        window_seconds: float = 4.0,
    ):
        self.nperseg = nperseg
        self.noverlap_ratio = noverlap_ratio
        self.noverlap = int(nperseg * noverlap_ratio)
        self.nfft = nfft
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.time_bins = time_bins
        self.freq_bins = freq_bins
        self.window_seconds = window_seconds
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate STFT and spectrogram parameters."""
        if self.noverlap >= self.nperseg:
            raise ValueError(
                f"noverlap ({self.noverlap}) must be < nperseg ({self.nperseg})"
            )
        if self.nfft < self.nperseg:
            raise ValueError(
                f"nfft ({self.nfft}) must be >= nperseg ({self.nperseg})"
            )
        if self.freq_min >= self.freq_max:
            raise ValueError(
                f"freq_min ({self.freq_min}) must be < freq_max ({self.freq_max})"
            )
        if self.window_seconds <= 0:
            raise ValueError(
                f"window_seconds ({self.window_seconds}) must be positive"
            )
        if self.time_bins <= 0 or self.freq_bins <= 0:
            raise ValueError(
                "time_bins and freq_bins must be positive integers"
            )
    
    def _find_center_time(self, trace: Trace) -> float:
        """
        Find time of maximum energy for centering using quick STFT.
        
        Matches research paper implementation:
        - Uses nperseg=64 (shorter window for quick detection)
        - Finds maximum in time-frequency spectrogram
        - Centers 4-second window on this maximum
        
        Parameters
        ----------
        trace : obspy.Trace
            Input seismic trace
            
        Returns
        -------
        center_time : obspy.UTCDateTime
            UTCDateTime of the center time for window extraction
        """
        df = trace.stats.sampling_rate
        
        # If trace too short, use start time
        if len(trace.data) < 128:
            logger.warning(f"Trace too short ({len(trace.data)} samples), using starttime")
            return trace.stats.starttime
        
        # Quick STFT with 64-sample segments (research paper parameters)
        f1, t1, zxx1 = stft(
            trace.data,
            fs=df,
            nperseg=64,
            noverlap=int(64 * 0.9),
            nfft=512
        )
        
        # Find location of maximum amplitude in time-frequency space
        j, k = np.where(np.abs(zxx1) == np.max(np.abs(zxx1)))
        center_time = trace.stats.starttime + t1[k[0]]
        
        logger.debug(f"Center time found at {center_time} (offset: {t1[k[0]]}s)")
        
        return center_time
    
    def generate(
        self, 
        trace: Trace, 
        center_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spectrogram from trace.
        
        Parameters
        ----------
        trace : obspy.Trace
            Seismic trace
        center_time : float, optional
            Center time for window. If None, uses max amplitude
            
        Returns
        -------
        f : np.ndarray
            Frequency array
        t : np.ndarray
            Time array
        amp_zxx : np.ndarray
            Normalized amplitude spectrogram (freq_bins x time_bins)
        """
        df = trace.stats.sampling_rate
        
        # Validate freq_max against Nyquist frequency
        nyquist = df / 2.0
        if self.freq_max > nyquist:
            logger.warning(
                f"freq_max ({self.freq_max} Hz) exceeds Nyquist frequency "
                f"({nyquist} Hz). Clamping to Nyquist."
            )
            effective_freq_max = nyquist
        else:
            effective_freq_max = self.freq_max
        
        # Determine center time
        if center_time is None:
            center_time = self._find_center_time(trace)
        
        # Extract window
        tr_window = trace.copy()
        half_window = self.window_seconds / 2
        tr_window.trim(center_time - half_window, center_time + half_window)
        
        # Validate and fix window length
        expected_samples = int(self.window_seconds * df) + 1
        actual_samples = len(tr_window.data)
        
        if actual_samples != expected_samples:
            if actual_samples < expected_samples * 0.9:
                # More than 10% short - this is a problem
                raise ValueError(
                    f"Insufficient data for spectrogram: got {actual_samples} samples, "
                    f"expected {expected_samples} (window={self.window_seconds}s at {df} Hz)"
                )
            elif actual_samples < expected_samples:
                # Pad to expected length
                logger.warning(
                    f"Padding window: got {actual_samples}, expected {expected_samples}"
                )
                padding = expected_samples - actual_samples
                tr_window.data = np.pad(tr_window.data, (0, padding), mode='constant')
            else:
                # Truncate to expected length
                logger.warning(
                    f"Truncating window: got {actual_samples}, expected {expected_samples}"
                )
                tr_window.data = tr_window.data[:expected_samples]
        
        # Compute STFT
        f, t, zxx = stft(
            tr_window.data,
            fs=df,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft
        )
        
        # Remove DC component (research paper: f2 = f2[1:]; amp_zxx = np.abs(zxx2)[1:])
        f = f[1:]
        amp_zxx = np.abs(zxx)[1:]
        
        # Filter frequencies
        freq_mask = (f >= self.freq_min) & (f <= effective_freq_max)
        f = f[freq_mask]
        amp_zxx = amp_zxx[freq_mask, :]
        
        # Resample to target dimensions
        amp_zxx = self._resample_spectrogram(amp_zxx, t, f)
        
        # Normalize
        amp_zxx = self._normalize(amp_zxx)
        
        # Generate target arrays for output
        t_out = np.linspace(0.0, self.window_seconds, self.time_bins)
        f_out = np.linspace(self.freq_min, effective_freq_max, self.freq_bins)
        
        return f_out, t_out, amp_zxx
    
    def _resample_spectrogram(
        self, 
        amp_zxx: np.ndarray, 
        t: np.ndarray, 
        f: np.ndarray
    ) -> np.ndarray:
        """Resample spectrogram to target dimensions."""
        # Time resampling
        target_t = np.linspace(0.0, self.window_seconds, self.time_bins)
        
        if len(t) != self.time_bins:
            t_start = t[0]
            t_end = t[-1] if t[-1] > t_start else (t_start + 1e-9)
            scaled_t = (t - t_start) * (self.window_seconds / (t_end - t_start))
            
            amp_resampled = np.zeros((amp_zxx.shape[0], self.time_bins))
            for i in range(amp_zxx.shape[0]):
                amp_resampled[i] = np.interp(target_t, scaled_t, amp_zxx[i])
            amp_zxx = amp_resampled
        
        # Frequency resampling
        if amp_zxx.shape[0] != self.freq_bins:
            f_interp = interp1d(
                np.linspace(0, 1, amp_zxx.shape[0]),
                amp_zxx,
                axis=0,
                kind='linear'
            )
            amp_zxx = f_interp(np.linspace(0, 1, self.freq_bins))
        
        return amp_zxx
    
    def _normalize(self, amp_zxx: np.ndarray) -> np.ndarray:
        """Normalize spectrogram to [0, 1]."""
        amp_zxx = amp_zxx - np.min(amp_zxx)
        max_val = np.max(amp_zxx)
        if max_val > 1e-10:  # Use epsilon to avoid division by near-zero
            amp_zxx = amp_zxx / max_val
        else:
            logger.warning(
                "Spectrogram has near-zero amplitude range, returning zeros"
            )
            amp_zxx = np.zeros_like(amp_zxx)
        return amp_zxx


class EventExtractor:
    """
    Extract individual events from triggered waveforms.
    
    Parameters
    ----------
    spectrogram_generator : SpectrogramGenerator
        Spectrogram generator instance
    
    Examples
    --------
    >>> spec_gen = SpectrogramGenerator()
    >>> extractor = EventExtractor(spec_gen)
    >>> events = extractor.extract_events(trace, triggers)
    """
    
    def __init__(self, spectrogram_generator: SpectrogramGenerator):
        self.spec_gen = spectrogram_generator
    
    def extract_events(
        self,
        trace: Trace,
        triggers: list,
        component: str = "EHZ"
    ) -> list:
        """
        Extract individual events from trace.
        
        Parameters
        ----------
        trace : obspy.Trace
            Seismic trace
        triggers : list of tuples
            List of (onset, offset) sample indices
        component : str
            Component name
            
        Returns
        -------
        events : list of dict
            List of event dictionaries containing metadata and data
        """
        events = []
        df = trace.stats.sampling_rate
        t = trace.stats.starttime
        
        for onset, offset in triggers:
            try:
                # Extract event window
                start_time = t + onset / df
                end_time = t + offset / df
                duration = (offset - onset) / df
                
                # Trim trace to event window
                tr_event = trace.copy()
                tr_event.trim(start_time, end_time)
                
                # Calculate RMS energy (research paper approach)
                energy = np.sqrt(np.mean(tr_event.data ** 2))
                
                # Find center time using max energy in spectrogram
                # This matches research paper: center on max spectrogram amplitude
                center = self.spec_gen._find_center_time(tr_event)
                
                # Generate spectrogram centered on max energy point
                f, t_spec, amp_zxx = self.spec_gen.generate(trace, center_time=center)
                
                # Create event metadata
                event = {
                    'station': trace.stats.station,
                    'channel': trace.stats.channel,
                    'component': component,
                    'starttime': str(start_time),
                    'endtime': str(end_time),
                    'duration': duration,
                    'center_time': str(center),
                    'energy': energy,
                    'sampling_rate': df,
                    'waveform': tr_event.data,
                    'spectrogram': amp_zxx,
                    'frequency': f,
                    'time': t_spec,
                }
                
                events.append(event)
                
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Skipping event at {onset}-{offset} due to data issue: {e}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error extracting event at {onset}-{offset}: {e}",
                    exc_info=True
                )
                continue
        
        return events
