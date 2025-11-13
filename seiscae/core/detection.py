"""Event detection using STA/LTA trigger algorithm."""

import numpy as np
from obspy import read, Stream
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def multi_sta_lta(a, nsta, nlta, delta_sta, delta_lta, epsilon):
    """
    Multiple time window STA/LTA written in Python.
    
    Parameters
    ----------
    a : np.ndarray
        Seismic Trace
    nsta : int
        Length of minimum (or maximum) duration short time average window in samples
    nlta : int
        Length of maximum (or maximum) duration long time average window in samples
    delta_sta : float
        Ratio between the length of the longest and shortest time windows for the short 
        time average; positive gives longer durations than nsta, negative gives shorter durations
    delta_lta : float
        Ratio between the length of the longest and shortest time windows for the long 
        time average; positive gives longer durations than nlta, negative gives shorter durations
    epsilon : float
        Maximum ratio between length of adjacent short time windows
    
    Returns
    -------
    charfct : np.ndarray
        Characteristic function of recursive STA/LTA
    """
    # determine the number of time windows
    if (delta_sta == 1 and delta_lta == 1) or delta_sta <= 0 or delta_lta <= 0:
        nwindows = 1
    else:
        # determine minimum number of time windows for specified epsilon ratio
        nwindows_sta = int(np.log(max(delta_sta, 1./delta_sta))/np.log(max(epsilon, 1./epsilon)) + 1)
        nwindows_lta = int(np.log(max(delta_lta, 1./delta_lta))/np.log(max(epsilon, 1./epsilon)) + 1)
        nwindows = max(nwindows_sta, nwindows_lta)
        # find exact value of the epsilon ratio for this integer number of time windows
        epsilon_sta = np.exp(np.log(delta_sta)/(nwindows - 1))
        epsilon_lta = np.exp(np.log(delta_lta)/(nwindows - 1))
    # compute the characteristic function of the STA/LTA for each time window
    charfct = np.zeros(len(a))
    for i in range(0, nwindows):
        # determine the length of the short time and long time average window in samples
        if nwindows == 1:
            nsta_tmp = nsta
            nlta_tmp = nlta
        else:
            nsta_tmp = int(nsta*epsilon_sta**i)
            nlta_tmp = max(nsta_tmp, int(nlta*epsilon_lta**i))
        # call recursive_sta_lta function; this can be changed to another STA/LTA algorithm
        charfct_tmp = recursive_sta_lta(a, nsta_tmp, nlta_tmp)
        # flag initial time steps for the longer duration STA/LTAs as STA is often larger than LTA
        if (i > 0 and epsilon_lta > 1) or (i < nwindows - 1 and epsilon_lta < 1):
            charfct_tmp[0:int((nsta_tmp + nlta_tmp)/2)] = 0
        charfct = np.maximum(charfct, charfct_tmp)
    return charfct
    
class EventDetector:
    """
    Detect seismic events using STA/LTA trigger algorithm.
    
    Supports both classic single-window and multi-window STA/LTA methods
    with configurable event filtering.
    
    Parameters
    ----------
    sta_seconds : float
        Short-term average window in seconds
    lta_seconds : float
        Long-term average window in seconds
    threshold_on : float
        Trigger ON threshold
    threshold_off : float
        Trigger OFF threshold
    highpass_freq : float, optional
        Highpass filter frequency in Hz
    method : str
        Detection method: "classic" or "multi_window" (default: "classic")
    delta_sta : float
        Ratio between longest and shortest STA windows (for multi_window)
    delta_lta : float
        Ratio between longest and shortest LTA windows (for multi_window)
    epsilon : float
        Maximum ratio between adjacent window lengths (for multi_window)
    min_event_duration : float
        Minimum event duration in seconds (default: 0.0)
    dead_time : float
        Minimum time in seconds between consecutive events (default: 0.0)
    
    Examples
    --------
    Classic STA/LTA with filtering:
    
    >>> detector = EventDetector.create_classic(
    ...     min_event_duration=1.0,
    ...     dead_time=5.0
    ... )
    >>> stream, cft, triggers = detector.detect_file("data.mseed")
    
    Multi-window STA/LTA with filtering:
    
    >>> detector = EventDetector.create_multi_window(
    ...     min_event_duration=0.5,
    ...     dead_time=1.2
    ... )
    >>> stream, cft, triggers = detector.detect_file("data.sac")
    >>> print(f"Found {len(triggers)} events")
    
    Notes
    -----
    The dead_time parameter helps avoid detecting multiple triggers
    for the same event or closely spaced overlapping events.
    """
    
    def __init__(
        self,
        sta_seconds: float = 0.5,
        lta_seconds: float = 30.0,
        threshold_on: float = 25.0,
        threshold_off: float = 3.0,
        highpass_freq: Optional[float] = 1.0,
        method: str = "classic",
        delta_sta: float = 20.0,
        delta_lta: float = 20.0,
        epsilon: float = 2.0,
        min_event_duration: float = 0.0,
        dead_time: float = 1.0,
    ):
        self.sta_seconds = sta_seconds
        self.lta_seconds = lta_seconds
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.highpass_freq = highpass_freq
        self.method = method
        self.delta_sta = delta_sta
        self.delta_lta = delta_lta
        self.epsilon = epsilon
        self.min_event_duration = min_event_duration
        self.dead_time = dead_time
        
        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate detection parameters."""
        if self.sta_seconds >= self.lta_seconds:
            raise ValueError("STA window must be smaller than LTA window")
        if self.threshold_on <= self.threshold_off:
            raise ValueError("Threshold ON must be greater than threshold OFF")

        if self.method not in ["classic", "multi_window"]:
            raise ValueError(f"Unknown method: {self.method}. Use 'classic' or 'multi_window'")

        if self.method == "multi_window":
            if self.delta_sta <= 0 or self.delta_lta <= 0:
                raise ValueError("delta_sta and delta_lta must be positive for multi_sta_lta method")
            if self.epsilon <= 1.0:
                raise ValueError("epsilon must be greater than 1.0 for multi_window method")

        if self.min_event_duration < 0:
            raise ValueError("min_event_duration must be non-negative")

        if self.dead_time < 0:
            raise ValueError("dead_time must be non-negative")

    def _apply_filters(
        self,
        triggers: np.ndarray,
        sampling_rate: float
    ) -> np.ndarray:
        """
        Apply minimum duration and dead time filters to triggers.

        Parameters
        ----------
        triggers : np.ndarray
            Array of (onset, offset) trigger pairs
        sampling_rate : float
            Sampling rate in Hz

        Returns
        -------
        filtered_triggers : np.ndarray
            Filtered array of (onset, offset) trigger pairs
        """
        if len(triggers) == 0:
            return triggers

        # Filter by minimum duration
        if self.min_event_duration > 0:
            min_samples = int(self.min_event_duration * sampling_rate)
            triggers = np.array([
                [onset, offset] for onset, offset in triggers
                if (offset - onset) >= min_samples
            ])
            logger.info(f"After min_duration filter: {len(triggers)} events")

        # Apply dead time filter
        if self.dead_time > 0 and len(triggers) > 1:
            dead_time_samples = int(self.dead_time * sampling_rate)
            filtered_triggers = [triggers[0]] # Always keep the first trigger

            for i in range(1, len(triggers)):
                # Check if this trigger starts after the dead time from the last accepted trigger
                if triggers[i][0] >= (filtered_triggers[-1][1] + dead_time_samples):
                    filtered_triggers.append(triggers[i])

            triggers = np.array(filtered_triggers)
            logger.info(f"After dead_time filter: {len(triggers)} events")

        return triggers
    
    def detect_file(self, filepath: str) -> Tuple[Stream, np.ndarray, List[Tuple[int, int]]]:
        """
        Detect events in a single seismic file.
        
        Parameters
        ----------
        filepath : str
            Path to seismic data file (supports ObsPy readable formats)
            
        Returns
        -------
        stream : obspy.Stream
            Original stream
        cft : np.ndarray
            Characteristic function (STA/LTA ratio)
        triggers : list of tuples
            List of (onset, offset) sample indices for each trigger
        """
        logger.info(f"Processing file: {filepath}")
        
        # Read and preprocess
        st = read(filepath)
        tr = st[0].copy()
        tr.detrend()
        tr.taper(max_percentage=0.05)
        
        if self.highpass_freq:
            tr.filter("highpass", freq=self.highpass_freq)
        
        # Calculate STA/LTA using selected method
        df = tr.stats.sampling_rate
        sta_samples = int(self.sta_seconds * df)
        lta_samples = int(self.lta_seconds * df)
        
        if self.method == "multi_window":
            logger.info(f"Computing multi-window STA/LTA (delta_sta={self.delta_sta}, "
                       f"delta_lta={self.delta_lta}, epsilon={self.epsilon})")
            cft = multi_sta_lta(tr.data, sta_samples, lta_samples, 
                               self.delta_sta, self.delta_lta, self.epsilon)
        else:  # classic
            logger.info("Computing classic STA/LTA")
            cft = classic_sta_lta(tr.data, sta_samples, lta_samples)
        
        # Detect triggers
        triggers = trigger_onset(cft, self.threshold_on, self.threshold_off)
        logger.info(f"Initial triggers detected: {len(triggers)}")
        
        # Apply filters
        triggers = self._apply_filters(triggers, df)
        
        logger.info(f"Final event count: {len(triggers)} events using {self.method} method")
        return st, cft, triggers
    
    def detect_directory(
        self, 
        directory: str, 
        pattern: str = "*",
        component: Optional[str] = None
    ) -> List[Tuple[str, Stream, np.ndarray, List]]:
        """
        Detect events in all files matching pattern in directory.
        
        Parameters
        ----------
        directory : str
            Directory containing seismic data
        pattern : str
            File pattern to match (e.g., "*.mseed", "*.sac")
        component : str, optional
            Component filter (e.g., "EHZ")
            
        Returns
        -------
        results : list
            List of (filepath, stream, cft, triggers) tuples
        """
        path = Path(directory)
        if component:
            path = path / component
        
        files = sorted(path.glob(pattern))
        logger.info(f"Found {len(files)} files to process in {path}")
        
        results = []
        for filepath in files:
            try:
                st, cft, triggers = self.detect_file(str(filepath))
                results.append((str(filepath), st, cft, triggers))
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        return results

    @classmethod
    def create_multi_window(
        cls,
        sta_seconds: float = 0.05,
        lta_seconds: float = 1.0,
        delta_sta: float = 20.0,
        delta_lta: float = 20.0,
        epsilon: float = 2.0,
        threshold_on: float = 2.5,
        threshold_off: float = 1.2,
        highpass_freq: Optional[float] = 1.0,
        min_event_duration: float = 0.0,
        dead_time: float = 0.0,
    ) -> "EventDetector":
        """
        Create detector configured for multi-window STA/LTA.
        
        Parameters
        ----------
        sta_seconds : float
            Short-term average window (default: 0.05s)
        lta_seconds : float
            Long-term average window (default: 1.0s)
        delta_sta : float
            STA window ratio (default: 20.0)
        delta_lta : float
            LTA window ratio (default: 20.0)
        epsilon : float
            Adjacent window ratio (default: 2.0)
        threshold_on : float
            Trigger ON threshold (default: 2.5)
        threshold_off : float
            Trigger OFF threshold (default: 1.2)
        highpass_freq : float, optional
            Highpass filter frequency in Hz (default: 1.0)
        min_event_duration : float
            Minimum event duration in seconds (default: 0.0)
        dead_time : float
            Dead time after event in seconds (default: 0.0)
            
        Returns
        -------
        detector : EventDetector
            Configured detector instance
            
        Examples
        --------
        >>> detector = EventDetector.create_multi_window(
        ...     min_event_duration=0.5,
        ...     dead_time=1.2
        ... )
        >>> st, cft, triggers = detector.detect_file("data.sac")
        """
        return cls(
            sta_seconds=sta_seconds,
            lta_seconds=lta_seconds,
            threshold_on=threshold_on,
            threshold_off=threshold_off,
            highpass_freq=highpass_freq,
            method="multi_window",
            delta_sta=delta_sta,
            delta_lta=delta_lta,
            epsilon=epsilon,
            min_event_duration=min_event_duration,
            dead_time=dead_time,
        )
    
    @classmethod
    def create_classic(
        cls,
        sta_seconds: float = 0.5,
        lta_seconds: float = 30.0,
        threshold_on: float = 25.0,
        threshold_off: float = 3.0,
        highpass_freq: Optional[float] = 1.0,
        min_event_duration: float = 0.0,
        dead_time: float = 0.0,
    ) -> "EventDetector":
        """
        Create detector configured for classic STA/LTA.
        
        Parameters
        ----------
        sta_seconds : float
            Short-term average window (default: 0.5s)
        lta_seconds : float
            Long-term average window (default: 30.0s)
        threshold_on : float
            Trigger ON threshold (default: 25.0)
        threshold_off : float
            Trigger OFF threshold (default: 3.0)
        highpass_freq : float, optional
            Highpass filter frequency in Hz (default: 1.0)
        min_event_duration : float
            Minimum event duration in seconds (default: 0.0)
        dead_time : float
            Dead time after event in seconds (default: 0.0)
            
        Returns
        -------
        detector : EventDetector
            Configured detector instance
            
        Examples
        --------
        >>> detector = EventDetector.create_classic(
        ...     min_event_duration=1.0,
        ...     dead_time=5.0
        ... )
        >>> st, cft, triggers = detector.detect_file("data.mseed")
        """
        return cls(
            sta_seconds=sta_seconds,
            lta_seconds=lta_seconds,
            threshold_on=threshold_on,
            threshold_off=threshold_off,
            highpass_freq=highpass_freq,
            method="classic",
            min_event_duration=min_event_duration,
            dead_time=dead_time,
        )
