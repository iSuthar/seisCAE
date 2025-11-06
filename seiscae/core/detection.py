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
    
    Examples
    --------
    >>> detector = EventDetector(sta_seconds=0.5, lta_seconds=30.0)
    >>> stream, cft, triggers = detector.detect_file("data.mseed")
    >>> print(f"Found {len(triggers)} events")
    """
    
    def __init__(
        self,
        sta_seconds: float = 0.5,
        lta_seconds: float = 30.0,
        threshold_on: float = 25.0,
        threshold_off: float = 3.0,
        highpass_freq: Optional[float] = 1.0,
    ):
        self.sta_seconds = sta_seconds
        self.lta_seconds = lta_seconds
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.highpass_freq = highpass_freq
        
        # Validate parameters
        if sta_seconds >= lta_seconds:
            raise ValueError("STA window must be smaller than LTA window")
        if threshold_on <= threshold_off:
            raise ValueError("Threshold ON must be greater than threshold OFF")
    
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
        
        # Calculate STA/LTA
        df = tr.stats.sampling_rate
        sta_samples = int(self.sta_seconds * df)
        lta_samples = int(self.lta_seconds * df)
        
        cft = classic_sta_lta(tr.data, sta_samples, lta_samples)
        triggers = trigger_onset(cft, self.threshold_on, self.threshold_off)
        
        logger.info(f"Found {len(triggers)} events")
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
