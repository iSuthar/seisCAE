"""Seismic data reading utilities."""

from obspy import read, Stream
from pathlib import Path
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


def read_seismic_data(filepath: Union[str, Path]) -> Stream:
    """
    Read seismic data from file.
    
    Supports all formats compatible with ObsPy, including:
    - MiniSEED (.mseed)
    - SAC (.sac)
    - SEG-Y (.segy, .sgy)
    - GSE2 (.gse)
    - And many more...
    
    Parameters
    ----------
    filepath : str or Path
        Path to seismic data file
        
    Returns
    -------
    stream : obspy.Stream
        Seismic data stream
    
    Examples
    --------
    >>> stream = read_seismic_data("data.mseed")
    >>> print(f"Read {len(stream)} traces")
    """
    try:
        stream = read(str(filepath))
        logger.debug(f"Successfully read {filepath}: {len(stream)} traces")
        return stream
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        raise


def read_multiple_files(
    filepaths: List[Union[str, Path]]
) -> List[Stream]:
    """
    Read multiple seismic data files.
    
    Parameters
    ----------
    filepaths : list of str or Path
        List of file paths
        
    Returns
    -------
    streams : list of obspy.Stream
        List of seismic data streams
    
    Examples
    --------
    >>> files = ["data1.mseed", "data2.mseed"]
    >>> streams = read_multiple_files(files)
    """
    streams = []
    for filepath in filepaths:
        try:
            stream = read_seismic_data(filepath)
            streams.append(stream)
        except Exception as e:
            logger.warning(f"Skipping {filepath}: {e}")
    
    logger.info(f"Successfully read {len(streams)}/{len(filepaths)} files")
    return streams
