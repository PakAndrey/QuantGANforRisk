"""
Utility code
This code from  ICascha/QuantGANs-replication 
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def rolling_window(x, k, sparse=True):
    """compute rolling windows from timeseries

    Args:
        x ([2d array]): x contains the time series in the shape (timestep, sample).
        k ([int]): positive window length.
        sparse (bool): Keep only complete windows. If False, pad the final
            windows with NaN. Defaults to True.

    Returns:
        [3d array]: array of rolling windows in the shape (window, timestep, sample).
    """    
    if k < 1:
        raise ValueError("window length must be positive")

    n = len(x)
    n_windows = max(n - k + 1, 0) if sparse else n
    out = np.full((k, n_windows, *x.shape[1:]), np.nan)
    for lag in range(min(k, n)):
        values = x[lag:lag + n_windows]
        out[lag, :len(values)] = values
    return out

def acf(x, k, le=False):

    
    arr = rolling_window(x, k, sparse=False)
    a = (arr[0] - np.nanmean(arr[0], axis=0))
    if le:
        arr **=2
    b = (arr - np.nanmean(arr, axis=1, keepdims=True))

    return np.nansum((a * b), axis=1) / np.sqrt(np.nansum(a**2, axis=0) * np.nansum(b**2, axis=1))

def cross_acf(x, y, k, le=False):
    
    arr = rolling_window(y, k, sparse=False)
    a = (x - x.mean(axis=0))
    
    if le:
        arr **=2
    b = (arr - np.nanmean(arr, axis=1, keepdims=True))

    return np.nansum((a * b), axis=1) / np.sqrt(np.nansum(a**2, axis=0) * np.nansum(b**2, axis=1))
