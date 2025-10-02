import numpy as np
from astropy.timeseries import LombScargle

def calc_lomb_scargle(t, y, fmin=None, fmax=None, oversample=10, return_type='muhz'):
    """
    Compute the Lomb-Scargle amplitude spectrum.
    
    Parameters
    ----------
    t : array_like
        Time array (in days)
    y : array_like
        Flux (normalized)
    fmin : float or None
        Minimum frequency [in c/d]; if None, defaults to 1 / baseline
    fmax : float or None
        Maximum frequency [in c/d]; if None, defaults to Nyquist frequency
    oversample : int
        Frequency oversampling factor
    return_type : str
        'muhz' or 'cd' for output frequency units
        
    Returns
    -------
    freq : ndarray
        Frequencies (μHz or c/d)
    amp : ndarray
        Amplitude spectrum (same units as input y)
    """
    t = np.asarray(t)
    y = y - np.mean(y)

    baseline = t.max() - t.min()
    df = 1.0 / baseline
    res = np.median(np.diff(t))
    nyquist = 0.5 / res

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = nyquist

    freq_cd = np.arange(fmin, fmax, df / oversample)

    ls = LombScargle(t, y)
    power = ls.power(freq_cd, normalization='psd')
    amp = np.sqrt(power) * np.sqrt(4.0 / len(t))

    if return_type == 'muhz':
        return freq_cd * 11.574, amp
    else:
        return freq_cd, amp
    
    
def spectral_window(t, oversample=1, return_type='muhz'):
    """
    Compute the spectral window and effective observation time Tobs.

    Parameters
    ----------
    t : array
        Time array (in days)
    oversample : int
        Oversampling factor for frequency grid
    return_type : str
        'muhz' or 'cd' for frequency units

    Returns
    -------
    freq : array
        Frequency array (μHz or c/d)
    window : array
        Spectral window power (dimensionless)
        the true spectral window should be normalized by Tobs
    Tobs : float
        Effective observing time (in days)
    """
    df = 1.0 / (t.max() - t.min())
    res = np.median(np.diff(t))
    nyquist = 0.5 / res
    freq_cd = np.arange(df, nyquist, df / oversample)

    nu = 0.5 * (df + nyquist)
    model = LombScargle(t, np.sin(2 * np.pi * nu * t))
    power = model.power(freq_cd, normalization="psd")
    power /= len(t)
    power *= 4.0

    delta_freq = np.median(np.diff(freq_cd))
    Tobs = 1.0 / np.sum(delta_freq * power)

    if return_type == 'muhz':
        return freq_cd * 11.574, power, Tobs
    else:
        return freq_cd, power, Tobs
    
