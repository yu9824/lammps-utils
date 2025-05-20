# savitzky-golay


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Apply Savitzky-Golay filter to smooth data.

    Parameters
    ----------
    y : array_like
        The input data to be smoothed.
    window_size : int
        The length of the filter window (must be odd).
    order : int
        The order of the polynomial used to fit the samples.
    deriv : int, optional
        The order of the derivative to compute (default is 0, which means only smoothing).
    rate : float, optional
        The rate of change of the derivative (default is 1).

    Returns
    -------
    array_like
        The smoothed data.
    """
    from scipy.signal import savgol_filter

    return savgol_filter(y, window_size, order, deriv=deriv, delta=rate)
