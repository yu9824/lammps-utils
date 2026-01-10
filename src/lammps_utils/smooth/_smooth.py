"""Module for data smoothing using Savitzky-Golay filter."""

from numpy.typing import ArrayLike


def savitzky_golay(
    y: ArrayLike, window_size: int, order: int, deriv: int = 0, rate: float = 1
) -> ArrayLike:
    """
    Apply Savitzky-Golay filter to smooth data.

    This function uses scipy's savgol_filter to apply a Savitzky-Golay filter,
    which fits a polynomial to local subsets of the data and uses the polynomial
    to determine the smoothed value at each point.

    Parameters
    ----------
    y : ArrayLike
        The input data to be smoothed.
    window_size : int
        The length of the filter window (must be odd and greater than `order`).
    order : int
        The order of the polynomial used to fit the samples. Must be less than `window_size`.
    deriv : int, optional
        The order of the derivative to compute. Default is 0 (smoothing only).
    rate : float, optional
        The rate of change of the derivative. Default is 1.

    Returns
    -------
    ArrayLike
        The smoothed data with the same shape as the input.

    Notes
    -----
    The Savitzky-Golay filter preserves features of the data such as relative maxima,
    minima, and width, which are usually flattened by other smoothing techniques.
    """
    from scipy.signal import savgol_filter

    return savgol_filter(y, window_size, order, deriv=deriv, delta=rate)
