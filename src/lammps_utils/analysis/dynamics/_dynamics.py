from array import array
from typing import Literal

import numpy as np

from lammps_utils.helpers import is_installed

if is_installed("tqdm"):
    from tqdm.auto import tqdm
else:
    from lammps_utils.helpers import dummy_tqdm as tqdm


def compute_msd(
    coordinates: np.ndarray,
    method: Literal["simple", "fft"] = "fft",
    max_lag: int | None = None,
) -> list[float]:
    """
    Compute the mean squared displacement (MSD).

    The input coordinates must be unwrapped. Periodic boundary
    crossings should already be removed before calling this function.

    Parameters
    ----------
    coordinates : np.ndarray
        Atomic **unwrapped** coordinates with shape
        ``(n_timesteps, n_atoms, n_dim)``.
    method : {"simple", "fft"}, default="fft"
        Method used to compute the MSD.

        - ``"simple"``: direct calculation using time-lagged differences.
        - ``"fft"``: FFT-based calculation.
    max_lag : int, optional
        Maximum lag time. If ``None``, use
        ``n_timesteps // 4``.

    Returns
    -------
    list[float]
        MSD values for lag times from 0 to ``max_lag - 1``.

    Notes
    -----
    Coordinates must be unwrapped. Using wrapped coordinates
    under periodic boundary conditions will give incorrect MSD
    values.

    The MSD is defined as

    MSD(t) = <|r(t0 + t) - r(t0)|^2>

    where the average is taken over atoms and time origins.
    """
    N_timesteps, N_atoms, n_dim = coordinates.shape
    if max_lag is None:
        max_lag = N_timesteps // 4

    if method == "simple":
        arr_msd_simple = array("f", [0.0])  # n_lag == 0
        for n_lag in tqdm(range(1, max_lag)):
            arr_msd_simple.append(
                np.mean(
                    # np.hstack(
                    #     [
                    #         np.square(np.diff(coordinates[i::n_lag], axis=0))
                    #         .sum(axis=2)
                    #         .ravel()
                    #         for i in range(n_lag)
                    #     ]
                    # ),
                    np.square(coordinates[n_lag:] - coordinates[:-n_lag]).sum(
                        axis=2
                    ),
                    axis=None,
                ).item()
            )
        return arr_msd_simple.tolist()
    elif method == "fft":
        n_fft = 2 * N_timesteps

        f = np.fft.rfft(coordinates, n=n_fft, axis=0)
        p = f * np.conjugate(f)  # |f|^2
        ac_sum = np.fft.irfft(p, n=n_fft, axis=0)[
            :N_timesteps
        ].real  # (N_timesteps, N_atoms, n_dim)
        denom = (N_timesteps - np.arange(N_timesteps))[
            :, np.newaxis, np.newaxis
        ]  # (N_timesteps, 1, 1)
        ac = ac_sum / denom

        cumsum = np.empty(
            (N_timesteps + 1, N_atoms, n_dim), dtype=coordinates.dtype
        )
        cumsum[1:] = np.cumsum(np.square(coordinates), axis=0)
        cumsum[0] = 0.0

        tau = np.arange(N_timesteps)
        sum0 = cumsum[N_timesteps] - cumsum[tau]
        sum1 = cumsum[N_timesteps - tau]
        s1 = (sum0 + sum1) / denom

        msd_comp = s1 - 2.0 * ac
        msd_i = msd_comp.sum(axis=2)
        msd = msd_i.mean(axis=1)

        msd[0] = 0.0
        return msd.tolist()[:max_lag]
    else:
        raise ValueError(f"Invalid method: {method}")


def compute_gs(
    coordinates: np.ndarray,
    bins: int = 100,
    r_max: float = 20.0,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the self part of the van Hove correlation function.

    The input coordinates must be unwrapped.

    Parameters
    ----------
    coordinates : np.ndarray
        Atomic **unwrapped** coordinates with shape
        ``(n_timesteps, n_atoms, n_dim)``.
    bins : int, default=100
        Number of histogram bins.
    r_max : float, default=20.0
        Maximum displacement included in the histogram.
    max_lag : int, optional
        Maximum lag time. If ``None``, use
        ``n_timesteps // 4``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Histogram values with shape
        ``(max_lag, bins)``
        and bin edges.
    """
    n_timesteps = coordinates.shape[0]

    if max_lag is None:
        max_lag = n_timesteps // 4

    r_edges = np.linspace(0.0, r_max, bins + 1)

    gs = np.empty((max_lag, bins), dtype=np.float64)

    gs[0] = 0.0
    gs[0, 0] = 1.0

    for n_lag in tqdm(range(1, max_lag)):
        dr = coordinates[n_lag:] - coordinates[:-n_lag]

        r = np.linalg.norm(dr, axis=2).ravel()

        hist, _ = np.histogram(
            r,
            bins=r_edges,
            density=True,
        )

        gs[n_lag] = hist

    return gs, r_edges


def compute_fs(
    coordinates: np.ndarray,
    k: float,
    max_lag: int | None = None,
) -> list[float]:
    """
    Compute the self-intermediate scattering function.

    The input coordinates must be unwrapped.

    Parameters
    ----------
    coordinates : np.ndarray
        Atomic **unwrapped** coordinates with shape
        ``(n_timesteps, n_atoms, n_dim)``.
    k : float
        Wave number.
    max_lag : int, optional
        Maximum lag time. If ``None``, use
        ``n_timesteps // 4``.

    Returns
    -------
    list[float]
        Values of the self-intermediate scattering function.
    """
    n_timesteps = coordinates.shape[0]

    if max_lag is None:
        max_lag = n_timesteps // 4

    fs = array("f", [1.0])

    for n_lag in tqdm(range(1, max_lag)):
        dr = coordinates[n_lag:] - coordinates[:-n_lag]

        r = np.linalg.norm(dr, axis=2)

        kr = k * r

        sinc = np.ones_like(kr)

        mask = kr > 1.0e-12

        sinc[mask] = np.sin(kr[mask]) / kr[mask]

        fs.append(sinc.mean().item())

    return fs.tolist()
