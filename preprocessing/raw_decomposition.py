"""Pure-array raw SSH preprocessing and hard frequency-wavenumber separation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend


FloatArray = NDArray[np.float64]


def _as_finite_ssh(ssh: ArrayLike) -> FloatArray:
    array = np.asarray(ssh, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("ssh must have shape [time, y, x]")
    if min(array.shape) < 2:
        raise ValueError("all SSH dimensions must contain at least two samples")
    if not np.isfinite(array).all():
        raise ValueError("ssh contains NaN or infinite values")
    return array


def preprocess_raw_ssh(ssh: ArrayLike) -> FloatArray:
    """Remove each frame's spatial mean and each grid point's temporal trend."""

    processed = _as_finite_ssh(ssh).copy()
    processed -= processed.mean(axis=(-2, -1), keepdims=True)
    return np.asarray(detrend(processed, axis=0, type="linear"), dtype=np.float64)


def hard_dispersion_decomposition(
    preprocessed_ssh: ArrayLike,
    *,
    f_local: float,
    c_cutoff: float,
    dx: float,
    dy: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Separate BM and UBM with a hard IGW dispersion cutoff.

    Spectral coefficients satisfying

        omega**2 > f_local**2 + c_cutoff**2 * (kx**2 + ky**2)

    are assigned to UBM. All remaining coefficients are assigned to BM.
    Spatial axes use meters, time uses seconds, ``f_local`` uses rad/s, and
    ``c_cutoff`` uses m/s.
    """

    ssh = _as_finite_ssh(preprocessed_ssh)
    scalars = np.asarray((f_local, c_cutoff, dx, dy, dt), dtype=np.float64)
    if not np.isfinite(scalars).all():
        raise ValueError("decomposition parameters must be finite")
    if c_cutoff <= 0 or dx <= 0 or dy <= 0 or dt <= 0:
        raise ValueError("c_cutoff, dx, dy, and dt must be positive")

    nt, ny, nx = ssh.shape
    field = np.transpose(ssh, (1, 2, 0))
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)

    radial_wavenumber_squared = ky[:, None] ** 2 + kx[None, :] ** 2
    threshold = float(f_local) ** 2 + float(c_cutoff) ** 2 * radial_wavenumber_squared
    ubm_mask = omega[None, None, :] ** 2 > threshold[:, :, None]

    spectrum = np.fft.fftn(field)
    ubm = np.fft.ifftn(spectrum * ubm_mask).real
    bm = np.fft.ifftn(spectrum * ~ubm_mask).real
    return np.transpose(bm, (2, 0, 1)), np.transpose(ubm, (2, 0, 1))


def preprocess_and_decompose_raw(
    ssh: ArrayLike,
    *,
    f_local: float,
    c_cutoff: float,
    dx: float,
    dy: float,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Apply raw preprocessing and return processed SSH, BM, and UBM."""

    processed = preprocess_raw_ssh(ssh)
    bm, ubm = hard_dispersion_decomposition(
        processed,
        f_local=f_local,
        c_cutoff=c_cutoff,
        dx=dx,
        dy=dy,
        dt=dt,
    )
    return processed, bm, ubm

