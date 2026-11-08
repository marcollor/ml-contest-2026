"""Single-sample mean imputation and global normalization compiled with Numba."""

import numpy as np
from numba import njit

# Preserve NaN/Inf checks by keeping the nnan and ninf flags disabled.
FASTMATH_SAFE = {
    "reassoc",  # Allow vectorization across reductions.
    "contract",  # Allow fused multiply-add instructions.
    "arcp",  # Allow reciprocal approximations for division.
    "nsz",  # Treat positive and negative zero identically.
}


@njit(cache=True, fastmath=FASTMATH_SAFE)
def fast_normalize_series_numba(x: np.ndarray) -> np.ndarray:
    """Impute channel means, apply a global Z-score, and clip to [-10, 10].

    Parameters
    ----------
    x : np.ndarray
        Raw sample of shape (4, 256). NaN and infinity are treated as missing.
        Missing values are replaced with the mean of finite values in their
        channel. A channel with no finite values is filled with zeros.

    Returns
    -------
    np.ndarray
        Float32 array of shape (1, 4, 256). A single mean and population
        variance are computed over all 1024 values after imputation, separately
        for each sample (not across the dataset). Scaling uses sqrt(var + 1e-6).
        The input is not modified. Compile during untimed setup if needed.
    """
    if x.shape != (4, 256):
        raise ValueError("Expected a sample of shape (4, 256)")

    imputed = np.empty((4, 256), dtype=np.float64)
    out = np.empty((1, 4, 256), dtype=np.float32)

    for c in range(4):
        channel_sum = 0.0
        finite_count = 0
        for j in range(256):
            value = np.float64(x[c, j])
            if np.isfinite(value):
                channel_sum += value
                finite_count += 1
        channel_mean = channel_sum / finite_count if finite_count else 0.0
        for j in range(256):
            value = np.float64(x[c, j])
            imputed[c, j] = value if np.isfinite(value) else channel_mean

    # Compute shared statistics over all channels and time steps after imputation.
    total = 0.0
    for c in range(4):
        for j in range(256):
            total += imputed[c, j]
    mean = total / 1024.0

    # Center before squaring to avoid cancellation for nearly constant samples.
    squared_deviations = 0.0
    for c in range(4):
        for j in range(256):
            delta = imputed[c, j] - mean
            squared_deviations += delta * delta
    inv_std = 1.0 / np.sqrt(squared_deviations / 1024.0 + 1e-6)

    for c in range(4):
        for j in range(256):
            value = (imputed[c, j] - mean) * inv_std
            out[0, c, j] = min(10.0, max(-10.0, value))
    return out
