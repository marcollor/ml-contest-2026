"""Fixed V024 preprocessing for raw (4, 256) signals."""

import numpy as np
from numba import njit

ROBUST_CLIP = 6.0
EPSILON = 1e-6
CLIP = 10.0

PREPROCESSING = dict(
    version=1,
    imputation="linear",
    normalization="channel-zscore",
    robust_clip=ROBUST_CLIP,
    epsilon=EPSILON,
    clip=CLIP,
)

# Keep NaN and infinity checks enabled while allowing vectorized reductions.
FASTMATH_SAFE = {"reassoc", "contract", "arcp", "nsz"}


@njit(cache=True, fastmath=FASTMATH_SAFE)
def preprocess_series(x):
    """Return normalized float32 data of shape (1, 4, 256) without modifying x."""
    if x.ndim != 2 or x.shape != (4, 256):
        raise ValueError("Expected a sample of shape (4, 256)")

    imputed = np.empty((4, 256), np.float64)
    for channel in range(4):
        finite_ids = np.where(np.isfinite(x[channel]))[0]
        if len(finite_ids) == 0:
            imputed[channel] = 0.0
            continue

        values = x[channel, finite_ids].astype(np.float64)
        median = np.median(values)
        scale = max(1.4826 * np.median(np.abs(values - median)), np.sqrt(EPSILON))
        bound = ROBUST_CLIP * scale
        values = np.minimum(np.maximum(values, median - bound), median + bound)
        imputed[channel] = np.interp(np.arange(256), finite_ids, values)

    for channel in range(4):
        imputed[channel] -= np.mean(imputed[channel])
        imputed[channel] /= np.sqrt(np.mean(imputed[channel] ** 2) + EPSILON)

    return np.minimum(np.maximum(imputed, -CLIP), CLIP).astype(np.float32).reshape(1, 4, 256)
