"""Optional fused single-sample normalization. Compile during untimed setup."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def fast_normalize_series_numba(x: np.ndarray) -> np.ndarray:
    """Fused 1D linear interpolation, Z-score scaling, and clipping in Numba.

    Parameters
    ----------
    x : np.ndarray
        Raw input array of shape (4, 256).

    Returns
    -------
    np.ndarray
        Output tensor of shape (1, 4, 256) in float32 format.
    """
    out = np.empty((1, 4, 256), dtype=np.float32)
    interp = np.empty(256, dtype=np.float64)

    for c in range(4):
        # -------------------------------------------------------------
        # Step 1: Find the first and last valid (finite) indices
        # -------------------------------------------------------------
        first_valid = -1
        last_valid = -1
        for j in range(256):
            if np.isfinite(x[c, j]):
                if first_valid == -1:
                    first_valid = j
                last_valid = j

        # Channel contains only non-finite values (NaN / Inf): zero out
        if first_valid == -1:
            for j in range(256):
                out[0, c, j] = 0.0
            continue

        # Single valid observation: constant series with zero variance
        if first_valid == last_valid:
            for j in range(256):
                out[0, c, j] = 0.0
            continue

        # -------------------------------------------------------------
        # Step 2: Linear interpolation and boundary padding
        # -------------------------------------------------------------
        # Backward fill before the first valid observation
        for j in range(first_valid):
            interp[j] = x[c, first_valid]

        # Interior two-pointer linear interpolation
        prev_idx = first_valid
        for j in range(first_valid + 1, last_valid + 1):
            if np.isfinite(x[c, j]):
                curr_idx = j
                v_start = x[c, prev_idx]
                v_end = x[c, curr_idx]
                step = (v_end - v_start) / (curr_idx - prev_idx)

                interp[prev_idx] = v_start
                for k in range(prev_idx + 1, curr_idx):
                    interp[k] = v_start + step * (k - prev_idx)

                prev_idx = curr_idx
        interp[last_valid] = x[c, last_valid]

        # Forward fill after the last valid observation
        for j in range(last_valid + 1, 256):
            interp[j] = x[c, last_valid]

        # -------------------------------------------------------------
        # Step 3: Compute statistics, scale (Z-score), and clamp [-10, 10]
        # -------------------------------------------------------------
        sum_val = 0.0
        sum_sq = 0.0
        for j in range(256):
            v = interp[j]
            sum_val += v
            sum_sq += v * v

        mean = sum_val / 256.0
        var = max(0.0, (sum_sq / 256.0) - (mean * mean))
        inv_std = 1.0 / np.sqrt(var + 1e-6)

        for j in range(256):
            norm_val = (interp[j] - mean) * inv_std
            if norm_val > 10.0:
                out[0, c, j] = 10.0
            elif norm_val < -10.0:
                out[0, c, j] = -10.0
            else:
                out[0, c, j] = norm_val

    return out