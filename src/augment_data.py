"""Seeded corruptions of raw signals, independent of labels and metadata."""
import numpy as np


def augment_raw(raw, rng, missing_max=0.0, noise_max=0.0,
                spike_probability=0.0, spike_scale=10.0,
                gain_min=1.0, gain_max=1.0):
    """Return a copy; noise and spikes are relative to each channel's MAD scale.

    Missing probability and noise intensity vary per sample. Existing missing
    values remain missing. Entirely missing or constant channels use a scale floor.
    """
    x = np.array(raw, dtype=np.float32, copy=True)
    clean = np.where(np.isfinite(x), x, np.nan)
    # Explicit handling avoids all-NaN reduction warnings.
    empty = ~np.isfinite(clean).any(axis=2)
    clean[empty] = 0
    median = np.nanmedian(clean, axis=2, keepdims=True)
    scale = np.maximum(1.4826*np.nanmedian(np.abs(clean-median), axis=2, keepdims=True), 1e-3)
    if gain_min != 1 or gain_max != 1:
        gain = np.exp(rng.uniform(np.log(gain_min), np.log(gain_max), (len(x), 4, 1)))
        x = median + (x-median)*gain
        scale = scale*gain
    if noise_max:
        intensity = rng.uniform(0, noise_max, (len(x), 1, 1))
        x += rng.normal(size=x.shape)*intensity*scale
    if spike_probability:
        mask = rng.random(x.shape) < spike_probability
        x += mask*rng.choice(np.array([-1, 1]), size=x.shape)*spike_scale*scale
    if missing_max:
        probability = rng.uniform(0, missing_max, (len(x), 1, 1))
        x[rng.random(x.shape) < probability] = np.nan
    return x.astype(np.float32)


def inverse_frequency_weights(labels, n_classes=8):
    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Class weighting requires all classes in the training split")
    weights = 1.0/counts
    return (weights/weights.mean()).astype(np.float32)
