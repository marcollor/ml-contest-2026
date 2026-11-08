"""Check numerical preprocessing against an independent NumPy reference."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalize_data import fast_normalize_series_numba


class NormalizeDataTests(unittest.TestCase):
    def assert_normalized(self, raw):
        original = raw.copy()
        imputed = raw.astype(np.float64, copy=True)
        for channel in imputed:
            finite = np.isfinite(channel)
            channel[~finite] = channel[finite].mean() if finite.any() else 0.0
        expected = np.clip(
            (imputed - imputed.mean()) / np.sqrt(imputed.var() + 1e-6), -10, 10
        )[None].astype(np.float32)
        actual = fast_normalize_series_numba(raw)
        self.assertEqual(actual.shape, (1, 4, 256))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_array_equal(raw, original)
        self.assertFalse(np.shares_memory(actual, raw))
        return actual

    def test_global_statistics_across_channels_and_input_layouts(self):
        # Constant channels with different levels distinguish global scaling
        # from independent per-channel scaling.
        raw = np.repeat(np.array([-5, 0, 2, 20])[:, None], 256, axis=1)
        for dtype in (np.float32, np.float64):
            for values in (raw.astype(dtype), np.asfortranarray(raw, dtype=dtype),
                           raw.astype(dtype)[:, ::-1]):
                with self.subTest(dtype=dtype, strides=values.strides):
                    self.assert_normalized(values)

    def test_nan_and_infinities_use_their_channel_mean(self):
        raw = np.random.default_rng(42).normal(size=(4, 256)).astype(np.float32)
        raw[0, :3] = [np.nan, np.inf, -np.inf]
        raw[1] = np.nan
        raw[2] = np.inf
        raw[3, :255] = -np.inf
        self.assert_normalized(raw)

    def test_constant_and_entirely_missing_samples_become_zero(self):
        for value in (0, 7, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                actual = self.assert_normalized(np.full((4, 256), value, np.float32))
                np.testing.assert_array_equal(actual, np.zeros((1, 4, 256), np.float32))

    def test_nearly_constant_large_values_preserve_variance(self):
        raw = np.full((4, 256), 1e9, np.float64)
        raw[:, ::2] += 0.01
        self.assert_normalized(raw)

    def test_extreme_outliers_are_clipped_in_both_directions(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                raw = np.zeros((4, 256), np.float32)
                raw[0, 0] = sign * 1e6
                actual = self.assert_normalized(raw)
                self.assertEqual(actual[0, 0, 0], sign * 10)

    def test_read_only_input_is_supported(self):
        raw = np.arange(1024, dtype=np.float32).reshape(4, 256)
        raw.setflags(write=False)
        self.assert_normalized(raw)

    def test_invalid_sample_dimensions_are_rejected(self):
        for shape in ((4, 255), (3, 256), (256, 4)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                fast_normalize_series_numba(np.zeros(shape, np.float32))


if __name__ == "__main__":
    unittest.main()
