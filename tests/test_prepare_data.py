"""Exercise dataset verification and reproducible splits without network access."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import prepare_data
from competition_api import Dataset


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data/train_100k.npz"
        self.data.parent.mkdir()
        with zipfile.ZipFile(self.data, "w") as archive:
            archive.writestr("fixture", b"small archive for integrity checks")
        self.out = self.root / "outputs"
        self.dataset = Dataset(
            series=np.broadcast_to(np.arange(100000, dtype=np.float32).reshape(-1, 1, 1),
                                   (100000, 4, 256)),
            labels=np.tile(np.arange(8), 12500),
            sample_index=np.arange(100000),
            meta={"subject": np.arange(100000), "group": np.arange(100000) % 3},
        )
        # Only the large release's size and signature are simulated. ZIP
        # validation, splitting and output serialization use real implementations.
        original_stat = Path.stat
        data = self.data

        def fixture_stat(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            return SimpleNamespace(st_size=413201540) if path == data else result

        self.start_patch(patch.object(Path, "stat", fixture_stat))
        self.digest = self.start_patch(patch.object(prepare_data.hashlib, "file_digest"))
        self.digest.return_value.hexdigest.return_value = (
            "1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668"
        )
        self.load = self.start_patch(patch("competition_api.load_dataset", return_value=self.dataset))
        self.download = self.start_patch(patch.object(prepare_data.urllib.request, "urlretrieve"))
        self.start_patch(patch.object(sys, "path", sys.path.copy()))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_main(self):
        argv = ["prepare_data.py", "--repo", str(self.root), "--out", str(self.out)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            prepare_data.main()

    def test_splits_are_disjoint_stratified_and_reproducible_with_aligned_validation(self):
        previous = None
        for _ in range(2):
            self.run_main()
            with np.load(self.out / "split_indices.npz") as saved:
                parts = {name: saved[name] for name in saved.files}
            self.assertEqual({name: len(indices) for name, indices in parts.items()},
                             {"train": 70000, "reserve": 5000, "dev": 5000, "iid": 20000})
            np.testing.assert_array_equal(
                np.sort(np.concatenate(list(parts.values()))), np.arange(100000)
            )
            for indices in (parts["dev"], parts["iid"],
                            np.concatenate((parts["train"], parts["reserve"]))):
                np.testing.assert_array_equal(
                    np.bincount(self.dataset.labels[indices], minlength=8),
                    np.full(8, len(indices) // 8),
                )
            if previous is not None:
                for name, indices in parts.items():
                    np.testing.assert_array_equal(indices, previous[name])
            previous = parts
        with np.load(self.out / "val.npz") as validation:
            self.assertEqual(set(validation.files), {"series", "labels", "meta__subject", "meta__group"})
            np.testing.assert_array_equal(validation["series"], self.dataset.series[parts["iid"]])
            np.testing.assert_array_equal(validation["labels"], self.dataset.labels[parts["iid"]])
            for name, values in self.dataset.meta.items():
                np.testing.assert_array_equal(validation["meta__" + name], values[parts["iid"]])
        self.download.assert_not_called()
        self.load.assert_called_with(self.data)

    def test_missing_dataset_downloads_via_temporary_file(self):
        archive_bytes = self.data.read_bytes()
        self.data.unlink()

        def download(url, destination):
            self.assertFalse(self.data.exists())
            self.assertEqual(destination, self.data.with_suffix(".download"))
            destination.write_bytes(archive_bytes)

        self.download.side_effect = download
        # Stop after integrity checks to avoid repeating the large output test.
        self.load.side_effect = RuntimeError("dataset verified")
        with self.assertRaisesRegex(RuntimeError, "dataset verified"):
            self.run_main()
        self.download.assert_called_once_with(
            "https://github.com/GDG-UAM/ml-contest-materials/releases/download/v1.0/train_100k.npz",
            self.data.with_suffix(".download"),
        )
        self.assertEqual(self.data.read_bytes(), archive_bytes)
        self.assertFalse(self.data.with_suffix(".download").exists())

    def test_failed_download_does_not_publish_dataset_or_outputs(self):
        self.data.unlink()
        self.download.side_effect = OSError("download interrupted")
        with self.assertRaisesRegex(OSError, "download interrupted"):
            self.run_main()
        self.assertFalse(self.data.exists())
        self.assertFalse(self.out.exists())
        self.load.assert_not_called()

    def test_unexpected_file_size_is_rejected_before_loading(self):
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1)):
            with self.assertRaisesRegex(AssertionError, "Unexpected dataset size"):
                self.run_main()
        self.digest.assert_not_called()
        self.load.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_unexpected_signature_is_rejected_before_loading(self):
        self.digest.return_value.hexdigest.return_value = "wrong signature"
        with self.assertRaisesRegex(AssertionError, "Unexpected dataset signature"):
            self.run_main()
        self.load.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_corrupt_zip_member_is_rejected_before_loading(self):
        with patch.object(zipfile.ZipFile, "testzip", return_value="damaged.npy"):
            with self.assertRaises(AssertionError):
                self.run_main()
        self.load.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_invalid_series_shape_or_dtype_is_rejected(self):
        for shape, dtype in (((99999, 4, 256), np.float32), ((100000, 4, 256), np.float64)):
            with self.subTest(shape=shape, dtype=dtype):
                self.load.return_value = SimpleNamespace(series=SimpleNamespace(shape=shape, dtype=dtype))
                with self.assertRaisesRegex(AssertionError, "Unexpected dataset size"):
                    self.run_main()
                self.assertFalse(self.out.exists())

    def test_out_of_range_labels_are_rejected(self):
        for label in (-1, 8):
            with self.subTest(label=label):
                self.dataset.labels[0] = label
                with self.assertRaisesRegex(AssertionError, "Unexpected dataset classes"):
                    self.run_main()
                self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
