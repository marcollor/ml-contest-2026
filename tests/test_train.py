"""Check architecture selection and colocated training outputs."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("training_script", ROOT / "src/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class SmallSelection:
    """Provide a small batch while keeping the production split checks active."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, indices):
        return self.values


class FixtureDataset:
    series = SmallSelection(np.zeros((16, 4, 256), np.float32))
    labels = SmallSelection(np.zeros(16, np.int64))

    def __len__(self):
        return 100000


class TrainDirectoryTests(unittest.TestCase):
    def test_selected_architecture_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "data/train_100k.npz").touch()
            split = root / "split.npz"
            np.savez(
                split,
                train=np.arange(70000),
                dev=np.arange(70000, 75000),
                iid=np.arange(75000, 95000),
                reserve=np.arange(95000, 100000),
            )
            for flag, width in (("--model-dir", 4), ("--out", 6)):
                with self.subTest(flag=flag):
                    directory = root / f"network_{width}"
                    directory.mkdir()
                    (directory / "architecture.py").write_text(
                        "from torch import nn\n"
                        "class TemporalCNN(nn.Module):\n"
                        "    def __init__(self):\n"
                        "        super().__init__()\n"
                        f"        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(1024, {width}), nn.ReLU(), nn.Linear({width}, 8))\n"
                        "    def forward(self, x):\n"
                        "        return self.layers(x)\n",
                        encoding="utf-8",
                    )
                    argv = ["train.py", "--repo", str(root), flag, directory.name,
                            "--split", str(split), "--epochs", "1"]
                    with patch.object(sys, "argv", argv), \
                         patch("competition_api.load_dataset", return_value=FixtureDataset()), \
                         patch.object(train, "normalized", side_effect=lambda raw: raw.copy()), \
                         patch.object(torch, "set_num_interop_threads"), \
                         contextlib.redirect_stdout(io.StringIO()):
                        train.main()
                    self.assertTrue((directory / "weights.pt").is_file())
                    self.assertTrue((directory / "history.json").is_file())
                    registry = json.loads((root / "registry.json").read_text())
                    entry = next(row for row in registry["models"] if row["name"] == directory.name)
                    self.assertEqual(entry["status"], "trained")
                    summary = json.loads((directory / "training.json").read_text())
                    self.assertEqual(summary["parameters"], 1025 * width + 8 * width + 8)
                    weights = torch.load(directory / "weights.pt", weights_only=True)
                    self.assertEqual(tuple(weights["layers.1.weight"].shape), (width, 1024))
            self.assertFalse((root / "weights.pt").exists())

    def test_missing_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Architecture file not found"):
                train.load_model(Path(temporary))

    def test_missing_model_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "architecture.py").write_text("VALUE = 1\n")
            with self.assertRaisesRegex(TypeError, "TemporalCNN"):
                train.load_model(directory)


class TrainingArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "network"
        self.directory.mkdir()
        (self.directory / "architecture.py").write_text("# Parsing must not execute this file\n")
        (self.root / "data").mkdir()
        (self.root / "data/train_100k.npz").touch()
        (self.root / "data/split_indices.npz").touch()
        self.base = ["--repo", str(self.root), "--model-dir", "network"]

    def assert_invalid(self, arguments, message):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as result:
            train.parse_arguments(arguments)
        self.assertEqual(result.exception.code, 2)
        self.assertIn(message, errors.getvalue())
        self.assertIn("Examples:", errors.getvalue())

    def test_model_is_required_and_no_arguments_prints_help(self):
        self.assert_invalid(["--epochs", "1"], "--model-dir")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            train.parse_arguments([])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("Model selection and input paths", output.getvalue())
        self.assertIn("Execution mode", output.getvalue())

    def test_defaults_and_paths(self):
        args = train.parse_arguments(self.base)
        self.assertEqual(args.epochs, 24)
        self.assertEqual(args.patience, 10)
        self.assertEqual(train.parse_arguments(self.base + ["--patience", "0"]).patience, 0)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.data, str(self.root / "data/train_100k.npz"))
        self.assertEqual(args.model_dir, str(self.directory))
        args = train.parse_arguments(["--repo", str(self.root), "--out", "network"])
        self.assertEqual(args.model_dir, str(self.directory))
        parser_defaults = train.build_parser().parse_args(["--model-dir", "models/example"])
        self.assertEqual(Path(parser_defaults.repo), ROOT)

    def test_invalid_values_and_incompatible_modes(self):
        for arguments, message in (
            (["--epochs", "0"], "positive"),
            (["--threads", "-1"], "positive"),
            (["--patience", "-1"], "nonnegative"),
            (["--seed", "-1"], "--seed"),
            (["--seed", str(2**32)], "--seed"),
            (["--check-only", "--replace"], "not allowed"),
            (["--criterion", "loss"], "invalid choice"),
            (["--unknown"], "unrecognized arguments"),
            (["--thr", "4"], "unrecognized arguments"),
            (["unknown-command"], "unrecognized arguments"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_invalid(self.base + arguments, message)

    def test_missing_paths_and_cuda_show_help(self):
        self.assert_invalid(self.base + ["--data", "missing.npz"], "--data file not found")
        self.assert_invalid(self.base + ["--split", "missing.npz"], "--split file not found")
        self.assert_invalid(["--repo", str(self.root), "--model-dir", "missing"], "architecture.py")
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assert_invalid(self.base + ["--device", "cuda"], "CUDA is unavailable")

    def test_existing_weights_rejected_before_dataset_loading(self):
        weights = self.directory / "weights.pt"
        weights.write_bytes(b"original weights")
        self.assert_invalid(self.base, "use --replace")
        self.assertTrue(train.parse_arguments(self.base + ["--replace"]).replace)
        self.assertTrue(train.parse_arguments(self.base + ["--check-only"]).check_only)
        with patch.object(sys, "argv", ["train.py"] + self.base), \
             patch("competition_api.load_dataset") as load_dataset, \
             patch.object(train, "load_model") as load_model, \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit):
            train.main()
        load_dataset.assert_not_called()
        load_model.assert_not_called()
        self.assertEqual(weights.read_bytes(), b"original weights")


if __name__ == "__main__":
    unittest.main()
