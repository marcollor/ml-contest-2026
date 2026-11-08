"""Verify registry ranking uses the stored checkpoint, not peak history metrics."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_registry import file_sha256, get_best_model, load_best_model, refresh_registry


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model(self, name, metrics, best_epoch=1):
        directory = self.root / name
        directory.mkdir()
        (directory / "architecture.py").write_text("# Test architecture\n")
        (directory / "weights.pt").write_bytes(b"test weights")
        (directory / "training.json").write_text(json.dumps({
            "best_epoch": best_epoch, "parameters": 123, "epochs": len(metrics),
            "arguments": {"criterion": "accuracy"},
        }))
        (directory / "history.json").write_text(json.dumps([
            dict(epoch=i + 1, dev_score=score, dev_accuracy=accuracy)
            for i, (score, accuracy) in enumerate(metrics)
        ]))
        return directory

    def test_ranking_uses_saved_epoch_and_selected_criterion(self):
        first = self.model("first", [(10, 0.8), (50, 0.7)], best_epoch=1)
        second = self.model("second", [(9, 0.9)])
        self.assertEqual(get_best_model(self.root, "score"), first)
        self.assertEqual(get_best_model(self.root, "accuracy"), second)
        registry = json.loads((self.root / "registry.json").read_text())
        self.assertEqual(registry["best_models"], {"score": "first", "accuracy": "second"})
        self.assertEqual(registry["models"][0]["parameters"], 123)
        self.assertEqual(registry["models"][0]["checkpoint_metrics"]["dev_score"], 10)

    def test_pending_missing_and_malformed_models_are_excluded(self):
        (self.root / "pending").mkdir()
        untrained = self.root / "untrained"
        untrained.mkdir()
        (untrained / "architecture.py").write_text("# Not trained\n")
        broken = self.model("broken", [(10, 0.9)])
        (broken / "history.json").write_text("invalid json")
        registry = refresh_registry(self.root)
        self.assertEqual({entry["status"] for entry in registry["models"]},
                         {"pending", "untrained", "incomplete"})
        self.assertIsNone(registry["best_models"]["score"])
        with self.assertRaisesRegex(ValueError, "No trained models"):
            get_best_model(self.root)

    def test_refresh_excludes_deleted_checkpoint(self):
        first = self.model("first", [(10, 0.8)])
        second = self.model("second", [(9, 0.9)])
        self.assertEqual(get_best_model(self.root), first)
        (first / "weights.pt").unlink()
        self.assertEqual(get_best_model(self.root), second)

    def test_changed_artifact_is_not_ranked_with_old_metrics(self):
        directory = self.model("changed", [(10, 0.8)])
        summary_path = directory / "training.json"
        summary = json.loads(summary_path.read_text())
        summary["weights_sha256"] = file_sha256(directory / "weights.pt")
        summary_path.write_text(json.dumps(summary))
        (directory / "weights.pt").write_bytes(b"different weights")
        registry = refresh_registry(self.root)
        self.assertEqual(registry["models"][0]["status"], "incomplete")
        self.assertIn("no longer matches", registry["models"][0]["error"])

    def test_ties_and_nonfinite_metrics(self):
        self.model("z", [(10, 0.8)])
        better = self.model("a", [(10, 0.9)])
        self.model("nan", [(float("nan"), 1.0)])
        self.assertEqual(get_best_model(self.root), better)
        with self.assertRaisesRegex(ValueError, "criterion"):
            get_best_model(self.root, "loss")

    def test_load_best_model_loads_weights_and_sets_eval(self):
        import torch
        directory = self.model("network", [(10, 0.8)])
        (directory / "architecture.py").write_text(
            "from torch import nn\n"
            "class TemporalCNN(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(4, 8)\n"
            "    def forward(self, x):\n"
            "        return self.fc(x.mean(dim=-1))\n"
        )
        weights = {"fc.weight": torch.ones(8, 4), "fc.bias": torch.zeros(8)}
        torch.save(weights, directory / "weights.pt")
        net = load_best_model(self.root)
        self.assertFalse(net.training)
        torch.testing.assert_close(net(torch.ones(1, 4, 256)), torch.full((1, 8), 4.0))


if __name__ == "__main__":
    unittest.main()
