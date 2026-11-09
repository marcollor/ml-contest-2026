"""Competition entry point for adaptative_model."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TRAINED_CHECKPOINT = MODEL_DIR / "retrained.pt"

sys.path[:0] = [str(MODEL_DIR), str(ROOT / "src")]
from architecture import TemporalCNN
from normalize_data import PREPROCESSING, preprocess_series


class Model:
    """Predict a class from one raw (4, 256) sample."""

    def __init__(self, trained=False):
        torch.set_num_threads(1)
        if trained:
            checkpoint = torch.load(TRAINED_CHECKPOINT, map_location="cpu", weights_only=True)
            if checkpoint["preprocessing"] != PREPROCESSING:
                raise ValueError("Checkpoint preprocessing does not match the V024 recipe.")
            weights = checkpoint["state_dict"]
        else:
            weights = torch.load(MODEL_DIR / "weights.pt", map_location="cpu", weights_only=True)

        self.net = TemporalCNN().cpu().eval()
        self.net.load_state_dict(weights)
        # Compile preprocessing for the harness's read-only inputs during setup.
        dummy = np.zeros((4, 256), dtype=np.float32)
        dummy.setflags(write=False)
        preprocess_series(dummy)

    @torch.inference_mode()
    def predict(self, sample):
        inputs = torch.from_numpy(preprocess_series(sample.series))
        return int(self.net(inputs).argmax(dim=1).item())
