"""Export the competition TemporalCNN to ONNX (export environment only)."""

import argparse

import torch

from model import MODEL_DIR, TRAINED_CHECKPOINT
from src.architecture import TemporalCNN
from src.normalize_data import PREPROCESSING


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", action="store_true", help="Export model/retrained.pt")
    args = parser.parse_args()

    checkpoint_path = TRAINED_CHECKPOINT if args.trained else MODEL_DIR / "weights.pt"
    output_path = MODEL_DIR / ("retrained.onnx" if args.trained else "model.onnx")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        if checkpoint.get("preprocessing") != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing does not match the V024 recipe.")
        weights = checkpoint["state_dict"]
    else:
        weights = checkpoint

    net = TemporalCNN().cpu()
    net.load_state_dict(weights)
    net.eval()
    with torch.inference_mode():
        torch.onnx.export(
            net,
            torch.zeros((1, 4, 256), dtype=torch.float32),
            str(output_path),
            input_names=["series"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"series": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    print(f"Exported {checkpoint_path.name} to {output_path}")


if __name__ == "__main__":
    main()
