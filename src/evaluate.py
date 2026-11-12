"""Evaluate TemporalCNN on the local demo dataset."""
from pathlib import Path
import sys

from cli import HelpArgumentParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = HelpArgumentParser(description=__doc__)
    parser.add_argument("--trained", action="store_true", help="Evaluate inference artifacts exported from model/retrained.pt.")
    args = parser.parse_args(argv)

    demo_path = ROOT / "data/test_demo.npz"
    if not demo_path.is_file():
        parser.error("Missing data/test_demo.npz. See data/README.md.")
    from competition_api import load_dataset
    from model import Model, TRAINED_CHECKPOINT

    if args.trained and not TRAINED_CHECKPOINT.is_file():
        parser.error("No trained checkpoint found. Run python src/train.py first.")

    demo = load_dataset(demo_path)
    model = Model(trained=args.trained)
    correct = sum(model.predict(sample) == label for sample, label in zip(demo, demo.labels))
    print(f"Demo accuracy: {correct / len(demo):.2%} ({correct}/{len(demo)})")


if __name__ == "__main__":
    main()
