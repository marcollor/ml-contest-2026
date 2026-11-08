"""Train the temporal CNN and select a checkpoint using the development split.

This didactic recipe does not reproduce the historical directed checkpoint
selector. Run with ``python src/train.py --model-dir models/temporal_cnn``. Use
``--check-only`` to validate the data splits, normalization, and forward pass
on 16 training and development samples without training or writing outputs.
Paths supplied through the CLI are resolved relative to ``--repo``.
``--model-dir`` is required and must contain architecture.py defining
TemporalCNN(); weights.pt, history.json, and training.json are saved there.
``--out`` remains an alias for ``--model-dir``.
"""

import os

# Configure numerical backends before importing NumPy, Numba, or PyTorch.
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from cli import HelpArgumentParser
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from normalize_data import fast_normalize_series_numba
from model_registry import file_sha256, refresh_registry


def normalized(raw: np.ndarray) -> np.ndarray:
    """Normalize raw samples from (N, 4, 256) to a float32 array of that shape.

    Each sample is imputed with channel means, standardized globally, and clipped by the
    same normalization routine used for single-sample preprocessing.
    """
    return np.concatenate([fast_normalize_series_numba(sample) for sample in raw])


def load_model(model_dir: Path) -> torch.nn.Module:
    """Instantiate TemporalCNN() from the selected directory's architecture.py.

    The architecture must accept (N, 4, 256) inputs and return (N, 8) logits.
    Its directory is available for imports of sibling helper modules.
    """
    architecture_path = model_dir / "architecture.py"
    if not architecture_path.is_file():
        raise FileNotFoundError(f"Architecture file not found: {architecture_path}")
    spec = importlib.util.spec_from_file_location("_training_architecture", architecture_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load architecture: {architecture_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(model_dir))
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, "TemporalCNN", None)
        if not callable(factory):
            raise TypeError("architecture.py must define a callable TemporalCNN()")
        net = factory()
        if not isinstance(net, torch.nn.Module):
            raise TypeError("TemporalCNN() must return a torch.nn.Module")
        return net
    finally:
        sys.path.pop(0)


class TrainingHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Preserve usage examples and display defaults in the training help."""


def build_parser() -> HelpArgumentParser:
    """Describe the required model selection and optional training controls."""
    parser = HelpArgumentParser(
        description="Train an architecture and save its best development checkpoint.",
        formatter_class=TrainingHelpFormatter,
        epilog="""Examples:
  python src/train.py --model-dir models/temporal_cnn --check-only
  python src/train.py --model-dir models/temporal_cnn --epochs 24 --threads 4
  python src/train.py --model-dir models/temporal_cnn --device cuda --replace

All relative paths use --repo as their base. The selected model directory must
contain architecture.py defining TemporalCNN(). Outputs (weights.pt,
history.json, training.json) are written to that same directory; its parent
registry.json is refreshed after training. No arguments displays this help.
--check-only checks 16 samples per split without training or writing outputs.
""",
    )
    paths = parser.add_argument_group("Model selection and input paths")
    paths.add_argument(
        "--model-dir", "--out", dest="model_dir", required=True, default=argparse.SUPPRESS, metavar="DIR",
        help="Required: architecture directory and destination for training outputs. --out is an alias.",
    )
    paths.add_argument(
        "--repo", default=str(Path(__file__).resolve().parents[1]), metavar="DIR",
        help="Base directory for all relative paths; defaults to this script's repository.",
    )
    paths.add_argument(
        "--data", default="data/train_100k.npz", metavar="FILE",
        help="Existing labeled dataset (.npz or .parquet).",
    )
    paths.add_argument(
        "--split", default="data/split_indices.npz", metavar="FILE",
        help="Existing NPZ with train/dev/iid/reserve indices; generate it using src/prepare_data.py.",
    )
    training = parser.add_argument_group("Training settings (used when training)")
    training.add_argument("--epochs", type=int, default=24, metavar="N", help="Maximum epochs; must be positive.")
    training.add_argument(
        "--criterion", choices=["score", "accuracy"], default="score",
        help="Primary checkpoint metric; the other metric breaks ties.",
    )
    runtime = parser.add_argument_group("Runtime and reproducibility")
    runtime.add_argument("--seed", type=int, default=20261112, metavar="N", help="Random seed: 0 through 4294967295.")
    runtime.add_argument("--threads", type=int, default=2, metavar="N", help="PyTorch CPU threads; must be positive.")
    runtime.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device; CUDA must be available if selected.")
    execution = parser.add_argument_group("Execution mode")
    modes = execution.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Validate a small batch without training or writing outputs.")
    modes.add_argument("--replace", action="store_true", help="Train and allow overwriting existing weights and logs.")
    return parser


def parse_arguments(argv=None):
    """Validate arguments and paths before loading any datasets or architectures."""
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        parser.exit(0)
    args = parser.parse_args(arguments)
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive integers")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        parser.error(f"--repo is not a directory: {repo}")
    args.repo = str(repo)
    for name in ("model_dir", "data", "split"):
        setattr(args, name, str((repo / getattr(args, name)).resolve()))
    architecture = Path(args.model_dir) / "architecture.py"
    if not architecture.is_file():
        parser.error(f"--model-dir must contain architecture.py: {architecture}")
    for name in ("data", "split"):
        if not Path(getattr(args, name)).is_file():
            parser.error(f"--{name} file not found: {getattr(args, name)}")
    if not args.check_only and not args.replace and (Path(args.model_dir) / "weights.pt").exists():
        parser.error("Output already contains weights: use --replace or choose another --model-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable: use --device cpu")
    return args


def main() -> None:
    """Validate the fixed split, train with augmentation, and save the best model."""
    # 1.- Parse and validate command-line settings before loading input data.
    args = parse_arguments()

    print("Training configuration:", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    # 2.- Resolve the repository root and load shared competition utilities.
    repo = Path(args.repo).resolve()
    model_dir = (repo / args.model_dir).resolve()
    # Allow shared competition modules to be imported when running this script.
    sys.path.insert(0, str(repo))
    from competition_api import load_dataset
    from harness import class_geomean_score

    # 3.- Configure reproducibility, CPU threads, and the compute device.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 4.- Load the dataset and validate the fixed data partitions.
    ds = load_dataset(repo / args.data)
    # Validate the complete fixed split before reducing it for a preflight check.
    with np.load(repo / args.split) as indices:
        partitions = [
            set(indices[key].tolist()) for key in ("train", "dev", "iid", "reserve")
        ]
        assert sum(map(len, partitions)) == len(set.union(*partitions)) == len(ds)
        assert len(indices["train"]) == 70000 and len(indices["dev"]) == 5000
        train_ids, dev_ids = indices["train"], indices["dev"]
    if args.check_only:
        train_ids, dev_ids = train_ids[:16], dev_ids[:16]

    # 5.- Preprocess signals and convert inputs and labels to training arrays.
    x = torch.from_numpy(normalized(ds.series[train_ids]))
    v = torch.from_numpy(normalized(ds.series[dev_ids]))
    y = torch.from_numpy(ds.labels[train_ids].astype(np.int64))
    vy = np.asarray(ds.labels[dev_ids], np.int64)

    # 6.- Create the network and optionally run a forward-only preflight check.
    net = load_model(model_dir).to(args.device)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    if args.check_only:
        net.eval()
        with torch.inference_mode():
            result = net(v.to(args.device))
        assert result.shape == (16, 8) and torch.isfinite(result).all()
        print("Preflight passed: splits, normalization, and forward pass; no training.")
        return

    # 7.- Prepare the output directory and protect existing model weights.
    out = model_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "weights.pt").exists() and not args.replace:
        raise FileExistsError(
            "Output already contains weights: use --replace or choose another --model-dir"
        )

    # 8.- Initialize the optimizer, random generators, and training history.
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.001)
    # Keep shuffling and augmentation on independent, reproducible random streams.
    order_rng = torch.Generator().manual_seed(args.seed + 1000)
    aug_rng = torch.Generator().manual_seed(args.seed + 2000)
    history, best, bad = [], None, 0
    start = time.perf_counter()

    # 9.- Train each epoch with shuffled batches and augmented signals.
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for ids in torch.randperm(len(x), generator=order_rng).split(256):
            batch = x[ids]
            # Apply a per-channel gain and additive Gaussian noise on the CPU.
            gain = 0.8 + 0.4 * torch.rand((len(ids), 4, 1), generator=aug_rng)
            noise = 0.05 * torch.randn(batch.shape, generator=aug_rng)
            batch = (batch * gain + noise).to(args.device)

            # 10.- Compute the loss, backpropagate, and update model parameters.
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(batch), y[ids].to(args.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 11.- Evaluate development samples without gradients or augmentation.
        net.eval()
        with torch.inference_mode():
            logits = torch.cat(
                [net(batch.to(args.device)).cpu() for batch in v.split(256)]
            ).numpy()
        pred = logits.argmax(1)
        # The harness expects 8 class columns plus abstention and invalid columns.
        cm = np.zeros((8, 10), np.int64)
        np.add.at(cm, (vy, pred), 1)
        score = class_geomean_score(
            SimpleNamespace(n_classes=8, confusion=cm, wrong_penalty=0.33)
        )
        acc = float((pred == vy).mean())
        key = (score, acc) if args.criterion == "score" else (acc, score)
        record = dict(
            epoch=epoch + 1,
            dev_accuracy=acc,
            dev_score=score,
            loss=float(np.mean(losses)),
            elapsed_s=time.perf_counter() - start,
        )
        history.append(record)
        # 12.- Save improvements, log metrics, and apply early stopping.
        if best is None or key > best:
            best, bad, best_epoch = key, 0, epoch + 1
            torch.save(
                {key: value.detach().cpu() for key, value in net.state_dict().items()},
                out / "weights.pt",
            )
        else:
            bad += 1
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(record, flush=True)
        # Train at least eight epochs, then stop after four without improvement.
        if epoch + 1 >= 8 and bad >= 10:
            break

    # 13.- Write the final training configuration and summary.
    (out / "training.json").write_text(
        json.dumps(
            dict(
                recipe="didactic; historical directed checkpoint selector not reproduced",
                arguments=vars(args),
                train_n=len(x),
                dev_n=len(v),
                epochs=len(history),
                best_epoch=best_epoch,
                parameters=parameter_count,
                architecture_sha256=file_sha256(model_dir / "architecture.py"),
                weights_sha256=file_sha256(out / "weights.pt"),
                preprocessing="channel_mean_imputation; sample_global_zscore; clip_10",
                training_s=time.perf_counter() - start,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    # 14.- Refresh the model registry with the completed training artifacts.
    refresh_registry(model_dir.parent)
    print("Model registry updated:", model_dir.parent / "registry.json")
    print("Checkpoint selected using development data only:", out / "weights.pt")


if __name__ == "__main__":
    main()
