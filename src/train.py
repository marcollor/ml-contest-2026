"""Train adaptative_model with the fixed 80-epoch recipe."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
from importlib.metadata import version
from pathlib import Path
from platform import python_version
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from augment_data import augment_raw, inverse_frequency_weights
from cli import HelpArgumentParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from harness import class_geomean_score
from model import TRAINED_CHECKPOINT
from src.architecture import TemporalCNN
from normalize_data import PREPROCESSING, preprocess_series

EPOCHS = 80
BATCH_SIZE = 256
SEED = 20261114
STRESS_SEED = 20261113
CLEAN_FRACTION = 0.25
OPTIMIZER = {"lr": 0.002, "weight_decay": 0.001}
MIN_LR = 5e-5
AUGMENTATION = dict(missing_max=0.06, noise_max=0.6, spike_probability=0.002,
                    spike_scale=10.0, gain_min=0.6, gain_max=1.6)
STRESS = {**AUGMENTATION, "missing_max": 0.03, "noise_max": 0.3, "spike_probability": 0.001}


def normalized(raw):
    return torch.from_numpy(np.concatenate([preprocess_series(sample) for sample in raw]))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def evaluate(net, inputs, labels, device):
    net.eval()
    pred = torch.cat([net(batch.to(device)).argmax(1).cpu()
                      for batch in inputs.split(BATCH_SIZE)]).numpy()
    confusion = np.zeros((8, 10), np.int64)
    np.add.at(confusion, (labels, pred), 1)
    counts = confusion.sum(1)
    recall = np.divide(confusion[:, :8].diagonal(), counts, out=np.zeros(8), where=counts > 0)
    score = class_geomean_score(SimpleNamespace(n_classes=8, confusion=confusion, wrong_penalty=0.33))
    return float((pred == labels).mean()), float(score), recall.tolist()


def main(argv=None):
    parser = HelpArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check data and model setup on a small batch.")
    args = parser.parse_args(argv)

    data_path = ROOT / "data/train_100k.npz"
    split_path = ROOT / "data/split_indices.npz"
    if not data_path.is_file() or not split_path.is_file():
        parser.error("Missing training data or splits. Run python src/prepare_data.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = load_dataset(data_path)
    with np.load(split_path) as splits:
        expected_sizes = {"train": 70000, "dev": 5000, "iid": 20000, "reserve": 5000}
        indices = [splits[name] for name in expected_sizes]
        combined = np.concatenate(indices)
        if (any(len(splits[name]) != size for name, size in expected_sizes.items())
                or not np.array_equal(np.sort(combined), np.arange(len(data)))):
            raise ValueError("Invalid data split. Run python src/prepare_data.py to rebuild it.")
        train_ids, dev_ids = splits["train"], splits["dev"]
    if args.check_only:
        train_ids, dev_ids = train_ids[:16], dev_ids[:16]

    raw_train, raw_dev = data.series[train_ids], data.series[dev_ids]
    train_inputs = normalized(raw_train)
    dev_inputs = normalized(raw_dev)
    labels = torch.from_numpy(data.labels[train_ids].astype(np.int64))
    dev_labels = data.labels[dev_ids]

    stress_raw = augment_raw(raw_dev, np.random.default_rng(STRESS_SEED), **STRESS)
    for sample in stress_raw:
        for channel in sample:
            finite = np.isfinite(channel)
            if finite.any():
                channel[finite] -= channel[finite].mean()
    stress_inputs = normalized(stress_raw)

    net = TemporalCNN().to(device)
    if args.check_only:
        net.eval()
        with torch.inference_mode():
            for inputs in (train_inputs, dev_inputs, stress_inputs):
                logits = net(inputs.to(device))
                if logits.shape != (16, 8) or not torch.isfinite(logits).all():
                    raise ValueError("Model returned invalid logits during setup.")
        print(f"Training setup verified on {device}.")
        return

    print(f"Training adaptative_model for {EPOCHS} epochs on {device}.", flush=True)
    weights = torch.from_numpy(inverse_frequency_weights(labels.numpy())).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), **OPTIMIZER)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    order_rng = torch.Generator().manual_seed(SEED + 1000)
    aug_rng = np.random.default_rng(SEED + 2000)

    source = Path(__file__).resolve().parent
    inputs = {
        "data": data_path, "split": split_path, "architecture": source / "architecture.py",
        "train": source / "train.py", "augmentation": source / "augment_data.py",
        "normalization": source / "normalize_data.py", "score": source.parent / "harness.py",
        "data_loader": source.parent / "competition_api.py",
    }
    fingerprints = {name: sha256(path) for name, path in inputs.items()}
    runtime = dict(python=python_version(), torch=str(torch.__version__), numpy=np.__version__,
                   numba=version("numba"), device=device, threads=torch.get_num_threads(),
                   deterministic_algorithms=True)
    if device == "cuda":
        runtime.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(),
                       cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"])

    history = []
    best_key = None

    for epoch in range(1, EPOCHS + 1):
        net.train()
        losses = []
        for ids in torch.randperm(len(labels), generator=order_rng).split(BATCH_SIZE):
            batch = train_inputs[ids]
            corrupt = aug_rng.random(len(ids)) >= CLEAN_FRACTION
            if corrupt.any():
                raw_batch = augment_raw(raw_train[ids.numpy()[corrupt]], aug_rng, **AUGMENTATION)
                batch[corrupt] = normalized(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(batch.to(device)), labels[ids].to(device),
                                                     weight=weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        dev_accuracy, dev_score, dev_recall = evaluate(net, dev_inputs, dev_labels, device)
        stress_accuracy, stress_score, stress_recall = evaluate(net, stress_inputs, dev_labels, device)
        key = (min(dev_score, stress_score), min(dev_accuracy, stress_accuracy))
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            best_recall = {"dev": dev_recall, "stress": stress_recall}
        mean_loss = float(np.mean(losses))
        history.append(dict(epoch=epoch, loss=mean_loss, dev_accuracy=dev_accuracy, dev_score=dev_score,
                            stress_accuracy=stress_accuracy, stress_score=stress_score))
        scheduler.step()
        print(f"{epoch:02d}/{EPOCHS}  loss={mean_loss:.4f}  dev={dev_accuracy:.2%}  "
              f"stress={stress_accuracy:.2%}  score={key[0]:.3f}", flush=True)

    # Publish the complete candidate as one file so inference can load it safely.
    TRAINED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TRAINED_CHECKPOINT.parent) as temporary:
        pending = Path(temporary) / "checkpoint.pt"
        torch.save(dict(
            state_dict=best_state, preprocessing=PREPROCESSING, best_epoch=best_epoch,
            history=history, best_recall=best_recall,
            recipe=dict(epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, stress_seed=STRESS_SEED,
                        optimizer={"name": "AdamW", **OPTIMIZER},
                        scheduler={"name": "CosineAnnealingLR", "eta_min": MIN_LR},
                        augmentation={**AUGMENTATION, "clean_fraction": CLEAN_FRACTION},
                        stress={**STRESS, "center_channels": True},
                        selection="max_worst_score_then_accuracy", class_weights=weights.cpu().tolist()),
            sha256=fingerprints, runtime=runtime,
        ), pending)
        pending.replace(TRAINED_CHECKPOINT)
    print(f"Saved epoch {best_epoch} to {TRAINED_CHECKPOINT}")


if __name__ == "__main__":
    main()
