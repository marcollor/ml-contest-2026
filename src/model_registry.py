"""Index model artifacts and select the best saved development checkpoint.

Run ``python src/model_registry.py refresh`` to rebuild models/registry.json or
``python src/model_registry.py best --criterion accuracy`` to print a model path.
Registry inspection reads metadata only; it never imports architecture code.
"""

from cli import HelpArgumentParser
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def file_sha256(path: Path) -> str:
    """Fingerprint an artifact without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _inspect_model(directory: Path) -> dict:
    """Describe a folder and metrics from its saved epoch, never another epoch."""
    entry = {
        "name": directory.name,
        "directory": directory.name,
        "status": "pending",
        "architecture": None,
        "weights": None,
        "parameters": None,
        "checkpoint_metrics": None,
    }
    for filename, key in (("architecture.py", "architecture"), ("weights.pt", "weights")):
        path = directory / filename
        if path.is_file():
            entry[key] = {
                "path": f"{directory.name}/{filename}",
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    if entry["architecture"] is None:
        return entry
    entry["status"] = "untrained" if entry["weights"] is None else "incomplete"
    if entry["weights"] is None:
        return entry
    try:
        summary = json.loads((directory / "training.json").read_text(encoding="utf-8-sig"))
        history = json.loads((directory / "history.json").read_text(encoding="utf-8-sig"))
        if not isinstance(summary, dict) or not isinstance(history, list):
            raise ValueError("Expected a training object and a history list")
        best_epoch = summary["best_epoch"]
        matching = [row for row in history if row.get("epoch") == best_epoch]
        if len(matching) != 1:
            raise ValueError("Saved best_epoch must match exactly one history entry")
        checkpoint = matching[0]
        if not all(_finite_number(checkpoint.get(key)) for key in ("dev_score", "dev_accuracy")):
            raise ValueError("Saved checkpoint metrics must be finite numbers")
        # New training runs fingerprint the actual saved artifacts. Legacy runs
        # remain readable even though their summaries do not contain hashes.
        for field, artifact in (("architecture_sha256", "architecture"), ("weights_sha256", "weights")):
            if field in summary and summary[field] != entry[artifact]["sha256"]:
                raise ValueError(f"Artifact no longer matches its training summary: {artifact}")
        entry.update(
            status="trained",
            parameters=summary.get("parameters"),
            best_epoch=best_epoch,
            epochs=summary.get("epochs"),
            train_n=summary.get("train_n"),
            dev_n=summary.get("dev_n"),
            training_s=summary.get("training_s"),
            arguments=summary.get("arguments", {}),
            preprocessing=summary.get("preprocessing"),
            input_shape=[4, 256],
            n_classes=8,
            checkpoint_metrics={key: checkpoint[key] for key in ("dev_score", "dev_accuracy")},
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        entry["error"] = str(error)
    return entry


def _select(entries: list, criterion: str) -> dict:
    if criterion not in ("score", "accuracy"):
        raise ValueError("criterion must be 'score' or 'accuracy'")
    candidates = [entry for entry in entries if entry["status"] == "trained"]
    if not candidates:
        raise ValueError("No trained models with valid saved checkpoint metrics are available")
    primary, secondary = (
        ("dev_score", "dev_accuracy") if criterion == "score" else ("dev_accuracy", "dev_score")
    )
    # Entries are sorted by name, so exact ties have a stable result.
    return max(candidates, key=lambda entry: (
        entry["checkpoint_metrics"][primary], entry["checkpoint_metrics"][secondary]
    ))


def refresh_registry(models_dir: Path = Path("models")) -> dict:
    """Rebuild registry.json from the current immediate model subdirectories.

    Missing/incomplete artifacts are listed but excluded from selection. Paths
    inside the registry are relative to models_dir. Writing is atomic so readers
    never observe a partially written JSON file.
    """
    models_dir = Path(models_dir).resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    entries = [
        _inspect_model(directory)
        for directory in sorted(models_dir.iterdir())
        if directory.is_dir() and not directory.name.startswith((".", "__"))
    ]
    best = {}
    for criterion in ("score", "accuracy"):
        try:
            best[criterion] = _select(entries, criterion)["name"]
        except ValueError:
            best[criterion] = None
    registry = {"schema_version": 1, "models": entries, "best_models": best}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=models_dir, prefix=".registry-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(registry, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, models_dir / "registry.json")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return registry


def get_best_model(models_dir: Path = Path("models"), criterion: str = "score") -> Path:
    """Refresh the index and return the best stored model's absolute directory."""
    registry = refresh_registry(models_dir)
    selected = _select(registry["models"], criterion)
    return Path(models_dir).resolve() / selected["directory"]


def load_best_model(models_dir: Path = Path("models"), criterion: str = "score", device: str = "cpu"):
    """Return the winning network with saved weights, ready for inference.

    Inputs must use the same preprocessing as during that model's training.
    """
    import torch
    from train import load_model

    directory = get_best_model(models_dir, criterion)
    net = load_model(directory).to(device)
    net.load_state_dict(torch.load(directory / "weights.pt", map_location=device, weights_only=True))
    net.eval()
    return net


def main() -> None:
    """Refresh the model index or print the best saved model's directory."""
    parser = HelpArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh", "best"],
                        help="refresh: rebuild the registry; best: print the best model directory.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Directory containing the model subdirectories and registry.json.")
    parser.add_argument("--criterion", choices=["score", "accuracy"], default="score",
                        help="Metric used to choose the best saved checkpoint.")
    args = parser.parse_args()
    try:
        if args.command == "refresh":
            registry = refresh_registry(args.models_dir)
            print(f"Registered {len(registry['models'])} model directories in {args.models_dir / 'registry.json'}")
        else:
            print(get_best_model(args.models_dir, args.criterion))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
