# Local training workflow

Shared training and preprocessing code lives in `src/`. Each model directory
contains its own `architecture.py` and the outputs of its training run:

```text
src/
    train.py
    prepare_data.py
    normalize_data.py
    model_registry.py
models/
    registry.json
    temporal_cnn/
        architecture.py
        weights.pt
        history.json
        training.json
    model_2/           # Reserved; model files not yet available.
    model_3/           # Reserved; model files not yet available.
data/
    train_100k.npz
    split_indices.npz
    val.npz
```

The existing trained model has been moved to `models/temporal_cnn/` without
changing its architecture, weights, or JSON records. Arguments stored in an
existing `training.json` describe the original run and retain its original paths.
Only one custom trained model is currently present in this checkout. The
`model_2/` and `model_3/` directories are reserved for the other two models;
they cannot be trained until their architecture.py files are supplied.

Run commands from the repository root:

```bash
python src/prepare_data.py
python src/train.py --model-dir models/temporal_cnn --check-only
python src/train.py --model-dir models/temporal_cnn --replace
```

For another model, place an `architecture.py` defining `TemporalCNN()` in its
own subdirectory of `models/` and pass that directory via `--model-dir`:

```bash
python src/train.py --model-dir models/another_model
```

The model must accept `(N, 4, 256)` inputs and return `(N, 8)` logits. Training
writes `weights.pt`, `history.json`, and `training.json` to the selected model
directory. Existing weights require `--replace`. `--out` remains an alias for
`--model-dir`. Relative CLI paths are resolved against `--repo`.

The original competition interface (`competition_api.py`), evaluator
(`harness.py`), and baseline examples remain in their original locations.

## Model registry and best-model selection

`models/registry.json` indexes every immediate model subdirectory, including the
reserved folders. It records architecture and weights paths, sizes and SHA-256
fingerprints, parameter count, training settings, sample counts, elapsed training
time, and metrics from the **saved checkpoint's epoch**. Pending or incomplete
models are listed but never selected. Legacy training runs may have no recorded
preprocessing description or artifact fingerprints in their training summary.

Training refreshes the registry automatically when it completes. To refresh it
manually after adding, moving, or removing models:

```bash
python src/model_registry.py refresh
```

Get the absolute directory of the best saved model (score is the default):

```bash
python src/model_registry.py best
python src/model_registry.py best --criterion accuracy
```

Ranking maximizes the selected metric, using the other metric to break ties.
Exact ties use alphabetical model order. A query refreshes the registry first,
so removed weights are excluded. It fails clearly when no trained model has
valid metrics. New training summaries contain architecture and checkpoint
fingerprints; modified artifacts with stale summaries are excluded as well.
Models should be compared using the same development data and scoring setup.
These are development metrics, not a separate test-set evaluation.

From Python, when `src/` is on the import path:

```python
from model_registry import get_best_model, load_best_model

directory = get_best_model("models", criterion="accuracy")
net = load_best_model("models", criterion="accuracy", device="cpu")
```

`load_best_model` loads the selected architecture and weights and sets evaluation
mode. Preprocess raw inputs with the same routine used during that model's
training before passing them to the network.

## Command-line help

All local commands support `-h` and `--help`. Help includes their options and
default values:

```bash
python src/train.py --help
python src/prepare_data.py --help
python src/model_registry.py --help
```

An unknown command, option, invalid value, or missing required argument prints
an error followed by the complete help menu and exits with status 2 before
training, downloading data, or updating the registry. Option names must be
written in full; abbreviations are rejected. Explicit help exits with status 0.

### Training argument rules

`--model-dir DIR` is required, including for `--check-only`; explicitly select
the architecture to use. `--out` is retained as an alias for the same directory.
Running `python src/train.py` without arguments displays help and exits.

The help menu groups model/input paths, training settings, runtime settings,
and execution mode, and includes examples. All other options have defaults:
`--repo` uses the repository containing the script (independent of the working
directory), while data and split paths default to `data/train_100k.npz` and
`data/split_indices.npz` relative to that repository. Parsed paths are printed
as absolute paths in the effective configuration.

Input files and architecture.py must exist. Epochs and threads must be positive
integers; the seed must be in [0, 4294967295]. Selecting CUDA requires an available
CUDA device. `--check-only` and `--replace` are mutually exclusive. Existing
weights require `--replace` for training, but do not prevent a read-only check.
Epochs and the checkpoint criterion apply to training only. Validation errors
show the full help menu before any dataset loading or output writes.
