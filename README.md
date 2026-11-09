# GDG UAM ML Contest

This repository trains and runs [adaptative_model](model/README.md),
a temporal CNN that classifies four-channel time series into eight classes.

## Local workflow

Run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python src/evaluate.py
```

Preparation downloads the datasets when needed and recreates the fixed splits.
Training uses CUDA when available and CPU otherwise. It saves the best checkpoint
to `model/retrained.pt` using the fixed 80-epoch recipe.

Evaluation uses the bundled V024 weights and prints accuracy on all demo samples.
Use `python src/evaluate.py --trained` to evaluate the checkpoint from training.
Use `python src/train.py --check-only` for a quick setup check.

The contest loads `Model()` from the root `model.py`. Its evaluator and data API
are included as `harness.py` and `competition_api.py`.
The architecture, bundled weights and training records are in `model/`.

## Your submission

Submit a GitHub repository that contains:

- `model.py` at the top level, defining `class Model` (see below);
- your trained model: weights and any helper modules `model.py` loads;
- `requirements.txt` with the libraries `model.py` uses: we install them before the evaluation,
  since there is no internet while your model runs.

Nothing is trained on our side: `Model()` must load your already trained model from the
repository. Email the repository link to all four organizers: **gdguam@gmail.com,
ibondmingo4@gmail.com, adam.maltoni@gmail.com, hector.tablerodiaz@gmail.com**. Make sure we can
clone it (public, or shared with us if private). GitHub rejects files over 100 MB, so use
[Git LFS](https://git-lfs.com) for larger weights.

Before sending it, check the local workflow from a fresh clone.

`model.py` defines `class Model`:

```python
from pathlib import Path

from competition_api import ABSTAIN, Sample

HERE = Path(__file__).parent  # load your files relative to this file


class Model:
    def __init__(self):
        # Load weights here. Untimed, but must finish within the setup timeout.
        self.weights = HERE / "weights.pt"

    def predict(self, x: Sample) -> int:
        # Return a class id in 0..7, or ABSTAIN (-1) to pass.
        return ABSTAIN
```

- `x.series` is a read-only `float32` array of shape `(4, 256)` (4 channels, 256 time steps) that may contain NaN.
- `x.meta` is a dict with `sample_rate`, `group` and `level`.
- Labels are never passed to your model.
- An exception, or any return value other than an `int` in `0..7` or `-1`, counts as a wrong answer.

**Use relative paths.** Your model runs with its own folder as the working directory, and that
folder is on `sys.path`, both locally and in the official evaluation. Keep the files it needs
(weights, helper modules) next to it or in subfolders, and build their paths from `__file__` as
above. Absolute paths such as `C:/Users/you/...` or `/home/you/...` do not exist on the evaluation
machine; the harness warns you when it finds one in your model file.

## Training data

```python
from competition_api import load_dataset

ds = load_dataset("data/train_100k.npz")   # memory-mapped, so opening it is instant
ds.series   # (100000, 4, 256) float32, may contain NaN
ds.labels   # (100000,) class ids 0..7
ds.meta     # {"sample_rate": array, "group": array, "level": array}
ds[j]       # row j as the exact Sample your model would receive
```

- The evaluation data may differ from the training data in noise, ranges and class balance. Don't
  rely on quirks of the training file.
- `data/test_demo.npz` has 400 labeled samples drawn like the evaluation data, so you can check
  your model on them (see the local workflow above).

## How evaluation works

**Every model is evaluated in Google Colab**, so all teams are compared under the same conditions.
You don't need Colab to develop: any machine with Python 3.10+ works, and `harness.py` applies
the same limits there. Speeds differ between machines, though, so if your model is close to the
time limits, try it in Colab too ([`colab.ipynb`](colab.ipynb)).

Each team's model is evaluated in 3 independent runs, and the ranking uses the mean score. Each
run starts a fresh process and goes through three phases:

1. **Setup:** `Model()` is created. Untimed, but it must finish within 120 s.
2. **Warmup (unscored):** up to 100 samples or 5 s of prediction time. If a call overruns the
   warmup limits (2 s per call, or the time left of the 5 s), your model is restarted (setup runs
   again) and goes straight to the scored phase.
3. **Scored:** samples arrive one at a time. Only `predict` is timed, and the run ends once your
   total prediction time reaches **60 s**. The faster you are, the more samples you answer.
    - A call that takes longer than **0.5 s** counts as wrong (its time is still charged).
    - A call that has not returned after 0.75 s is killed, and the run ends there.

Numerical libraries are limited to one thread (`OMP_NUM_THREADS=1` and similar), and Python's `random`,
`numpy.random` and `torch` are seeded for each run. The authoritative values are in `EVAL_CONFIG`
at the top of `harness.py`.

## Scoring

```math
\text{Score} = \left(\prod_{k=0}^{7} T_k\right)^{1/8},
\qquad
T_k = \max\left(\varepsilon,\; n_k^{\alpha} \cdot \frac{c_k - \lambda w_k}{n_k} \cdot \left(\frac{c_k}{n_k}\right)^{p}\right)
```

Over the scored samples whose true class is $k$: $n_k$ is how many there were, $c_k$ how many you
got right and $w_k$ how many you got wrong. Timeouts, exceptions and invalid outputs count as
wrong. Abstentions count in $n_k$, but as neither right nor wrong.

| Parameter     | Value | Meaning                                                  |
| ------------- | ----- | -------------------------------------------------------- |
| $\varepsilon$ | 0.01  | minimum of each class term                               |
| $\lambda$     | 0.33  | penalty for each wrong answer                            |
| $\alpha$      | 0.5   | speed exponent: answering 4× more samples doubles a term |
| $p$           | 2     | per-class precision is squared                           |

Every class matters: the geometric mean drops sharply if one class term falls to $\varepsilon$.
