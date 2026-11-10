# adaptative_model

The final `model_4_transfer_v024` checkpoint, with the original weights,
architecture, and preprocessing recipe. It has not been retrained or adapted to
the demo data.

The model is a temporal CNN with 7,352 trainable parameters and a branch that
captures temporal and cross-channel relationships. It accepts four channels
with 256 time steps and returns a class from 0 to 7.

The contest entry point uses the native C engine and `model/weights.bin`.
The engine checks the packed weights against `model/model.onnx`. The PyTorch
checkpoint and its training records are in [training/](training/). See the [root README](../README.md) for export
and validation commands.

## Running the model

Download the demo dataset as described in [data/README.md](../data/README.md).
From the repository root:

```bash
python -m pip install -r requirements.txt
python src/evaluate.py
```

`python src/train.py` trains the same architecture with the fixed recipe and
saves a checkpoint to `model/retrained.pt`. Run `python -m tools.export_onnx --trained`
and `python -m tools.export_c --trained` before evaluating it with
`python src/evaluate.py --trained`.

## Preprocessing

Each channel is processed independently using the saved recipe:

1. Treat NaN and infinity as missing.
2. Clip finite values at the median ± six robust scales, where each scale is
   `max(1.4826 * MAD, 0.001)`.
3. Interpolate missing values linearly, extending the nearest finite value at
   boundaries. Entirely missing channels become zero.
4. Subtract the temporal mean and divide by `sqrt(variance + 1e-6)`.
5. Clip to `[-10, 10]` and return a float32 array of shape `(1, 4, 256)`.

The input is not modified. Pass raw signals: do not apply an additional global
normalization or normalize the input twice.

## Checkpoint provenance

[history.json](training/history.json) records loss, accuracy and score by epoch.
[training.json](training/training.json)
records the original recipe, file fingerprints and recall for the selected
epoch.

Historical accuracy results from the previous audit, not new evaluations of
this package's runtime:

| Dataset | Accuracy |
|---|---:|
| Development, 5,000 windows | 92.02% |
| IID test, 20,000 windows | 91.65% |
| IID with perturbations | 91.06% |
| Demo, 400 windows | 86.50% (346/400) |

The demo was a known diagnostic set. These results do not guarantee equivalent
performance on new acquisitions; class 6 remains a weak point.

SHA-256 of [weights.pt](training/weights.pt):
`dd09948e3c59b1956cec702e06a254f53d2b728aafce40430e437de4263cd9ca`.
