# adaptative_model

The final `model_4_transfer_v024` checkpoint, with the original weights,
architecture, and preprocessing recipe. It has not been retrained or adapted to
the demo data.

The model is a temporal CNN with 7,352 trainable parameters and a branch that
captures temporal and cross-channel relationships. It accepts four channels
with 256 time steps and returns a class from 0 to 7.

Inference uses the original PyTorch network in evaluation and inference mode on
CPU, with one PyTorch thread. Preprocessing uses the shared Numba normalizer in
`src/normalize_data.py`, which `Model()` warms up during initialization. Keep
the repository layout when moving or deploying the model so it can find that file.

## Running the model

From the repository root:

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py
python src/evaluate.py
```

The official contest entry point is `model.py` at the repository root.
`python src/train.py` trains the same architecture with the fixed recipe and
saves a checkpoint to `model/retrained.pt`. Evaluate those weights with
`python src/evaluate.py --trained`.

`Model()` loads `weights.pt` and uses the fixed V024 preprocessing in
`src/normalize_data.py`. `predict(sample)` preprocesses the raw series and calls
`self.net`, the standard `TemporalCNN` PyTorch module. These are the entry points
for future optimization; no alternative backend is selected automatically.

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

`history.json` records loss, accuracy and score by epoch. `training.json`
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
performance on new acquisitions; class 6 remains a weak point. The earlier
optimized backend's latency measurements do not apply to this PyTorch runtime.

SHA-256 of `weights.pt`:
`dd09948e3c59b1956cec702e06a254f53d2b728aafce40430e437de4263cd9ca`.
