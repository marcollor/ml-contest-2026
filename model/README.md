# TemporalCNN model artifacts

TemporalCNN has 7,352 trainable parameters and a branch that captures temporal
and cross-channel relationships. `Model.predict(sample)` accepts a raw signal
with four channels and 256 time steps and returns a class from 0 to 7.

| File | Use |
| --- | --- |
| `weights.bin` | Packed FP32 weights loaded by the native C engine |
| `model.onnx` | Reference graph used to pack and validate the native weights |
| `training/weights.pt` | Bundled PyTorch checkpoint used to regenerate the inference artifacts |

## Running the model

Use Linux or WSL with Python 3.10+ and a C compiler available as `cc` or via
`CC`. Download the demo dataset using the [data instructions](../data/README.md),
then run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python src/evaluate.py
```

See the [export and validation commands](../README.md#development-export-validate-and-retrain)
to regenerate the bundled artifacts and the [training guide](../src/README.md)
to train and evaluate a new checkpoint.

## Preprocessing

The engine processes each channel independently:

1. Treat NaN and infinity as missing.
2. Clip finite values at the median ± six robust scales, where each scale is
   `max(1.4826 * MAD, 0.001)`.
3. Interpolate missing values linearly, extending the nearest finite value at
   boundaries. Entirely missing channels become zero.
4. Subtract the temporal mean and divide by `sqrt(variance + 1e-6)`.
5. Clip to `[-10, 10]` and return a float32 array of shape `(1, 4, 256)`.

Pass raw signals to `Model.predict`. Preprocessing preserves the input array.
