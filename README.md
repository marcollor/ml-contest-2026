# Native C inference

The contest entry point is `model.py`. On Linux or WSL, it compiles
`native/inference.c` into a shared library during setup and loads
`model/weights.bin` once. Each `Model.predict(sample)` call passes the raw
`float32` series to C via `ctypes`. The engine runs the V024 preprocessing,
the TemporalCNN and the final argmax. It uses AVX2/FMA kernels for the initial,
depthwise, and pointwise convolutions on compatible CPUs, with a scalar
fallback.

Run the contest evaluator from the repository root with NumPy, a C compiler
(`cc` or `CC`) and the dataset available:

```bash
python -m pip install -r requirements.txt
python harness.py evaluate --model model.py
```

## Export the weights

The repository includes the weights for `model.onnx`. To repack that graph:

```bash
python -m pip install onnx
python export_c.py
```

The exporter checks that the ONNX graph matches the supported architecture,
including the fixed downsampling filters. It uses the already folded
convolution and BatchNorm weights, packs the pointwise filters in groups of
four output channels, and writes 7,064 FP32 values with checksums.

To regenerate both artifacts from the PyTorch checkpoint:

```bash
python -m pip install torch numba onnx
python export_onnx.py
python export_c.py
```

Pass `--trained` to both commands for `model/retrained.pt`. The resulting
`model/retrained.bin` is used by `Model(trained=True)`. An older export with
only `retrained.onnx` can still run if `onnxruntime` and `numba` are installed.

## Check against ONNX Runtime

```bash
python -m pip install onnxruntime numba
python validate_native.py
python validate_native.py --full
```

The full check compares classes on the entire dataset and 10,000 synthetic
stress samples. It also compares normalized inputs and logits on the first
batch of each set. The default contest model runs with NumPy and the C compiler.
