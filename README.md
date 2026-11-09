# Inference with ONNX Runtime

From the repository root, install the production dependencies and the additional
tools needed in the export environment:

```bash
python -m pip install -r requirements.txt
python -m pip install torch onnx
python export_onnx.py
```

`export_onnx.py` imports `TemporalCNN` from `model/architecture.py`, loads
`model/weights.pt` on CPU, and calls `.eval()`. It exports `model.onnx` to the
repository root using `torch.onnx.export`, with opset 17, an input shape of
`(1, 4, 256)`, and a dynamic batch dimension for both input and output. PyTorch
and ONNX are only needed for export; include the generated file alongside the
code before evaluation.

`model.py` preserves the official `Model` class and `predict(self, sample)`
method. It loads ONNX Runtime with `CPUExecutionProvider` and one thread each
for intra-op and inter-op execution. It compiles the Numba preprocessor from
`src/normalize_data.py` during initialization; each prediction normalizes
`sample.series`, runs the session, and returns the `argmax` as an `int`.
Inference does not import PyTorch.

To evaluate locally, with the official dataset `data/train_100k.npz` available:

```bash
python harness.py evaluate --model model.py
```

Optionally, `python export_onnx.py --trained` exports `model/retrained.pt` to
`retrained.onnx`, which is loaded by `Model(trained=True)`. The official harness
uses `Model()` and therefore loads `model.onnx`.

Production dependencies are listed in the root `requirements.txt`.
Keep the folder structure intact so the preprocessor and models can be found.
