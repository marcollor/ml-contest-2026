"""Competition entry point using a specialized native CPU engine."""

from pathlib import Path

from native.runtime import NativeEngine

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TRAINED_CHECKPOINT = MODEL_DIR / "retrained.pt"


def reference_graph(trained=False):
    """Find the reference graph, including retrained exports from older layouts."""
    path = MODEL_DIR / ("retrained.onnx" if trained else "model.onnx")
    legacy = ROOT / "retrained.onnx"
    if trained and not path.is_file() and legacy.is_file():
        return legacy
    return path


class Model:
    """Predict a class from one raw (4, 256) sample."""

    def __init__(self, trained=False):
        onnx_path = reference_graph(trained)
        weights_path = MODEL_DIR / ("retrained.bin" if trained else "weights.bin")
        if trained and not weights_path.is_file():
            # Keep existing retrained ONNX exports usable until they are packed.
            self.engine = _LegacyOnnx(onnx_path)
        else:
            self.engine = NativeEngine(weights_path, onnx_path)

    def predict(self, sample):
        return self.engine.predict(sample.series)


class _LegacyOnnx:
    def __init__(self, model_path):
        import numpy as np
        import onnxruntime as ort

        from src.normalize_data import preprocess_series

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.preprocess = preprocess_series

        # Compile preprocessing for the harness's read-only inputs during setup.
        dummy = np.zeros((4, 256), dtype=np.float32)
        dummy.setflags(write=False)
        preprocess_series(dummy)

    def predict(self, raw):
        inputs = self.preprocess(raw)
        logits = self.session.run([self.output_name], {self.input_name: inputs})[0]
        return int(logits.argmax(axis=1).item())
