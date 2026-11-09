"""Competition entry point using ONNX Runtime on CPU."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.normalize_data import preprocess_series

ROOT = Path(__file__).resolve().parent


class Model:
    """Predict a class from one raw (4, 256) sample."""

    def __init__(self, trained=False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model_path = ROOT / ("retrained.onnx" if trained else "model.onnx")
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Compile preprocessing for the harness's read-only inputs during setup.
        dummy = np.zeros((4, 256), dtype=np.float32)
        dummy.setflags(write=False)
        preprocess_series(dummy)

    def predict(self, sample):
        inputs = preprocess_series(sample.series)
        logits = self.session.run([self.output_name], {self.input_name: inputs})[0]
        return int(np.argmax(logits, axis=1).item())
