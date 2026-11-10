"""Load the contest's specialized C engine and keep its weights alive."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import weakref

import numpy as np


WEIGHT_MAGIC = b"TCNNv1\0\0"
WEIGHT_HEADER = struct.Struct("<8s32s32sI")
WEIGHT_COUNT = 7064
_MODES = {"auto": -1, "scalar": 0, "avx2_fma": 1}


@lru_cache(maxsize=1)
def load_library():
    if sys.platform != "linux":
        raise RuntimeError("The native engine builds on Linux or WSL.")
    source = Path(__file__).with_name("inference.c")
    with TemporaryDirectory(prefix="temporal-cnn-") as directory:
        output = Path(directory) / "libtemporal_cnn.so"
        try:
            subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-O3", "-fPIC", "-shared",
                 "-ffp-contract=off", "-fno-fast-math", str(source), "-lm", "-o", str(output)],
                capture_output=True, text=True, check=True, timeout=60,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Native compilation failed:\n{error.stderr}") from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Could not build the native engine: {error}") from error
        library = ctypes.CDLL(str(output))

    pointer = ctypes.c_void_p
    library.engine_has_avx2.argtypes = []
    library.engine_has_avx2.restype = ctypes.c_int
    library.engine_create.argtypes = [pointer, ctypes.c_size_t, ctypes.c_int]
    library.engine_create.restype = pointer
    library.engine_destroy.argtypes = [pointer]
    library.engine_destroy.restype = None
    library.engine_mode.argtypes = [pointer]
    library.engine_mode.restype = ctypes.c_char_p
    library.engine_predict.argtypes = [pointer, pointer]
    library.engine_predict.restype = ctypes.c_int
    library.engine_logits.argtypes = [pointer, pointer, pointer]
    library.engine_logits.restype = ctypes.c_int
    library.engine_preprocess.argtypes = [pointer, pointer]
    library.engine_preprocess.restype = ctypes.c_int
    return library


def _read_weights(weights_path, onnx_path):
    data = weights_path.read_bytes()
    if len(data) != WEIGHT_HEADER.size + 4 * WEIGHT_COUNT:
        raise ValueError(f"Incorrect size for {weights_path.name}. Run export_c.py.")
    magic, onnx_hash, payload_hash, count = WEIGHT_HEADER.unpack_from(data)
    payload = memoryview(data)[WEIGHT_HEADER.size:]
    if (magic != WEIGHT_MAGIC or count != WEIGHT_COUNT
            or hashlib.sha256(payload).digest() != payload_hash):
        raise ValueError(f"Invalid {weights_path.name}. Run export_c.py.")
    if onnx_path is not None and onnx_path.is_file():
        if hashlib.sha256(onnx_path.read_bytes()).digest() != onnx_hash:
            raise ValueError(f"{weights_path.name} does not match {onnx_path.name}. Run export_c.py.")
    return np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)


def _input_array(raw):
    array = np.asarray(raw)
    if array.shape != (4, 256):
        raise ValueError(f"Expected a sample of shape (4, 256), got {array.shape}.")
    if array.dtype != np.float32 or not (array.flags.c_contiguous and array.flags.aligned):
        array = np.require(array, dtype=np.float32, requirements=("C", "A"))
    return array


class NativeEngine:
    def __init__(self, weights_path, onnx_path=None, mode="auto"):
        weights = _read_weights(weights_path, onnx_path)
        library = load_library()
        engine = library.engine_create(weights.ctypes.data, weights.size, _MODES[mode])
        if not engine:
            raise RuntimeError(f"Could not initialize native weights in mode {mode}.")
        self._engine = engine
        self._library = library
        self._predict = library.engine_predict
        self._destroy = weakref.finalize(self, library.engine_destroy, engine)
        self.mode = library.engine_mode(engine).decode("ascii")

    def predict(self, raw):
        array = _input_array(raw)
        prediction = self._predict(self._engine, array.ctypes.data)
        if not 0 <= prediction < 8:
            raise RuntimeError(f"Native prediction failed ({prediction}).")
        return prediction

    def logits(self, raw):
        array = _input_array(raw)
        output = np.empty(8, dtype=np.float32)
        if self._library.engine_logits(self._engine, array.ctypes.data, output.ctypes.data):
            raise RuntimeError("Native logit calculation failed.")
        return output

    def preprocess(self, raw):
        array = _input_array(raw)
        output = np.empty((1, 4, 256), dtype=np.float32)
        if self._library.engine_preprocess(array.ctypes.data, output.ctypes.data):
            raise RuntimeError("Native preprocessing failed.")
        return output
