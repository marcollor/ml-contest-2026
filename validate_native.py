"""Check the specialized C engine against the ONNX model without using labels."""

import argparse
from itertools import islice
from pathlib import Path

import numpy as np
import onnxruntime as ort

from competition_api import load_dataset
from native.runtime import NativeEngine, load_library
from src.normalize_data import preprocess_series


ROOT = Path(__file__).resolve().parent


def stress_samples(count):
    rng = np.random.default_rng(20261103)
    for i in range(count):
        raw = (rng.normal(size=(4, 256)) * 10.0 ** rng.uniform(-30, 30)).astype(np.float32)
        raw[rng.random(raw.shape) < 0.1] = np.nan
        if i % 7 == 0:
            raw[i % 4] = np.nan
        if i % 11 == 0:
            raw.fill(0)
        if i % 13 == 0:
            raw.fill(np.nan)
        if i % 17 == 0:
            raw[0, 0] = np.inf
        if i % 19 == 0:
            raw[1, -1] = np.finfo(np.float32).max
        raw.setflags(write=False)
        yield raw


def compare(samples, reference, engines):
    input_name = reference.get_inputs()[0].name
    mismatches = {engine.mode: 0 for engine in engines}
    logit_errors = {}
    total = 0
    while batch := list(islice(samples, 128)):
        normalized = np.concatenate([preprocess_series(raw) for raw in batch])
        logits = np.concatenate([
            reference.run(None, {input_name: row[None]})[0] for row in normalized
        ])
        expected = logits.argmax(axis=1)
        for engine in engines:
            predictions = np.array([engine.predict(raw) for raw in batch])
            mismatches[engine.mode] += int(np.count_nonzero(predictions != expected))
            if total == 0:
                native_input = np.concatenate([engine.preprocess(raw) for raw in batch])
                native_logits = np.array([engine.logits(raw) for raw in batch])
                np.testing.assert_allclose(native_input, normalized, rtol=1e-5, atol=2e-6)
                np.testing.assert_allclose(native_logits, logits, rtol=0, atol=1e-4)
                logit_errors[engine.mode] = float(np.max(np.abs(native_logits - logits)))
        total += len(batch)
    return total, mismatches, logit_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use all dataset rows and 10,000 stress cases.")
    parser.add_argument("--trained", action="store_true", help="Validate the retrained model.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "train_100k.npz")
    args = parser.parse_args()
    onnx_path = ROOT / ("retrained.onnx" if args.trained else "model.onnx")
    weights = ROOT / "model" / ("retrained.bin" if args.trained else "weights.bin")
    engines = [NativeEngine(weights, onnx_path, mode="scalar")]
    if load_library().engine_has_avx2():
        engines.append(NativeEngine(weights, onnx_path, mode="avx2_fma"))

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    dataset = load_dataset(args.dataset)
    rng = np.random.default_rng(20261103)
    indices = (range(len(dataset)) if args.full
               else rng.choice(len(dataset), size=min(len(dataset), 256), replace=False))
    results = {
        "real": compare((dataset.series[i] for i in indices), reference, engines),
        "stress": compare(stress_samples(10_000 if args.full else 128), reference, engines),
    }
    for name, (total, mismatches, logit_errors) in results.items():
        print(f"{name}: {total:,} samples, class mismatches={mismatches}, "
              f"max logit error on first batch={logit_errors}")
    if any(count for _, mismatches, _ in results.values() for count in mismatches.values()):
        raise SystemExit("Native and ONNX predictions differ.")


if __name__ == "__main__":
    main()
