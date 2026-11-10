"""Pack the weights of the fixed TemporalCNN ONNX graph for native inference."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx
from onnx import numpy_helper

from native.runtime import WEIGHT_COUNT, WEIGHT_HEADER, WEIGHT_MAGIC


ROOT = Path(__file__).resolve().parent
# Topology, attributes, and graph constants of the supported opset 17 export.
GRAPH_SIGNATURE = "7c564cd5c25bd80aa5cd78d634c3da80e50ffc314dff0d948e85975ff1daec86"


def graph_signature(model):
    """Fingerprint computation and constants without depending on generated tensor names."""
    graph = model.graph
    sources = {"": None}
    sources.update({value.name: ("initializer", value.data_type, tuple(value.dims))
                    for value in graph.initializer})

    def tensor_type(value):
        tensor = value.type.tensor_type
        return (tensor.elem_type, tuple(d.dim_value if d.HasField("dim_value") else None
                                        for d in tensor.shape.dim))

    for index, value in enumerate(graph.input):
        sources[value.name] = ("input", index)
    nodes = []
    for node in graph.node:
        inputs = [sources[name] for name in node.input]
        if node.op_type == "Identity" and not node.domain:
            sources[node.output[0]] = inputs[0]
            continue
        attributes = [attribute.SerializeToString(deterministic=True).hex()
                      for attribute in sorted(node.attribute, key=lambda item: item.name)]
        for index, name in enumerate(node.output):
            sources[name] = ("node", len(nodes), index)
        nodes.append((node.domain, node.op_type, inputs, attributes, len(node.output)))
    signature = (
        sorted((opset.domain, opset.version) for opset in model.opset_import),
        [tensor_type(value) for value in graph.input],
        nodes,
        [(sources[value.name], tensor_type(value)) for value in graph.output],
    )
    return hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()


def pack_weights(model):
    if graph_signature(model) != GRAPH_SIGNATURE:
        raise ValueError("Unsupported ONNX graph for this TemporalCNN C engine.")
    tensors = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in tensors:
            tensors[node.output[0]] = tensors[node.input[0]]

    convs = [node for node in model.graph.node if node.op_type == "Conv"]
    stem, *body = convs
    expected_filter = np.broadcast_to([0.25, 0.5, 0.25], (32, 1, 3))
    for index in (0, 3, 6):
        if not np.array_equal(tensors[body[index].input[1]], expected_filter):
            raise ValueError("Unexpected fixed downsampling filter.")
    depthwise = [body[index] for index in (1, 4, 7, 9)]
    pointwise = [body[index] for index in (2, 5, 8, 10)]
    relations, classifier = [node for node in model.graph.node if node.op_type == "Gemm"]
    # ONNX export has already folded BatchNorm into the convolution weights and biases.
    parts = [
        tensors[stem.input[1]],
        tensors[stem.input[2]],
        np.stack([tensors[node.input[1]] for node in depthwise]),
        np.stack([tensors[node.input[2]] for node in depthwise]),
        np.stack([tensors[node.input[1]].reshape(8, 4, 32).transpose(0, 2, 1)
                  for node in pointwise]),
        np.stack([tensors[node.input[2]] for node in pointwise]),
        tensors[relations.input[1]].T,
        tensors[relations.input[2]],
        tensors[classifier.input[1]].T,
        tensors[classifier.input[2]],
    ]
    weights = np.concatenate([part.ravel() for part in parts]).astype("<f4", copy=False)
    if weights.size != WEIGHT_COUNT or not np.isfinite(weights).all():
        raise ValueError("Invalid native weight count or non-finite weights.")
    return weights.tobytes()


def export_c(onnx_path, output):
    source = onnx_path.read_bytes()
    payload = pack_weights(onnx.load_model_from_string(source))
    header = WEIGHT_HEADER.pack(WEIGHT_MAGIC, hashlib.sha256(source).digest(),
                                hashlib.sha256(payload).digest(), WEIGHT_COUNT)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(header)
            temporary.write(payload)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", action="store_true", help="Use retrained.onnx.")
    parser.add_argument("--onnx", type=Path, help="Override the input path.")
    parser.add_argument("--out", type=Path, help="Override the output path.")
    args = parser.parse_args()
    onnx_path = args.onnx or ROOT / ("retrained.onnx" if args.trained else "model.onnx")
    output = args.out or ROOT / "model" / ("retrained.bin" if args.trained else "weights.bin")
    export_c(onnx_path, output)
    print(f"Packed {WEIGHT_COUNT} FP32 weights from {onnx_path.name} into {output}.")


if __name__ == "__main__":
    main()
