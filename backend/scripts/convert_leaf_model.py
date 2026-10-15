"""Перенос готовых весов листовой CNN из Keras в ONNX; это не обучение.

Для конвертации нужны дополнительные onnx и h5py."""
import io
import json
from pathlib import Path
import zipfile
import h5py
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def convert(source, destination):
    with zipfile.ZipFile(source) as archive:
        config = json.loads(archive.read("config.json"))
        weights_data = archive.read("model.weights.h5")
    nodes, weights = [], []
    previous = "input"
    flatten_shape = None
    with h5py.File(io.BytesIO(weights_data), "r") as store:
        for layer in config["config"]["layers"]:
            kind, cfg = layer["class_name"], layer["config"]
            name = cfg["name"]
            if kind in {"InputLayer", "Sequential"}:
                # Размер и масштаб пикселей готовит vision.py; случайные повороты нужны при обучении.
                if kind == "Sequential":
                    allowed = {"InputLayer", "Resizing", "Rescaling", "RandomFlip", "RandomRotation"}
                    if any(x["class_name"] not in allowed for x in cfg["layers"]):
                        raise ValueError("Unsupported nested layer")
                continue
            if kind == "Conv2D":
                if cfg["padding"] != "valid" or cfg["activation"] != "relu":
                    raise ValueError("Unexpected convolution contract")
                kernel = np.asarray(store[f"layers/{name}/vars/0"]).transpose(3, 2, 0, 1)
                bias = np.asarray(store[f"layers/{name}/vars/1"])
                weights += [numpy_helper.from_array(kernel, name + "_w"), numpy_helper.from_array(bias, name + "_b")]
                nodes += [helper.make_node("Conv", [previous, name + "_w", name + "_b"], [name + "_linear"],
                    kernel_shape=cfg["kernel_size"], strides=cfg["strides"]),
                    helper.make_node("Relu", [name + "_linear"], [name])]
            elif kind == "MaxPooling2D":
                nodes.append(helper.make_node("MaxPool", [previous], [name], kernel_shape=cfg["pool_size"], strides=cfg["strides"]))
            elif kind == "Flatten":
                flatten_shape = layer["build_config"]["input_shape"][1:]
                nodes.append(helper.make_node("Flatten", [previous], [name], axis=1))
            elif kind == "Dense":
                kernel = np.asarray(store[f"layers/{name}/vars/0"])
                if flatten_shape is not None:
                    height, width, channels = flatten_shape
                    kernel = kernel.reshape(height, width, channels, -1).transpose(2, 0, 1, 3).reshape(kernel.shape)
                    flatten_shape = None
                bias = np.asarray(store[f"layers/{name}/vars/1"])
                weights += [numpy_helper.from_array(kernel.T.copy(), name + "_w"), numpy_helper.from_array(bias, name + "_b")]
                nodes.append(helper.make_node("Gemm", [previous, name + "_w", name + "_b"], [name + "_linear"], transB=1))
                activation = "Relu" if cfg["activation"] == "relu" else "Softmax"
                nodes.append(helper.make_node(activation, [name + "_linear"], [name], **({"axis": 1} if activation == "Softmax" else {})))
            else:
                raise ValueError(f"Unsupported layer: {kind}")
            previous = name
    graph = helper.make_graph(nodes, "potato_leaf_cnn", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, 3])], initializer=weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
