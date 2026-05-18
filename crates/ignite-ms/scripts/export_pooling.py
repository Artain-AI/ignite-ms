"""Append mean-pooling + L2-norm to a raw ONNX transformer model.

Takes a standard transformer ONNX (output: last_hidden_state) and produces
a model with output: sentence_embedding (pooled + normalized).

This script is embedded in ignite-ms and invoked during model provisioning.
Requires: python3, onnx, numpy

Usage:
    python3 -c "$(cat this_script)" --input raw.onnx --output model.onnx --hidden-dim 384
"""

import sys
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def append_pool_and_norm(src_path: str, dst_path: str, hidden_dim: int) -> None:
    model = onnx.load(src_path)
    graph = model.graph

    float_type = TensorProto.FLOAT
    np_float = np.float32

    # Find transformer output
    target_out = None
    for o in graph.output:
        if o.name == "last_hidden_state":
            target_out = o
            break

    if target_out is None:
        # Some models use different output names — try first output
        if len(graph.output) == 1:
            target_out = graph.output[0]
            hidden_state_name = target_out.name
        else:
            names = [o.name for o in graph.output]
            raise RuntimeError(f"last_hidden_state not found in outputs: {names}")
    else:
        hidden_state_name = "last_hidden_state"

    # Drop existing outputs
    del graph.output[:]

    nodes = []

    # Cast attention_mask (int64) to float
    nodes.append(helper.make_node(
        "Cast", ["attention_mask"], ["mask_f"],
        to=float_type, name="pool_mask_cast"
    ))

    # Unsqueeze mask: (B, S) -> (B, S, 1)
    nodes.append(helper.make_node(
        "Unsqueeze", ["mask_f"], ["mask_unsq"],
        axes=[2], name="pool_mask_unsqueeze"
    ))

    # Multiply hidden states by mask
    nodes.append(helper.make_node(
        "Mul", [hidden_state_name, "mask_unsq"], ["masked_hidden"],
        name="pool_mask_mul"
    ))

    # Sum over seq dim
    nodes.append(helper.make_node(
        "ReduceSum", ["masked_hidden"], ["sum_hidden"],
        axes=[1], keepdims=0, name="pool_reduce_sum"
    ))

    # Count tokens per sample
    nodes.append(helper.make_node(
        "ReduceSum", ["mask_unsq"], ["token_count"],
        axes=[1], keepdims=0, name="pool_count_sum"
    ))

    # Clamp token count to avoid division by zero
    one = numpy_helper.from_array(np.array([1.0], dtype=np_float), "pool_one")
    graph.initializer.append(one)
    nodes.append(helper.make_node(
        "Max", ["token_count", "pool_one"], ["token_count_safe"],
        name="pool_count_max"
    ))

    # Mean pooling
    nodes.append(helper.make_node(
        "Div", ["sum_hidden", "token_count_safe"], ["mean_pooled"],
        name="pool_mean_div"
    ))

    # L2 normalization
    pow_exp = numpy_helper.from_array(np.array([2.0], dtype=np_float), "l2_pow_two")
    graph.initializer.append(pow_exp)
    nodes.append(helper.make_node(
        "Pow", ["mean_pooled", "l2_pow_two"], ["l2_sq"], name="l2_pow"
    ))
    nodes.append(helper.make_node(
        "ReduceSum", ["l2_sq"], ["l2_sum"],
        axes=[1], keepdims=1, name="l2_reduce"
    ))
    l2_eps = numpy_helper.from_array(np.array([1e-12], dtype=np_float), "l2_eps")
    graph.initializer.append(l2_eps)
    nodes.append(helper.make_node(
        "Max", ["l2_sum", "l2_eps"], ["l2_sum_safe"], name="l2_eps_max"
    ))
    nodes.append(helper.make_node(
        "Sqrt", ["l2_sum_safe"], ["l2_norm"], name="l2_sqrt"
    ))
    nodes.append(helper.make_node(
        "Div", ["mean_pooled", "l2_norm"], ["sentence_embedding"], name="l2_div"
    ))

    for n in nodes:
        graph.node.append(n)

    # Add output
    new_out = helper.make_tensor_value_info(
        "sentence_embedding", float_type, ["batch", hidden_dim]
    )
    graph.output.append(new_out)

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    size_mb = os.path.getsize(dst_path) / 1e6
    print(f"[export] saved {dst_path} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    args = parser.parse_args()
    append_pool_and_norm(args.input, args.output, args.hidden_dim)
