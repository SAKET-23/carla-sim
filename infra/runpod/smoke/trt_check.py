"""G1 Task 5 TensorRT check: build an FP16 engine for a tiny ONNX Conv net, run it, compare with onnxruntime-gpu.

Uses the repo's own builder and runner (hybrid_av.perception.build_engine / trt_common), so it also checks them.
Pass: the engine builds and its output matches onnxruntime within 1e-2.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from hybrid_av.perception.build_engine import build_engine
from hybrid_av.perception.trt_common import Engine


def tiny_conv_onnx(path: Path) -> np.ndarray:
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.2, "w")
    b = numpy_helper.from_array(rng.standard_normal(8).astype(np.float32) * 0.1, "b")
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 32, 32])],
        [w, b],
    )  # fmt: skip
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return rng.standard_normal((1, 3, 32, 32)).astype(np.float32)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        onnx_path, engine_path = Path(d) / "tiny.onnx", Path(d) / "tiny.engine"
        x = tiny_conv_onnx(onnx_path)
        ref = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in ref.get_providers():
            print("FAIL: onnxruntime has no CUDAExecutionProvider", file=sys.stderr)
            return 1
        expected = ref.run(None, {"x": x})[0]
        build_engine(onnx_path, engine_path, fp16=True, workspace_gb=1.0)
        got = Engine(engine_path)(x=torch.from_numpy(x).cuda())["y"].float().cpu().numpy()
    err = float(np.abs(got - expected).max())
    print(f"max abs difference vs onnxruntime-gpu: {err:.2e}")
    if err > 1e-2:
        print("FAIL: TensorRT output differs from onnxruntime by more than 1e-2", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
