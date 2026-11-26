"""TensorRT 10 engine loading and execution with torch.cuda buffers. `tensorrt` is imported lazily."""

from pathlib import Path

import numpy as np
import torch

_MSG = "TensorRT engine not found: {p}. Build it on this GPU with: uv run python scripts/export_models.py --build-engines"


def require_engine(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(_MSG.format(p=path))
    return path


class Engine:
    def __init__(self, path: Path) -> None:
        import tensorrt as trt

        self._trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(require_engine(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_in = lambda n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        self.inputs = [n for n in names if is_in(n)]
        self.outputs = [n for n in names if not is_in(n)]

    def _dtype(self, name: str) -> torch.dtype:
        return getattr(torch, np.dtype(self._trt.nptype(self.engine.get_tensor_dtype(name))).name)

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run with CUDA tensors keyed by engine input name; returns CUDA output tensors keyed by name."""
        keep = []
        for n in self.inputs:
            t = inputs[n].to(self._dtype(n)).contiguous()
            keep.append(t)
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outputs:
            outs[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=self._dtype(n), device="cuda")
            self.ctx.set_tensor_address(n, outs[n].data_ptr())
        stream = torch.cuda.current_stream()
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return outs
