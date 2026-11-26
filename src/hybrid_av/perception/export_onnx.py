"""ONNX export with the legacy TorchScript exporter, followed by onnx.checker."""

from pathlib import Path

import onnx
import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    opset: int = 17,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(),
        example_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))
    return path
