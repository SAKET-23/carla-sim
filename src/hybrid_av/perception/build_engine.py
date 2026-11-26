"""TensorRT 10 engine builder (pod only). `tensorrt` is imported inside the function.

Written against the TensorRT 10.x Python API without a GPU to run it; verified in Phase G3.
"""

import hashlib
import json
from pathlib import Path

Shape = tuple[int, ...]


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_gb: float = 4.0,
    profiles: dict[str, tuple[Shape, Shape, Shape]] | None = None,
) -> Path:
    """`profiles` maps a dynamic input name to (min, opt, max) shapes."""
    import tensorrt as trt
    import torch

    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10: networks are always explicit-batch
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if profiles:
        prof = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profiles.items():
            prof.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(prof)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(blob))
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "fp16": fp16,
    }
    engine_path.with_name(engine_path.name + ".json").write_text(json.dumps(meta, indent=2))
    return engine_path
