import json
import sys
import types

import pytest

from hybrid_av.perception.build_engine import build_engine


def _install_fakes(monkeypatch, flags):
    calls = []

    class Config:
        def set_memory_pool_limit(self, *a):
            pass

        def set_flag(self, f):
            calls.append(f)

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, f):
            return object()

        def create_builder_config(self):
            return Config()

        def build_serialized_network(self, network, config):
            return b"engine"

    class Parser:
        num_errors = 0

        def __init__(self, network, logger):
            pass

        def parse_from_file(self, p):
            return True

    trt = types.SimpleNamespace(
        Logger=type("Logger", (), {"WARNING": 0, "__init__": lambda self, level: None}),
        Builder=Builder,
        OnnxParser=Parser,
        MemoryPoolType=types.SimpleNamespace(WORKSPACE=0),
        BuilderFlag=types.SimpleNamespace(**flags),
        __version__="fake",
    )
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_name=lambda i: "fake gpu"))
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.mark.parametrize("has_flag", [True, False])
def test_build_engine_fp16_flag(tmp_path, monkeypatch, has_flag):
    calls = _install_fakes(monkeypatch, {"FP16": "FP16"} if has_flag else {"INT8": "INT8"})
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    out = build_engine(onnx, tmp_path / "m.engine", fp16=True)
    meta = json.loads(out.with_name(out.name + ".json").read_text())
    assert calls == (["FP16"] if has_flag else [])
    assert meta["fp16"] is has_flag
