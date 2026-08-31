import sys
import types

import torch

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForImageTextToText = object
    transformers_stub.AutoProcessor = object
    sys.modules["transformers"] = transformers_stub

from inference import _resolve_device_dtype


def test_cpu_half_precision_falls_back_to_float32():
    device, dtype = _resolve_device_dtype("cpu", torch.float16)
    assert device == "cpu"
    assert dtype == torch.float32


def test_cuda_half_precision_falls_back_when_unsupported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (5, 0))

    device, dtype = _resolve_device_dtype("cuda", torch.bfloat16)
    assert device == "cuda"
    assert dtype == torch.float32
