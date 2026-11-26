"""PyTorch segmentation backend (reference path for the TensorRT one)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hybrid_av.perception.classes import TRAIN_CLASSES


def preprocess(rgb: np.ndarray, mean, std, input_hw, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1)[None].float() / 255.0
    x = F.interpolate(x, size=tuple(input_hw), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


def postprocess(logits: torch.Tensor, size_hw: tuple[int, int]) -> np.ndarray:
    ids = logits.argmax(dim=1, keepdim=True).float()  # (1, 1, h, w); class map resized with nearest neighbour
    return F.interpolate(ids, size=size_hw, mode="nearest")[0, 0].to(torch.uint8).cpu().numpy()


class TorchSegmenter:
    classes = TRAIN_CLASSES

    def __init__(self, model: nn.Module, mean, std, input_hw, device: str = "cpu") -> None:
        self.model = model.eval().to(device)
        self.mean, self.std, self.input_hw, self.device = mean, std, input_hw, device

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(preprocess(rgb, self.mean, self.std, self.input_hw, self.device))
        return postprocess(logits, rgb.shape[:2])


def load_segmentation_model(cfg: dict) -> nn.Module:
    """Build `cfg["name"]` (segmentation.md) and load `cfg["weights"]`. Backends are imported lazily; both are
    unverified on the Mac (PIDNet repo on PYTHONPATH, HF transformers on the pod)."""
    name = cfg["name"]
    if name == "pidnet_s":
        from models.pidnet import get_pred_model  # PIDNet repo (XuJiacong/PIDNet), not a dependency

        model = get_pred_model("pidnet_s", len(TRAIN_CLASSES))
        model.load_state_dict(torch.load(cfg["weights"], map_location="cpu"))
        return model
    if name == "segformer_b0":
        from transformers import SegformerForSemanticSegmentation

        hf = SegformerForSemanticSegmentation.from_pretrained(cfg["weights"], num_labels=len(TRAIN_CLASSES))
        return _Logits(hf)
    raise ValueError(
        f"unsupported torch segmentation model {name!r} (mask2former_swin_t runs through TAO, not torch)"
    )


class _Logits(nn.Module):
    def __init__(self, hf: nn.Module) -> None:
        super().__init__()
        self.hf = hf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf(pixel_values=x).logits


def build(cfg: dict) -> TorchSegmenter:
    return TorchSegmenter(
        load_segmentation_model(cfg), cfg["mean"], cfg["std"], cfg["input_hw"], cfg.get("device", "cpu")
    )
