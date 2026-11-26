"""Policy checkpoint, JSON export and content hash (policy-mlp.md §6)."""

import hashlib
from pathlib import Path

import torch

from hybrid_av.policy.network import CompactPolicy


def save_policy(policy: CompactPolicy, path: Path, meta: dict) -> None:
    torch.save({"state_dict": policy.state_dict(), "meta": meta}, path)


def load_policy(path: Path) -> tuple[CompactPolicy, dict]:
    ckpt = torch.load(path, weights_only=True)
    policy = CompactPolicy()
    ref = policy.state_dict()
    sd = ckpt["state_dict"]
    if list(sd) != list(ref) or any(sd[k].shape != ref[k].shape for k in ref):
        raise ValueError(f"checkpoint does not match CompactPolicy: {path}")
    policy.load_state_dict(sd)
    return policy, ckpt["meta"]


def policy_to_json(policy: CompactPolicy) -> dict:
    return {k: v.tolist() for k, v in policy.state_dict().items()}


def policy_hash(policy: CompactPolicy) -> str:
    h = hashlib.sha256()
    sd = policy.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes())
    return h.hexdigest()
