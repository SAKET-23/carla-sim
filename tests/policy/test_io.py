import pytest
import torch

from hybrid_av.policy.io import load_policy, policy_hash, policy_to_json, save_policy
from hybrid_av.policy.network import CompactPolicy


def test_roundtrip(tmp_path):
    p = CompactPolicy()
    save_policy(p, tmp_path / "p.pt", {"step": 5})
    q, meta = load_policy(tmp_path / "p.pt")
    assert meta["step"] == 5 and policy_hash(p) == policy_hash(q)


def test_hash_detects_single_weight():
    p = CompactPolicy()
    h = policy_hash(p)
    with torch.no_grad():
        p.fc2.weight[3, 7] += 1e-6
    assert policy_hash(p) != h


def test_json_shapes():
    j = policy_to_json(CompactPolicy())
    assert len(j["fc1.weight"]) == 64 and len(j["fc1.weight"][0]) == 24 and len(j["fc3.bias"]) == 3


def test_rejects_wrong_architecture(tmp_path):
    torch.save({"state_dict": torch.nn.Linear(24, 3).state_dict(), "meta": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_policy(tmp_path / "bad.pt")
