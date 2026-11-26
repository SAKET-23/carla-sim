"""Editors: scripted, manual, random-sparse control and Claude (anthropic imported lazily)."""

from hybrid_av.weightspace.editors.base import Attempt, Editor
from hybrid_av.weightspace.schema import WeightSpaceConfig

__all__ = ["Attempt", "Editor", "make_editor"]


def make_editor(name: str, cfg: WeightSpaceConfig, **kw) -> Editor:
    if name == "scripted":
        from hybrid_av.weightspace.editors.scripted import ScriptedEditor

        return ScriptedEditor(**kw)
    if name == "manual":
        from hybrid_av.weightspace.editors.manual import ManualEditor

        return ManualEditor(**kw)
    if name == "random_sparse":
        from hybrid_av.weightspace.editors.random_sparse import RandomSparseEditor

        return RandomSparseEditor(**kw)
    if name == "anthropic":
        from hybrid_av.weightspace.editors.anthropic_editor import AnthropicEditor

        if cfg.editor_client == "claude_code" and "client" not in kw:
            from hybrid_av.weightspace.editors.claude_code_client import ClaudeCodeClient

            kw["client"] = ClaudeCodeClient()
        return AnthropicEditor(cfg, **kw)
    raise ValueError(f"unknown editor {name!r}")
