from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PURE_RL_ALE = ROOT / "agents" / "ale" / "pure_rl_ale_agent.py"


def test_pure_rl_ale_does_not_import_tool_assisted_agent() -> None:
    tree = ast.parse(PURE_RL_ALE.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert "agents.ale.ale_tetris_agent" not in imports


def test_pure_rl_ale_has_no_tool_assisted_primitives() -> None:
    source = PURE_RL_ALE.read_text(encoding="utf-8")
    forbidden = (
        "decode_board",
        "find_falling_piece",
        "enumerate_model_placements",
        "cloneState",
        "restoreState",
        "placement",
        "DEFAULT_WEIGHTS",
    )
    for token in forbidden:
        assert token not in source
