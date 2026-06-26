from __future__ import annotations

import gymnasium as gym
import ale_py


gym.register_envs(ale_py)

ENV_ID = "ALE/Tetris-v5"


def make_env(
    *,
    render_mode: str | None = None,
    sticky: float = 0.0,
    obs_type: str = "rgb",
    frameskip: int = 4,
) -> gym.Env:
    return gym.make(
        ENV_ID,
        obs_type=obs_type,
        render_mode=render_mode,
        repeat_action_probability=sticky,
        frameskip=frameskip,
        full_action_space=False,
    )


def estimate_atari_score(lines: int) -> int:
    # Conservative single-line-equivalent score. The ROM may award more for
    # multi-line clears; this keeps render/eval targets transparent.
    return int(lines * 100)


def reward_to_lines(reward: float) -> int:
    return int(round(reward))
