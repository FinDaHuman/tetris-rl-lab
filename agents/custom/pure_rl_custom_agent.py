from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env.engine import COLS, PIECES, VISIBLE_ROWS
from tetris_env.gym_env import TetrisScoreEnv


MODEL_NAME = "ppo_custom_pure"
PIECE_COUNT = len(PIECES)


class FlatTetrisObservation(gym.ObservationWrapper):
    """Flat observation for pure RL: board, active mask, piece state, current/next IDs."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        size = VISIBLE_ROWS * COLS * 2 + PIECE_COUNT * 2 + 3
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)

    def observation(self, obs):
        current_piece = np.zeros(PIECE_COUNT, dtype=np.float32)
        next_piece = np.zeros(PIECE_COUNT, dtype=np.float32)
        current_piece[int(obs["current_piece"])] = 1.0
        next_piece[int(obs["next_piece"])] = 1.0
        return np.concatenate(
            (
                obs["board"].astype(np.float32).reshape(-1),
                obs["active"].astype(np.float32).reshape(-1),
                obs["piece_state"].astype(np.float32).reshape(-1),
                current_piece,
                next_piece,
            )
        )


def make_env(*, max_pieces: int = 1000, seed: int | None = None) -> gym.Env:
    return FlatTetrisObservation(TetrisScoreEnv(max_pieces=max_pieces, seed=seed))


def _import_sb3():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    return PPO, DummyVecEnv, VecMonitor


def _model_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".zip" else path / f"{MODEL_NAME}.zip"


def train(args: argparse.Namespace) -> None:
    PPO, DummyVecEnv, VecMonitor = _import_sb3()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def build(rank: int):
        def _init():
            return make_env(max_pieces=args.max_pieces, seed=args.seed + rank)

        return _init

    env = VecMonitor(DummyVecEnv([build(rank) for rank in range(args.n_envs)]))
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=args.verbose,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps)
    target = outdir / f"{MODEL_NAME}.zip"
    model.save(target)
    env.close()
    (outdir / "meta.json").write_text(
        json.dumps(
            {
                "track": "custom_pure_rl",
                "algorithm": "PPO",
                "model": str(target),
                "timesteps": args.timesteps,
                "max_pieces": args.max_pieces,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "device": args.device,
                "observation": "locked board + active piece mask + piece state + current piece + one next piece",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {target}")


def evaluate(args: argparse.Namespace) -> None:
    PPO, _, _ = _import_sb3()
    model = PPO.load(_model_path(args.model), device=args.device)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = make_env(max_pieces=args.max_pieces, seed=seed)
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += float(reward)
        row = {
            "seed": seed,
            "reward": total_reward,
            "score": int(info["score"]),
            "lines": int(info["lines"]),
            "pieces": int(info["pieces"]),
        }
        rows.append(row)
        print(row)
        env.close()
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    lines = np.array([row["lines"] for row in rows], dtype=np.float64)
    print(f"mean_score={scores.mean():.1f} max_score={scores.max():.1f} mean_lines={lines.mean():.1f}")


def smoke(args: argparse.Namespace) -> None:
    env = make_env(max_pieces=args.max_pieces, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < args.steps:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += float(reward)
        steps += 1
    print(
        json.dumps(
            {
                "obs_shape": list(obs.shape),
                "steps": steps,
                "reward": total_reward,
                "score": info["score"],
                "lines": info["lines"],
                "pieces": info["pieces"],
            }
        )
    )
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure PPO agent for the custom Tetris Gym environment.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--steps", type=int, default=32)
    smoke_parser.add_argument("--max-pieces", type=int, default=100)
    smoke_parser.add_argument("--seed", type=int, default=0)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--outdir", default="artifacts/custom_pure_rl")
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--max-pieces", type=int, default=1000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--n-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.995)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--verbose", type=int, default=1)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--model", default="artifacts/custom_pure_rl/ppo_custom_pure.zip")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--max-pieces", type=int, default=1000)
    eval_parser.add_argument("--seed", type=int, default=1000)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.cmd == "smoke":
        smoke(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    print(f"elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
