from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ale.ale_tetris_agent import ENV_ID, estimate_atari_score, make_env, reward_to_lines


MODEL_NAME = "ppo_ale_pure"


def _import_sb3():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

    return PPO, DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage


def _model_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".zip" else path / f"{MODEL_NAME}.zip"


def make_vector_env(
    *,
    n_envs: int,
    sticky: float,
    seed: int,
    frame_stack: int,
    vec_env: str = "dummy",
    render_mode: str | None = None,
):
    _, DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage = _import_sb3()

    def build(_rank: int):
        def _init():
            return make_env(render_mode=render_mode, sticky=sticky, obs_type="rgb")

        return _init

    env_cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    env = env_cls([build(rank) for rank in range(n_envs)])
    env.seed(seed)
    env = VecMonitor(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="last")
    return VecTransposeImage(env)


def train(args: argparse.Namespace) -> None:
    PPO, _, _, _, _, _ = _import_sb3()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = make_vector_env(
        n_envs=args.n_envs,
        sticky=args.sticky,
        seed=args.seed,
        frame_stack=args.frame_stack,
        vec_env=args.vec_env,
    )
    model = PPO(
        "CnnPolicy",
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
                "track": "ale_pure_rl",
                "env": ENV_ID,
                "algorithm": "PPO",
                "model": str(target),
                "timesteps": args.timesteps,
                "sticky": args.sticky,
                "frame_stack": args.frame_stack,
                "n_envs": args.n_envs,
                "vec_env": args.vec_env,
                "seed": args.seed,
                "observation": "raw Atari RGB frames only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {target}")


def evaluate(args: argparse.Namespace) -> None:
    PPO, _, _, _, _, _ = _import_sb3()
    model = PPO.load(_model_path(args.model), device=args.device)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = make_vector_env(n_envs=1, sticky=args.sticky, seed=seed, frame_stack=args.frame_stack)
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        steps = 0
        last_info = {}
        while not bool(done[0]) and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0]
            steps += 1
        env.close()
        lines = reward_to_lines(total_reward)
        row = {
            "seed": seed,
            "native_reward": total_reward,
            "estimated_lines": lines,
            "estimated_score": estimate_atari_score(lines),
            "steps": steps,
            "terminated": bool(done[0]),
            "episode": last_info.get("episode"),
        }
        rows.append(row)
        print(row)
    rewards = np.array([row["native_reward"] for row in rows], dtype=np.float64)
    summary = {
        "model": str(_model_path(args.model)),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "sticky": args.sticky,
        "frame_stack": args.frame_stack,
        "deterministic": args.deterministic,
        "mean_native_reward": float(rewards.mean()),
        "max_native_reward": float(rewards.max()),
        "rows": rows,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    print(f"mean_native_reward={rewards.mean():.2f} max_native_reward={rewards.max():.2f}")


def smoke(args: argparse.Namespace) -> None:
    env = make_env(render_mode=None, sticky=args.sticky, obs_type="rgb")
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < args.steps:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += float(reward)
        steps += 1
    env.close()
    print(
        json.dumps(
            {
                "env": ENV_ID,
                "obs_shape": list(obs.shape),
                "steps": steps,
                "sticky": args.sticky,
                "native_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure PPO agent for ALE/Tetris-v5.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--steps", type=int, default=64)
    smoke_parser.add_argument("--sticky", type=float, default=0.25)
    smoke_parser.add_argument("--seed", type=int, default=0)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--outdir", default="artifacts/ale_pure_rl")
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--sticky", type=float, default=0.25)
    train_parser.add_argument("--frame-stack", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--n-steps", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--verbose", type=int, default=1)
    train_parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--model", default="artifacts/ale_pure_rl/ppo_ale_pure.zip")
    eval_parser.add_argument("--episodes", type=int, default=25)
    eval_parser.add_argument("--max-steps", type=int, default=20_000)
    eval_parser.add_argument("--sticky", type=float, default=0.25)
    eval_parser.add_argument("--frame-stack", type=int, default=4)
    eval_parser.add_argument("--seed", type=int, default=1000)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--deterministic", action="store_true")
    eval_parser.add_argument("--out", default=None)
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
