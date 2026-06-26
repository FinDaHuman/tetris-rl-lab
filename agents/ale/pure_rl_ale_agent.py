from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ale.env import ENV_ID, estimate_atari_score, make_env, reward_to_lines


MODEL_NAME = "ppo_ale_pure"


def _import_sb3():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

    return PPO, CallbackList, CheckpointCallback, EvalCallback, DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage


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
    preprocessing: str = "atari",
    screen_size: int = 84,
    noop_max: int = 30,
    render_mode: str | None = None,
):
    _, _, _, _, DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage = _import_sb3()

    def build(_rank: int):
        def _init():
            env = make_env(
                render_mode=render_mode,
                sticky=sticky,
                obs_type="rgb",
                frameskip=1 if preprocessing == "atari" else 4,
            )
            if preprocessing == "atari":
                env = AtariPreprocessing(
                    env,
                    noop_max=noop_max,
                    frame_skip=4,
                    screen_size=screen_size,
                    terminal_on_life_loss=False,
                    grayscale_obs=True,
                    grayscale_newaxis=True,
                    scale_obs=False,
                )
            return env

        return _init

    env_cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    env = env_cls([build(rank) for rank in range(n_envs)])
    env.seed(seed)
    env = VecMonitor(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="last")
    return VecTransposeImage(env)


def train(args: argparse.Namespace) -> None:
    PPO, CallbackList, CheckpointCallback, EvalCallback, _, _, _, _, _ = _import_sb3()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    env = make_vector_env(
        n_envs=args.n_envs,
        sticky=args.sticky,
        seed=args.seed,
        frame_stack=args.frame_stack,
        vec_env=args.vec_env,
        preprocessing=args.preprocessing,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
    )
    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // args.n_envs),
                save_path=str(logdir / "checkpoints"),
                name_prefix=MODEL_NAME,
            )
        )
    if args.eval_freq > 0:
        eval_env = make_vector_env(
            n_envs=1,
            sticky=args.sticky,
            seed=args.eval_seed,
            frame_stack=args.frame_stack,
            preprocessing=args.preprocessing,
            screen_size=args.screen_size,
            noop_max=args.noop_max,
        )
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(outdir / "best"),
                log_path=str(logdir / "eval"),
                eval_freq=max(1, args.eval_freq // args.n_envs),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
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
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        device=args.device,
        tensorboard_log=str(logdir / "tensorboard"),
    )
    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=args.timesteps, callback=callback)
    target = outdir / f"{MODEL_NAME}.zip"
    model.save(target)
    env.close()
    if args.eval_freq > 0:
        eval_env.close()
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
                "preprocessing": args.preprocessing,
                "screen_size": args.screen_size,
                "noop_max": args.noop_max,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
                "seed": args.seed,
                "observation": "Atari frames only; optional standard Atari preprocessing, no board decoding or planning",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {target}")


def evaluate(args: argparse.Namespace) -> None:
    PPO, _, _, _, _, _, _, _, _ = _import_sb3()
    model = PPO.load(_model_path(args.model), device=args.device)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = make_vector_env(
            n_envs=1,
            sticky=args.sticky,
            seed=seed,
            frame_stack=args.frame_stack,
            preprocessing=args.preprocessing,
            screen_size=args.screen_size,
            noop_max=args.noop_max,
        )
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
        "preprocessing": args.preprocessing,
        "screen_size": args.screen_size,
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
    env: gym.Env = make_env(
        render_mode=None,
        sticky=args.sticky,
        obs_type="rgb",
        frameskip=1 if args.preprocessing == "atari" else 4,
    )
    if args.preprocessing == "atari":
        env = AtariPreprocessing(
            env,
            noop_max=args.noop_max,
            frame_skip=4,
            screen_size=args.screen_size,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=False,
        )
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
                "preprocessing": args.preprocessing,
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
    smoke_parser.add_argument("--preprocessing", choices=("atari", "raw"), default="atari")
    smoke_parser.add_argument("--screen-size", type=int, default=84)
    smoke_parser.add_argument("--noop-max", type=int, default=30)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--outdir", default="artifacts/ale_pure_rl")
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--sticky", type=float, default=0.25)
    train_parser.add_argument("--frame-stack", type=int, default=4)
    train_parser.add_argument("--preprocessing", choices=("atari", "raw"), default="atari")
    train_parser.add_argument("--screen-size", type=int, default=84)
    train_parser.add_argument("--noop-max", type=int, default=30)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--n-steps", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-range", type=float, default=0.1)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--vf-coef", type=float, default=0.5)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--verbose", type=int, default=1)
    train_parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
    train_parser.add_argument("--logdir", default="runs/ale_pure_rl")
    train_parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    train_parser.add_argument("--eval-freq", type=int, default=250_000)
    train_parser.add_argument("--eval-episodes", type=int, default=5)
    train_parser.add_argument("--eval-seed", type=int, default=10_000)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--model", default="artifacts/ale_pure_rl/ppo_ale_pure.zip")
    eval_parser.add_argument("--episodes", type=int, default=25)
    eval_parser.add_argument("--max-steps", type=int, default=20_000)
    eval_parser.add_argument("--sticky", type=float, default=0.25)
    eval_parser.add_argument("--frame-stack", type=int, default=4)
    eval_parser.add_argument("--preprocessing", choices=("atari", "raw"), default="atari")
    eval_parser.add_argument("--screen-size", type=int, default=84)
    eval_parser.add_argument("--noop-max", type=int, default=30)
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
