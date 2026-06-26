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
VEC_NORMALIZE_NAME = "vec_normalize.pkl"
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
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    return PPO, CallbackList, CheckpointCallback, EvalCallback, DummyVecEnv, VecMonitor, VecNormalize


def _model_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".zip" else path / f"{MODEL_NAME}.zip"


def _stats_path(model: str | Path, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    path = Path(model)
    base = path.parent if path.suffix == ".zip" else path
    return base / VEC_NORMALIZE_NAME


def make_vector_env(*, n_envs: int, max_pieces: int, seed: int):
    _, _, _, _, DummyVecEnv, VecMonitor, _ = _import_sb3()

    def build(rank: int):
        def _init():
            return make_env(max_pieces=max_pieces, seed=seed + rank)

        return _init

    return VecMonitor(DummyVecEnv([build(rank) for rank in range(n_envs)]))


def apply_vec_normalize(env, *, enabled: bool, gamma: float, clip_obs: float):
    _, _, _, _, _, _, VecNormalize = _import_sb3()
    if not enabled:
        return env
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=clip_obs, gamma=gamma)


def train(args: argparse.Namespace) -> None:
    PPO, CallbackList, CheckpointCallback, EvalCallback, _, _, VecNormalize = _import_sb3()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    env = make_vector_env(n_envs=args.n_envs, max_pieces=args.max_pieces, seed=args.seed)
    env = apply_vec_normalize(env, enabled=args.normalize, gamma=args.gamma, clip_obs=args.clip_obs)
    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // args.n_envs),
                save_path=str(logdir / "checkpoints"),
                name_prefix=MODEL_NAME,
                save_vecnormalize=args.normalize,
            )
        )
    if args.eval_freq > 0:
        eval_env = make_vector_env(n_envs=1, max_pieces=args.max_pieces, seed=args.eval_seed)
        if args.normalize:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=args.clip_obs, gamma=args.gamma)
            eval_env.training = False
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
    policy_kwargs = {
        "net_arch": {
            "pi": [args.net_width] * args.net_layers,
            "vf": [args.net_width] * args.net_layers,
        }
    }
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=args.verbose,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        device=args.device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(logdir / "tensorboard"),
    )
    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=args.timesteps, callback=callback)
    target = outdir / f"{MODEL_NAME}.zip"
    model.save(target)
    if args.normalize:
        env.save(outdir / VEC_NORMALIZE_NAME)
    env.close()
    if args.eval_freq > 0:
        eval_env.close()
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
                "normalize": args.normalize,
                "vec_normalize": str(outdir / VEC_NORMALIZE_NAME) if args.normalize else None,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
                "net_layers": args.net_layers,
                "net_width": args.net_width,
                "observation": "locked board + active piece mask + piece state + current piece + one next piece",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {target}")


def evaluate(args: argparse.Namespace) -> None:
    PPO, _, _, _, _, _, VecNormalize = _import_sb3()
    model_path = _model_path(args.model)
    env = make_vector_env(n_envs=1, max_pieces=args.max_pieces, seed=args.seed)
    stats_path = _stats_path(model_path, args.vec_normalize)
    if args.normalize and stats_path.exists():
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    elif args.normalize:
        print(f"warning: VecNormalize stats not found at {stats_path}; evaluating without normalization")
    model = PPO.load(model_path, device=args.device)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        if episode > 0:
            env.seed(seed)
        obs = env.reset()
        total_reward = 0.0
        done = np.array([False])
        last_info = {}
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0]
        row = {
            "seed": seed,
            "reward": total_reward,
            "score": int(last_info["score"]),
            "lines": int(last_info["lines"]),
            "pieces": int(last_info["pieces"]),
        }
        rows.append(row)
        print(row)
    env.close()
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    lines = np.array([row["lines"] for row in rows], dtype=np.float64)
    summary = {
        "model": str(model_path),
        "vec_normalize": str(stats_path) if stats_path.exists() else None,
        "episodes": args.episodes,
        "seed_start": args.seed,
        "max_pieces": args.max_pieces,
        "deterministic": args.deterministic,
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max()),
        "mean_lines": float(lines.mean()),
        "max_lines": int(lines.max()),
        "rows": rows,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")
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
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.995)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--vf-coef", type=float, default=0.5)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--verbose", type=int, default=1)
    train_parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--clip-obs", type=float, default=10.0)
    train_parser.add_argument("--net-layers", type=int, default=2)
    train_parser.add_argument("--net-width", type=int, default=256)
    train_parser.add_argument("--logdir", default="runs/custom_pure_rl")
    train_parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    train_parser.add_argument("--eval-freq", type=int, default=250_000)
    train_parser.add_argument("--eval-episodes", type=int, default=5)
    train_parser.add_argument("--eval-seed", type=int, default=10_000)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--model", default="artifacts/custom_pure_rl/ppo_custom_pure.zip")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--max-pieces", type=int, default=1000)
    eval_parser.add_argument("--seed", type=int, default=1000)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--deterministic", action="store_true")
    eval_parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    eval_parser.add_argument("--vec-normalize", default=None)
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
