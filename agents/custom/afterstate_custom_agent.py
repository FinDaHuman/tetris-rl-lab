"""Track 5 — afterstate (placement-level) RL on the custom engine.

The single-variable experiment the four-track study is missing. Track 4 beats
Track 3 by ~190x, but it changes four things at once (action space, hand-authored
features, queue lookahead, CEM instead of PPO). This track changes **only the
action space**: same engine, same `lines` reward, same PPO, same MlpPolicy(2x256),
same VecNormalize, same hyperparameters as the promoted Track 3 run — but one
action places a piece instead of nudging it.

Budget is matched on *pieces experienced*, not env steps: Track 3's 100M primitive
steps at ~8.5 steps/piece is ~12M pieces, so this trains for ~12M placement steps.

Reading the result:
  lands near Track 4 (198 lines) -> the action abstraction carries the gap
  lands near Track 3 (1.04 lines) -> the hand-authored features carry it
Either way it is a real answer to the question the project currently cannot answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env.placement_env import PlacementTetrisEnv


MODEL_NAME = "ppo_custom_afterstate"
VEC_NORMALIZE_NAME = "vec_normalize.pkl"
TITLE = "Track 5 - custom engine\nafterstate RL (PPO, placement actions)"


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


def make_env(*, max_pieces: int = 500, seed: int | None = None, **kwargs):
    return PlacementTetrisEnv(max_pieces=max_pieces, seed=seed, **kwargs)


def make_vector_env(*, n_envs: int, max_pieces: int, seed: int, **kwargs):
    _, _, _, _, DummyVecEnv, VecMonitor, _ = _import_sb3()

    def build(rank: int):
        def _init():
            return make_env(max_pieces=max_pieces, seed=seed + rank, **kwargs)

        return _init

    return VecMonitor(DummyVecEnv([build(rank) for rank in range(n_envs)]))


def train(args: argparse.Namespace) -> None:
    PPO, CallbackList, CheckpointCallback, EvalCallback, _, _, VecNormalize = _import_sb3()
    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    env = make_vector_env(n_envs=args.n_envs, max_pieces=args.max_pieces, seed=args.seed)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma, clip_obs=args.clip_obs)

    eval_env = make_vector_env(n_envs=1, max_pieces=args.max_pieces, seed=args.eval_seed)
    if args.normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, gamma=args.gamma, clip_obs=args.clip_obs)
        eval_env.training = False

    policy_kwargs = {"net_arch": [args.net_width] * args.net_layers}
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(logdir / "tensorboard"),
    )

    # Write the training tables to <logdir>/log.txt and progress.csv at the source.
    # Console/transcript capture is not reliable for long runs (this bit the project
    # once already), so the trainer records its own evidence.
    from stable_baselines3.common.logger import configure as configure_logger

    model.set_logger(configure_logger(str(logdir), ["stdout", "log", "csv", "tensorboard"]))

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(logdir / "checkpoints"),
                name_prefix=MODEL_NAME,
                save_vecnormalize=args.normalize,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(outdir / "best"),
                log_path=str(logdir / "eval"),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            ),
        ]
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - start

    target = outdir / f"{MODEL_NAME}.zip"
    model.save(target)
    if args.normalize:
        env.save(str(outdir / VEC_NORMALIZE_NAME))
    (outdir / "meta.json").write_text(
        json.dumps(
            {
                "track": "custom_afterstate",
                "algorithm": "PPO",
                "action_space": "placement (rotation x column), one action per piece",
                "observation": "raw visible board + current piece + one next piece (NO hand features)",
                "lookahead": "none (one-piece preview only, same as Track 3)",
                "timesteps": args.timesteps,
                "max_pieces": args.max_pieces,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "net_layers": args.net_layers,
                "net_width": args.net_width,
                "normalize": args.normalize,
                "elapsed_hours": round(elapsed / 3600, 2),
                "purpose": "isolates the action abstraction vs Track 3 (all else held equal)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {target}  elapsed={elapsed/3600:.2f}h")


def load_policy(model_path, *, vec_normalize=None, device: str = "cpu", normalize: bool = True):
    PPO, _, _, _, _, _, VecNormalize = _import_sb3()
    model_path = _model_path(model_path)
    model = PPO.load(model_path, device=device)
    stats_path = _stats_path(model_path, vec_normalize)
    normalizer = None
    if normalize:
        if stats_path.exists():
            probe = make_vector_env(n_envs=1, max_pieces=10, seed=0)
            normalizer = VecNormalize.load(stats_path, probe)
            normalizer.training = False
            normalizer.norm_reward = False
        else:
            print(f"warning: VecNormalize stats not found at {stats_path}")
    return model, normalizer, model_path, stats_path


def play_and_render(model, normalizer, *, seed: int, max_pieces: int, deterministic: bool = True, writer=None) -> dict:
    """One episode. If a writer is given, animate each placement as real engine actions."""
    from tetris_env.render import render_frame
    from tetris_env.replay import placement_actions, placement_drop_states
    from tetris_env.engine import enumerate_placements

    env = make_env(max_pieces=max_pieces, seed=seed)
    obs, info = env.reset(seed=seed)
    game = env.game
    # Line-clear score only. PlacementTetrisEnv teleports the board and credits
    # placement.score_delta, but the animated path below drives the engine with a
    # real HARD_DROP, which also credits drop points. Tracking the score here (and
    # writing it back to the engine) keeps both paths on the evaluation convention.
    score = 0

    def draw(flash: bool = False, hold: int = 1) -> None:
        if writer is not None:
            writer.append(render_frame(game, title=TITLE, seed=seed, score=score, flash=flash), hold=hold)

    draw(hold=8)
    terminated = truncated = False
    while not (terminated or truncated):
        batch = obs[None, :]
        if normalizer is not None:
            batch = normalizer.normalize_obs(batch)
        action, _ = model.predict(batch, deterministic=deterministic)

        if writer is not None:
            # Animate the placement the policy is about to commit, so the video shows
            # real engine play rather than a teleport (same approach as Track 4).
            placements = enumerate_placements(game)
            if placements:
                chosen = env._resolve(int(action[0]), placements)
                actions = placement_actions(game, chosen)
                lines_before = game.lines
                if actions is not None:
                    for act in actions:
                        game.step(act)
                        draw()
                    score += chosen.score_delta
                    game.score = score  # drop points are not part of the eval score
                    obs, _, terminated, truncated, info = (
                        env._obs(),
                        0.0,
                        game.game_over,
                        game.pieces >= max_pieces,
                        env._info(),
                    )
                    if game.lines > lines_before:
                        draw(flash=True, hold=2)
                    continue
                for state in placement_drop_states(game, chosen):
                    game.current = state
                    draw()

        lines_before = game.lines
        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        score = game.score
        draw()
        if game.lines > lines_before:
            draw(flash=True, hold=2)
    draw(hold=20)
    return {
        "seed": seed,
        "score": int(score),
        "lines": int(info["lines"]),
        "pieces": int(info["pieces"]),
        "frames": 0 if writer is None else writer.frames,
    }


def evaluate(args: argparse.Namespace) -> None:
    model, normalizer, model_path, stats_path = load_policy(
        args.model, vec_normalize=args.vec_normalize, device=args.device, normalize=args.normalize
    )
    rows = []
    for episode in range(args.episodes):
        row = play_and_render(
            model, normalizer, seed=args.seed + episode, max_pieces=args.max_pieces,
            deterministic=args.deterministic,
        )
        row.pop("frames", None)
        rows.append(row)
        print(row)
    lines = np.array([r["lines"] for r in rows], dtype=np.float64)
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    pieces = np.array([r["pieces"] for r in rows], dtype=np.float64)
    summary = {
        "track": "custom_afterstate",
        "model": str(model_path),
        "vec_normalize": str(stats_path) if stats_path.exists() else None,
        "episodes": args.episodes,
        "seed_start": args.seed,
        "max_pieces": args.max_pieces,
        "deterministic": args.deterministic,
        "mean_lines": float(lines.mean()),
        "max_lines": int(lines.max()),
        "min_lines": int(lines.min()),
        "mean_score": float(scores.mean()),
        "mean_pieces": float(pieces.mean()),
        "rows": rows,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    print(
        f"mean_lines={lines.mean():.2f} max_lines={lines.max():.0f} "
        f"mean_pieces={pieces.mean():.1f} mean_score={scores.mean():.0f}"
    )
    print("compare: Track 3 (primitive actions) = 1.04 mean lines | Track 4 (features+search) = 198.1")


def render(args: argparse.Namespace) -> None:
    from agents.video import VideoWriter

    model, normalizer, _, _ = load_policy(args.model, device=args.device, normalize=args.normalize)
    out = Path(args.out)
    with VideoWriter(out, fps=args.fps) as writer:
        stats = play_and_render(
            model, normalizer, seed=args.seed, max_pieces=args.max_pieces, writer=writer
        )
    print(json.dumps(stats))
    print(f"wrote {out}")


def smoke(args: argparse.Namespace) -> None:
    env = make_env(max_pieces=args.max_pieces, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    total = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < args.steps:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += float(reward)
        steps += 1
    print(
        json.dumps(
            {
                "obs_shape": list(obs.shape),
                "action_space": int(env.action_space.n),
                "steps": steps,
                "reward": total,
                "score": info["score"],
                "lines": info["lines"],
                "pieces": info["pieces"],
                "note": "random placements; a trained policy should do far better",
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track 5: afterstate (placement-level) PPO on the custom engine.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("smoke")
    sp.add_argument("--steps", type=int, default=200)
    sp.add_argument("--max-pieces", type=int, default=500)
    sp.add_argument("--seed", type=int, default=0)

    tp = sub.add_parser("train")
    tp.add_argument("--outdir", default="runs/track5_afterstate")
    tp.add_argument("--logdir", default="runs/track5_afterstate_logs")
    tp.add_argument("--timesteps", type=int, default=12_000_000)
    tp.add_argument("--n-envs", type=int, default=8)
    tp.add_argument("--max-pieces", type=int, default=500)
    tp.add_argument("--seed", type=int, default=7)
    tp.add_argument("--n-steps", type=int, default=512)
    tp.add_argument("--batch-size", type=int, default=256)
    tp.add_argument("--n-epochs", type=int, default=10)
    tp.add_argument("--learning-rate", type=float, default=3e-4)
    tp.add_argument("--gamma", type=float, default=0.995)
    tp.add_argument("--gae-lambda", type=float, default=0.95)
    tp.add_argument("--clip-range", type=float, default=0.2)
    tp.add_argument("--ent-coef", type=float, default=0.01)
    tp.add_argument("--vf-coef", type=float, default=0.5)
    tp.add_argument("--net-layers", type=int, default=2)
    tp.add_argument("--net-width", type=int, default=256)
    tp.add_argument("--clip-obs", type=float, default=10.0)
    tp.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    tp.add_argument("--device", default="cpu")
    tp.add_argument("--verbose", type=int, default=1)
    tp.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    tp.add_argument("--eval-freq", type=int, default=250_000)
    tp.add_argument("--eval-episodes", type=int, default=5)
    tp.add_argument("--eval-seed", type=int, default=10_000)

    ep = sub.add_parser("evaluate")
    ep.add_argument("--model", default="runs/track5_afterstate/ppo_custom_afterstate.zip")
    ep.add_argument("--episodes", type=int, default=25)
    ep.add_argument("--max-pieces", type=int, default=500)
    ep.add_argument("--seed", type=int, default=1000)
    ep.add_argument("--device", default="cpu")
    ep.add_argument("--deterministic", action="store_true", default=True)
    ep.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    ep.add_argument("--vec-normalize", default=None)
    ep.add_argument("--out", default=None)

    rp = sub.add_parser("render")
    rp.add_argument("--model", default="runs/track5_afterstate/ppo_custom_afterstate.zip")
    rp.add_argument("--seed", type=int, default=1000)
    rp.add_argument("--max-pieces", type=int, default=500)
    rp.add_argument("--device", default="cpu")
    rp.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    rp.add_argument("--fps", type=int, default=15)
    rp.add_argument("--out", default="artifacts/best_plays/track5_custom_afterstate.mp4")
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
    elif args.cmd == "render":
        render(args)
    print(f"elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
