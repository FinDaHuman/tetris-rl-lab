"""Render the best episode of each track to artifacts/best_plays/ as watchable mp4.

Two passes per track: play ``--seeds`` seeded episodes with no capture, rank them
by (lines, score), then re-run the winning seed with a streaming video writer.
The policies are deterministic and the envs are seeded, so pass 2 reproduces
pass 1 exactly.

    python tools/render_best_plays.py --tracks 1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from agents.video import VideoWriter


TRACKS = {
    1: {
        "slug": "track1_ale_pure_rl",
        "env": "ALE/Tetris-v5",
        "method": "Pure RL (PPO, CnnPolicy, pixels)",
        "model": "artifacts/ale_pure_rl/ppo_ale_pure.zip",
    },
    2: {
        "slug": "track2_ale_tool",
        "env": "ALE/Tetris-v5",
        "method": "Tool-assisted (frame decode + placement search + CEM)",
        "model": "artifacts/ale_stable_high_score/best_weights.npy",
    },
    3: {
        "slug": "track3_custom_pure_rl",
        "env": "custom engine",
        "method": "Pure RL (PPO, MlpPolicy, primitive actions)",
        "model": "artifacts/custom_pure_rl/ppo_custom_pure.zip",
    },
    4: {
        "slug": "track4_custom_tool",
        "env": "custom engine",
        "method": "Tool-assisted (placement enumeration + Dellacherie + lookahead + CEM)",
        "model": "artifacts/custom_best/best_weights.npy",
    },
}


def _rank(stats: dict) -> tuple:
    return (stats.get("lines", 0), stats.get("score", 0), stats.get("pieces", stats.get("steps", 0)))


def _best_seed(play, seeds: list[int], label: str) -> dict:
    best = None
    for seed in seeds:
        stats = play(seed)
        print(f"  {label} seed={seed} lines={stats.get('lines')} score={stats.get('score')}")
        if best is None or _rank(stats) > _rank(best):
            best = stats
    return best


def run_track1(args, outdir: Path) -> dict:
    from agents.ale import pure_rl_ale_agent as t1

    PPO = t1._import_sb3()[0]
    model_path = t1._model_path(TRACKS[1]["model"])
    model = PPO.load(model_path, device=args.device)

    def play(seed: int, writer=None) -> dict:
        return t1.play_and_render(model, seed=seed, max_steps=args.max_steps, writer=writer)

    best = _best_seed(play, _seeds(args, 1), "track1")
    out = outdir / f"{TRACKS[1]['slug']}.mp4"
    with VideoWriter(out, fps=args.fps) as writer:
        stats = play(best["seed"], writer=writer)
    return {**stats, "video": out.name}


def run_track2(args, outdir: Path) -> dict:
    from agents.ale import ale_tetris_agent as t2

    weights = t2.load_weights(TRACKS[2]["model"], planner="legacy_model")

    def play(seed: int, frame_sink=None) -> dict:
        stats = t2.play_episode(
            weights,
            seed=seed,
            max_pieces=args.max_pieces_ale,
            planner="legacy_model",
            frame_sink=frame_sink,
        )
        stats.pop("frames", None)
        return stats

    best = _best_seed(play, _seeds(args, 2), "track2")
    out = outdir / f"{TRACKS[2]['slug']}.mp4"
    with VideoWriter(out, fps=args.fps) as writer:
        stats = play(best["seed"], frame_sink=writer.append)
        stats["frames"] = writer.frames
    return {**stats, "video": out.name}


def run_track3(args, outdir: Path) -> dict:
    from agents.custom import pure_rl_custom_agent as t3

    model, normalizer, model_path, stats_path = t3.load_policy(TRACKS[3]["model"], device=args.device)

    def play(seed: int, writer=None) -> dict:
        return t3.play_and_render(
            model, normalizer, seed=seed, max_pieces=args.max_pieces_custom, writer=writer
        )

    best = _best_seed(play, _seeds(args, 3), "track3")
    out = outdir / f"{TRACKS[3]['slug']}.mp4"
    with VideoWriter(out, fps=args.fps) as writer:
        stats = play(best["seed"], writer=writer)
    return {**stats, "video": out.name, "vec_normalize": str(stats_path)}


def run_track4(args, outdir: Path) -> dict:
    from agents.custom import render_custom_episode as t4
    from agents.custom.tetris_custom_agent import load_weights

    weights = load_weights(TRACKS[4]["model"])

    def play(seed: int, writer=None) -> dict:
        return t4.play_and_render(weights, seed=seed, max_pieces=args.max_pieces_custom, writer=writer)

    best = _best_seed(play, _seeds(args, 4), "track4")
    out = outdir / f"{TRACKS[4]['slug']}.mp4"
    with VideoWriter(out, fps=args.fps) as writer:
        stats = play(best["seed"], writer=writer)
    return {**stats, "video": out.name}


RUNNERS = {1: run_track1, 2: run_track2, 3: run_track3, 4: run_track4}


def _seeds(args, track: int) -> list[int]:
    # Track 4 plans 500 pieces with depth-2 lookahead per piece, so it is by far
    # the slowest to search; it gets its own (smaller) seed budget.
    count = args.track4_seeds if track == 4 else args.seeds
    return [args.seed_start + i for i in range(count)]


def write_readme(path: Path, entries: dict[int, dict], args) -> None:
    rows = []
    for track in sorted(entries):
        info = TRACKS[track]
        stats = entries[track]
        rows.append(
            f"| {track} | `{info['slug']}.mp4` | {info['env']} | {info['method']} | "
            f"{stats['seed']} | {stats['lines']} | {stats['score']} | {stats.get('pieces', '-')} | `{info['model']}` |"
        )
    table = "\n".join(rows)
    path.write_text(
        f"""# Best plays

The best episode of each track, rendered as mp4. Generated by
`python tools/render_best_plays.py` -- each track plays a batch of seeded
episodes, the best one (ranked by lines, then score) is replayed with frame
capture. The mp4 files are gitignored; regenerate them with the command above.

| Track | Video | Env | Method | Seed | Lines | Score | Pieces | Model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table}

Rendered at {args.fps} fps. `manifest.json` has the full stats.

## Watch it live

`live_play.py` (in this directory) plays an agent in a window in real time,
running until you close it:

```bash
python artifacts/best_plays/live_play.py            # track 4, endless
python artifacts/best_plays/live_play.py --track 3  # track 3, restarts on top-out
```

Quit with the close button, Esc, or Q. It drives the same agents and the same
renderer as the videos, so nothing is faked for the viewer.

## What you are watching

- **Track 1** clears **0 lines**, and that is the result, not a rendering bug: 10M
  frames of PPO on raw ALE pixels never learned to clear a row. The video shows the
  stack building until the game tops out. This matches `docs/REPORT.md`.
- **Track 2** decodes the Atari frame and searches placements, then drives the piece
  with real ALE controls. It hits the same 37 lines on every seed.
- **Track 3** is the only pure-RL line clearer. It plays with primitive actions
  (left/right/rotate/drop), so you can watch it move each piece, but it only survives
  a few dozen pieces.
- **Track 4** enumerates every placement and scores it with Dellacherie features plus
  a queue lookahead. Pieces are animated into position by replaying the chosen
  placement as real engine actions, so what you see is the board the planner produced.
  It does not top out: it was simulated past 10,000 pieces / 3,997 lines still alive,
  holding ~0.4 lines per piece (the theoretical maximum). Its piece cap exists only
  because a video needs an end -- run `live_play.py` to watch it go indefinitely.

## Score conventions

Scores match each track's own evaluation, which is why they are not directly
comparable across tracks. Tracks 1 and 2 report an estimated Atari score
(100/line). Track 3 reports the engine score, which includes soft/hard-drop
points. Track 4 reports line-clear score only, matching how it was evaluated.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tracks = [int(t) for t in args.tracks.split(",")]

    entries: dict[int, dict] = {}
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        # Keep tracks that were rendered in an earlier (possibly partial) run.
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {int(k): v for k, v in existing.get("tracks", {}).items()}

    for track in tracks:
        print(f"track {track}: searching {len(_seeds(args, track))} seeds")
        stats = RUNNERS[track](args, outdir)
        stats = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in stats.items()}
        entries[track] = {**stats, **{k: TRACKS[track][k] for k in ("env", "method", "model")}}
        print(f"track {track}: wrote {stats['video']} lines={stats['lines']} score={stats['score']}")

    manifest_path.write_text(
        json.dumps(
            {
                "generated_by": "python tools/render_best_plays.py",
                "fps": args.fps,
                "seed_start": args.seed_start,
                "seeds": args.seeds,
                "track4_seeds": args.track4_seeds,
                "tracks": {str(k): entries[k] for k in sorted(entries)},
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    write_readme(outdir / "README.md", entries, args)
    print(f"wrote {manifest_path} and {outdir / 'README.md'}")
    print(f"elapsed={time.time() - start:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the best episode of each track to mp4.")
    parser.add_argument("--tracks", default="1,2,3,4")
    parser.add_argument("--seeds", type=int, default=20, help="seeds to search per track (tracks 1-3)")
    parser.add_argument("--track4-seeds", type=int, default=5, help="seeds to search for the slow track 4 planner")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-pieces-custom", type=int, default=500)
    parser.add_argument("--max-pieces-ale", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default="artifacts/best_plays")
    return parser.parse_args()


if __name__ == "__main__":
    main()
