from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env import DEFAULT_WEIGHTS, FEATURE_NAMES, TetrisGame, enumerate_placements, placement_features
from tetris_env.engine import PieceState


def choose_placement(
    game: TetrisGame,
    weights: np.ndarray,
    *,
    lookahead_weight: float = 0.35,
    lookahead_candidates: int = 8,
):
    placements = enumerate_placements(game)
    if not placements:
        return None
    immediate_scores = [
        (float(np.dot(weights, placement_features(placement))), placement)
        for placement in placements
    ]
    immediate_scores.sort(key=lambda item: item[0], reverse=True)
    search = immediate_scores[:lookahead_candidates]

    def value(item):
        immediate, placement = item
        preview_game = game.clone()
        preview_game.board = placement.board.copy()
        preview_game.current = PieceState(game.next_piece, 0, 3, 0)
        next_options = enumerate_placements(preview_game, piece=game.next_piece)
        if not next_options:
            return immediate - 1000.0
        future = max(float(np.dot(weights, placement_features(option))) for option in next_options)
        return immediate + lookahead_weight * future

    return max(search, key=value)[1]


def play_episode(weights: np.ndarray, *, seed: int, max_pieces: int) -> dict[str, int]:
    game = TetrisGame(seed=seed)
    while not game.game_over and game.pieces < max_pieces:
        placement = choose_placement(game, weights)
        if placement is None:
            break
        game.board = placement.board.copy()
        game.score += placement.score_delta
        game.lines += placement.lines
        game.level = game.lines // 10
        game.pieces += 1
        game.spawn()
    return {
        "seed": seed,
        "score": int(game.score),
        "lines": int(game.lines),
        "pieces": int(game.pieces),
    }


def evaluate_weights(weights: np.ndarray, *, seeds: list[int], max_pieces: int) -> float:
    scores = []
    for seed in seeds:
        stats = play_episode(weights, seed=seed, max_pieces=max_pieces)
        scores.append(stats["score"] + stats["lines"] * 1000 + stats["pieces"] * 0.1)
    return float(np.mean(scores))


def load_weights(path: str | None) -> np.ndarray:
    if path is None:
        return DEFAULT_WEIGHTS.copy()
    weights = np.load(path)
    if len(weights) != len(DEFAULT_WEIGHTS):
        raise ValueError(f"expected {len(DEFAULT_WEIGHTS)} weights, got {len(weights)}")
    return weights.astype(np.float64)


def train(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mean = load_weights(args.warm_start)
    std = np.full(len(mean), args.init_std, dtype=np.float64)
    best_weights = mean.copy()
    best_fit = -math.inf
    history = []
    for generation in range(1, args.generations + 1):
        population = mean[None, :] + rng.standard_normal((args.population, len(mean))) * std[None, :]
        seeds = [args.seed * 10000 + generation * 100 + i for i in range(args.rollouts)]
        fitness = np.array([evaluate_weights(weights, seeds=seeds, max_pieces=args.max_pieces) for weights in population])
        elite_count = max(2, int(args.population * args.elite_frac))
        elite = population[np.argsort(fitness)[-elite_count:]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + args.noise_floor
        gen_best_idx = int(np.argmax(fitness))
        if float(fitness[gen_best_idx]) > best_fit:
            best_fit = float(fitness[gen_best_idx])
            best_weights = population[gen_best_idx].copy()
            np.save(outdir / "best_weights.npy", best_weights)
        row = {
            "generation": generation,
            "best_fit": best_fit,
            "gen_best": float(fitness.max()),
            "mean_fit": float(fitness.mean()),
        }
        history.append(row)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"gen {generation:03d}: gen_best={fitness.max():.1f} "
            f"mean={fitness.mean():.1f} best={best_fit:.1f}"
        )
    np.save(outdir / "mean_weights.npy", mean)
    (outdir / "meta.json").write_text(
        json.dumps({"feature_names": FEATURE_NAMES, "best_fit": best_fit, "weights": best_weights.tolist()}, indent=2),
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> None:
    weights = load_weights(args.weights)
    rows = [play_episode(weights, seed=args.seed + i, max_pieces=args.max_pieces) for i in range(args.episodes)]
    for row in rows:
        print(row)
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    lines = np.array([row["lines"] for row in rows], dtype=np.float64)
    print(f"mean_score={scores.mean():.1f} max_score={scores.max():.1f} mean_lines={lines.mean():.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--outdir", default="artifacts/custom_best")
    train_parser.add_argument("--generations", type=int, default=20)
    train_parser.add_argument("--population", type=int, default=32)
    train_parser.add_argument("--elite-frac", type=float, default=0.25)
    train_parser.add_argument("--init-std", type=float, default=4.0)
    train_parser.add_argument("--noise-floor", type=float, default=0.05)
    train_parser.add_argument("--rollouts", type=int, default=4)
    train_parser.add_argument("--max-pieces", type=int, default=3000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--warm-start", default=None)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--weights", default="artifacts/custom_best/best_weights.npy")
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--max-pieces", type=int, default=5000)
    eval_parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    print(f"elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
