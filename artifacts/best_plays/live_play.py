"""Watch an agent play live, in a window, until you stop it.

    python artifacts/best_plays/live_play.py              # track 4 (tool-assisted, endless)
    python artifacts/best_plays/live_play.py --track 3    # track 3 (pure RL, restarts on top-out)

Quit with the window's close button, Esc, or Q (Ctrl-C in the terminal also works).

The agent runs on a background thread and pushes frames into a bounded queue that
the Tk event loop drains at --fps. The queue is what paces it: the tool-assisted
planner thinks far faster than 15 frames/s, so a blocking put keeps it in step
with the display instead of racing ahead. This reuses the same play_and_render
loops (and the same renderer) that produce the mp4s, so what you watch live is
exactly what gets recorded.

Track 4 never tops out -- it holds ~0.4 lines/piece indefinitely -- so it runs
until you close it. Track 4 is capped at 500 pieces in the video only because a
video needs an end; there is no such cap here.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

ENDLESS = 10**9


class ViewerClosed(Exception):
    """Raised inside the agent thread to unwind it when the window goes away."""


class QueueWriter:
    """Writer-shaped sink (same .append/.frames contract as agents.video.VideoWriter)."""

    def __init__(self, frames: queue.Queue, stop: threading.Event):
        self._queue = frames
        self._stop = stop
        self.frames = 0

    def append(self, frame, *, hold: int = 1) -> None:
        for _ in range(hold):
            if self._stop.is_set():
                raise ViewerClosed
            while True:
                if self._stop.is_set():
                    raise ViewerClosed
                try:
                    # Blocking put with a timeout, so a closed window is noticed
                    # even while the consumer is not draining.
                    self._queue.put(frame, timeout=0.2)
                    break
                except queue.Full:
                    continue
            self.frames += 1


def make_agent(args):
    """Return a callable(seed, writer) that plays one episode."""
    if args.track == 4:
        from agents.custom import render_custom_episode as t4
        from agents.custom.tetris_custom_agent import load_weights

        weights = load_weights(str(ROOT / "artifacts/custom_best/best_weights.npy"))

        def play(seed: int, writer) -> dict:
            return t4.play_and_render(weights, seed=seed, max_pieces=ENDLESS, writer=writer)

        return play

    from agents.custom import pure_rl_custom_agent as t3

    model, normalizer, _, _ = t3.load_policy(str(ROOT / "artifacts/custom_pure_rl/ppo_custom_pure.zip"))

    def play(seed: int, writer) -> dict:
        return t3.play_and_render(model, normalizer, seed=seed, max_pieces=args.max_pieces, writer=writer)

    return play


def agent_thread(play, frames: queue.Queue, stop: threading.Event, seed: int) -> None:
    episode = 0
    try:
        while not stop.is_set():
            stats = play(seed + episode, QueueWriter(frames, stop))
            print(
                f"episode {episode} finished: lines={stats['lines']} "
                f"score={stats['score']} pieces={stats['pieces']}",
                flush=True,
            )
            episode += 1
    except ViewerClosed:
        pass


def main() -> None:
    args = parse_args()

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as exc:  # tkinter is stdlib but can be missing on stripped installs
        raise SystemExit(f"live viewer needs tkinter and pillow: {exc}")

    print(f"loading track {args.track} agent...", flush=True)
    play = make_agent(args)

    frames: queue.Queue = queue.Queue(maxsize=args.buffer)
    stop = threading.Event()
    worker = threading.Thread(target=agent_thread, args=(play, frames, stop, args.seed), daemon=True)

    root = tk.Tk()
    root.title(f"Tetris RL Lab - track {args.track} live")
    root.configure(bg="#0a0a0c")
    label = tk.Label(root, bg="#0a0a0c")
    label.pack()

    def shutdown(*_) -> None:
        stop.set()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", shutdown)
    root.bind("<Escape>", shutdown)
    root.bind("q", shutdown)
    signal.signal(signal.SIGINT, shutdown)

    delay = max(1, round(1000 / args.fps))

    def tick() -> None:
        if stop.is_set():
            return
        try:
            frame = frames.get_nowait()
        except queue.Empty:
            root.after(10, tick)  # agent still thinking; check back shortly
            return
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        label.configure(image=photo)
        label.image = photo  # keep a reference or Tk garbage-collects the image
        root.after(delay, tick)

    worker.start()
    root.after(0, tick)
    print(f"playing at {args.fps} fps - close the window, or press Esc/Q, to stop", flush=True)
    try:
        root.mainloop()
    finally:
        stop.set()
    print("stopped", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained agent play Tetris live.")
    parser.add_argument("--track", type=int, choices=(3, 4), default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--buffer", type=int, default=60, help="frames buffered ahead of the display")
    parser.add_argument("--max-pieces", type=int, default=ENDLESS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
