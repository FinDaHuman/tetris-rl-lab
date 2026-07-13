"""Streaming mp4 writer shared by the track renderers.

The existing renderers accumulated every frame in a list and called
``imageio.mimsave`` once. That is fine for a 20-frame slideshow and fatal for a
real animation: the tool-assisted custom episode runs ~500 pieces at ~15 frames
each, and a 440x480 RGB frame is ~0.6 MB, so the list alone would be several GB.
Frames are encoded as they are produced instead.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class VideoWriter:
    """Context manager wrapping an imageio ffmpeg writer.

    ``macro_block_size=None`` keeps the exact frame size instead of silently
    resizing up to a multiple of 16; the renderers already emit even dimensions,
    which is all libx264 needs.
    """

    def __init__(self, path: str | Path, *, fps: int = 15, quality: int = 8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames = 0
        self._writer = imageio.get_writer(
            self.path,
            fps=fps,
            quality=quality,
            macro_block_size=None,
        )

    def append(self, frame: np.ndarray, *, hold: int = 1) -> None:
        for _ in range(hold):
            self._writer.append_data(frame)
            self.frames += 1

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
