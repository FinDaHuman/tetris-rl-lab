# Training Notes — Tracks 1 & 3 (Pure RL)

Status: Tracks 1 (ALE pure RL) and 3 (Custom pure RL) are **untrained** — only
tiny smoke runs exist (4 and 8 timesteps). Tracks 2 and 4 (tool-assisted) are
the only ones with real results.

## Hardware (this machine)
- CPU: Intel i7-11800H, 8 cores / 16 logical
- GPU: NVIDIA RTX 3050 Ti Laptop (4 GB) — CUDA works (torch 2.6.0+cu124)
- RAM: 7.8 GB total (LOW — main OOM risk)
- Python 3.11.9, stable-baselines3 2.8.0

## Measured throughput (benchmarked, not guessed)
| Track | Policy | Device (actual) | Steady fps |
|-------|--------|-----------------|------------|
| 1 — ALE pure RL | CnnPolicy (84x84x4 frames) | CUDA | ~178 fps |
| 3 — Custom pure RL | MlpPolicy (417-float vec) | CPU (forced in code) | ~4,100 fps |

Both use `DummyVecEnv` → the 4 envs step **sequentially**, so extra CPU cores
don't add env parallelism. Track 1 bottleneck is the Atari emulator stepping,
not the GPU.

## Wall-clock estimates (timesteps ÷ fps + ~5-10s startup)

Track 3 — Custom (fast):
- 100k (default cmd): ~30 seconds
- 1M (real learning starts): ~4 minutes
- 10M (well-trained): ~40 minutes

Track 1 — ALE (slow):
- 100k (default cmd): ~9-10 minutes
- 1M: ~1.6 hours
- 5M: ~8 hours
- 10M (realistic for Atari Tetris): ~15-16 hours

## Key caveat
The default `--timesteps 100000` finishes fast but is **far too short to learn
Tetris**, especially on ALE. Atari from raw pixels needs ~1M-10M+ frames; 100k
gives a near-random policy. README defaults are smoke-sized, not training-sized.

## Recommendations / TODO
- Track 3: cheap — just run `--timesteps 5000000` (~20 min) for a genuinely
  trained model.
- Track 1: budget an overnight run at 5-10M timesteps (8-16 h).
  - Real speedup lever is env throughput, NOT the GPU: switch
    `DummyVecEnv` -> `SubprocVecEnv` and raise `--n-envs` to ~8-12 to
    parallelize Atari stepping across the 16 logical cores (~2-4x fps).
  - WATCH RAM: at 7.8 GB, high `--n-envs` with `SubprocVecEnv` (forks
    separate processes, each loads torch) is the main OOM risk. Scale up
    gradually.

## Commands
```bash
# Track 3 (custom, fast)
python agents/custom/pure_rl_custom_agent.py train --timesteps 5000000 --n-envs 4

# Track 1 (ALE, slow / overnight)
python agents/ale/pure_rl_ale_agent.py train --timesteps 10000000 --n-envs 4 --sticky 0.25
```
