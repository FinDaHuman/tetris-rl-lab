# Expected Performance per Track on Low-End Hardware

Written 2026-07-09. This file sets realistic score/performance expectations
for each of the four tracks given this machine's budget, compares each to
human level, and explains why — grounded in published research and
documented community results. Use it to sanity-check results and to write
the report's analysis sections.

## The hardware budget this is calibrated to

Low-end Windows laptop, CPU-only training (no usable GPU):

- Custom env (Track 3): ~3,900 env steps/s with 8 vectorized envs and an
  MLP policy → **100M steps ≈ 7–8 h**, i.e. one run per night/day slot.
- ALE (Track 1): CNN policy on frames, roughly 70–140 fps here → **3M
  frames ≈ 6–12 h**. That is ~3% of the 200M frames used by the classic
  DeepMind Atari results, and far below the 10⁹–10¹⁰ frame budgets of
  modern agents.
- Tool-assisted tracks (2 and 4) run search + CEM weight optimization on
  CPU, which this machine handles fine (a 500-piece planned episode ≈ 15 s
  on the custom engine).

The budget matters differently per track: pure-RL tracks are
*sample-starved* on this machine, while the tool-assisted tracks are
barely affected because heuristic search needs no training corpus.

## Human reference points

There is no official human baseline for ALE Tetris (it is a homebrew ROM,
absent from the standard Atari-57 human-normalized tables), so human
comparisons below use classic (NES-style) Tetris research and community
data:

- Cognitive-science work using Tetris as a paradigm
  ([Meta-T, Behavior Research Methods 2014](https://link.springer.com/article/10.3758/s13428-014-0547-y))
  records novices scoring in the low tens-of-thousands; classic-Tetris
  experts averaged ~465,000 points in control conditions
  (and collapsed to ~6,500 without the next-piece preview — one-piece
  lookahead matters enormously, which is also our Track 3/4 observation
  setting).
- Community consensus for NES-style scoring: casual players ~50k–200k
  points; solid experienced players 200k–500k
  ([Playbite](https://www.playbite.com/q/what-is-an-average-tetris-score),
  [Quora](https://www.quora.com/What-is-considered-a-good-tetris-score-Whats-the-average)).
  In lines: a casual player clears roughly 20–70 lines per game; strong
  classic players reach 200–300 lines (level-29 killscreen territory).
- Machine ceiling for context: feature-based controllers with placement
  search reach **~660,000 lines** (Dellacherie's handcrafted weights) and
  **~35,000,000 lines** (BCTS, cross-entropy-optimized) on the simplified
  research Tetris — orders of magnitude beyond any human
  ([Thiery & Scherrer 2009](https://inria.hal.science/inria-00418930/document),
  [The Game of Tetris in Machine Learning, arXiv:1905.01652](https://arxiv.org/abs/1905.01652)).

One more framing fact: even offline Tetris (whole piece sequence known) is
NP-complete ([Demaine, Hohenberger & Liben-Nowell 2003](https://arxiv.org/abs/cs/0210020)),
and the ML survey above concludes that *all* existing learned solutions
"fall far short of what can be achieved by expert players playing without
time pressure" on the full game — so nothing in this project should be
expected to "solve" Tetris.

---

## Track 1 — Pure RL on ALE/Tetris-v5 (PPO, CnnPolicy, pixels)

**Expected on this machine: 0 lines, score ≈ 0 (a valid negative result).**

- A 2025 study ran DQN, C51 and PPO specifically on ALE Atari Tetris and
  found all three decisively beaten by a simple heuristic agent in both
  score and compute cost
  ([Bairaktaris & Johannssen, Expert Systems with Applications 277, 2025](https://www.sciencedirect.com/science/article/pii/S0957417425008735)).
  That is with research-scale budgets — ours is a single 3M-frame night.
- The reasons are structural, not budget-only: rewards arrive only on line
  clears, dozens of frames after the decisions that caused them
  (the "within-piece credit assignment problem"), and random exploration
  from pixels almost never completes a line to get the first reward.
  A 2025 community project that did get a purely-learned frame-by-frame
  agent working needed a hierarchical controller and hundreds of thousands
  of episodes — and its flat (single-network) baselines, the closest
  analogue to our Track 1, **collapsed around 1.4M gradient steps**
  ([dev.to write-up](https://dev.to/stat_phantom/i-built-the-first-purely-learned-frame-by-frame-tetris-ai-then-it-started-cheating-322k)).
- **vs human: far below any human.** A first-time player clears a line
  within minutes; Track 1 will likely never clear one. This is the
  expected outcome for frame-by-frame pixel RL at ~3% of the classic DQN
  frame budget, and the literature above shows even full budgets struggle.

## Track 2 — Tool-assisted high score on ALE (decode + search + CEM)

**Expected on this machine: tens of lines; current stable result is
37 lines / score 3700 across seeds 0–9, and that is already near what this
setup can extract.**

- Research consistently shows hand-crafted feature heuristics with
  placement search dominate learned agents on Atari Tetris
  ([Bairaktaris & Johannssen 2025](https://www.sciencedirect.com/science/article/pii/S0957417425008735))
  and reach 10⁵–10⁷ lines on the research game
  ([Thiery & Scherrer 2009](https://inria.hal.science/inria-00418930/document)).
  So the binding constraint here is not intelligence but the *ALE
  interface*: a crude homebrew ROM, imperfect frame decoding, limited
  piece control compared to the simulator-perfect placement models used in
  the literature, and speed-up as the game progresses.
- **vs human: roughly a casual-to-intermediate human on this ROM**
  (37 lines ≈ a casual player's typical game). Expect improvements from here
  to be incremental (planner/decoder fixes), not order-of-magnitude.
- ⚠ **Do not read the identical 37-line result across seeds as consistency.**
  The ALE seed does not change this ROM's piece sequence (verified 2026-07-13),
  so all ten "seeds" are the same game — the zero variance is an artifact of the
  environment, not a property of the agent. What is genuinely demonstrated is
  robustness to *action-execution noise*: the planner is closed-loop and still
  scores exactly 37 lines under sticky actions at 0.25. See
  `docs/REPORTING_NOTES.md`.

## Track 3 — Pure RL on the custom env (PPO, MLP, primitive actions)

**Expected on this machine by the 7/13 deadline: mean lines in the low
single digits per episode. Current promoted result: mean 1.04 lines / 25
episodes (max 3) after 100M steps — the tuned runs may reach ~2–10, and
anything above that would exceed what the setting predicts.**

- The key fact from the literature: essentially *all* strong "RL Tetris"
  results (CEM, BCTS, approximate dynamic programming, the popular GitHub
  DQN-Tetris projects) act on an **afterstate/placement action space** —
  the agent picks *where the piece lands* and the simulator teleports it.
  Our Track 3 deliberately uses **primitive actions** (left/right/rotate/
  drop, gravity every step), which reintroduces the within-piece credit
  assignment problem: ~8–9 decisions per piece before any lock, reward
  only on clears. Purely-learned primitive-action Tetris is
  research-frontier territory as of 2025, not a solved problem
  ([survey, arXiv:1905.01652](https://arxiv.org/abs/1905.01652);
  [frame-by-frame project](https://dev.to/stat_phantom/i-built-the-first-purely-learned-frame-by-frame-tetris-ai-then-it-started-cheating-322k)).
- What our budget buys: 100M primitive steps ≈ ~11M piece placements ≈
  ~370k short episodes. That was enough to learn "build flat-ish and
  complete one row" (the 2026-07-09 Night 2 result — 21/25 episodes clear
  ≥1 line), but refinement is limited by CPU-scale PPO: no room for
  population-scale hyperparameter search, big networks, or 10⁹-step runs.
- Also structural: the lines-reward is quadratic in simultaneous clears,
  but discovering a double/triple by exploration is exponentially rarer
  than a single — expect the policy to remain a "singles" player
  (the frame-by-frame project's clear histogram was likewise ~4.5M singles
  vs 803 tetrises).
- **vs human: well below a novice.** A first-week human clears ~10–30
  lines per game; Track 3 clears ~1 and tops out after ~28 pieces. That
  gap is the honest headline for the report: with placement-level tools
  the same engine yields ~198 lines (Track 4), so the gap isolates exactly
  what "no tools, primitive actions, low-end compute" costs.

## Track 4 — Tool-assisted high score on the custom env (search + CEM)

**Expected on this machine: saturation of whatever piece cap we set —
already achieved: ~198 of the 200-line maximum at the 500-piece cap
(promoted mean score 215,530).**

- This is exactly the regime the literature says is machine-dominated:
  Dellacherie-style features + placement enumeration + a queue lookahead +
  CEM weight optimization is the same recipe that produces 660k–35M line
  runs uncapped
  ([Thiery & Scherrer 2009](https://inria.hal.science/inria-00418930/document);
  [Szita & Lőrincz 2006](https://www.researchgate.net/publication/6743957_Learning_Tetris_Using_the_Noisy_Cross-Entropy_Method)
  — noisy CEM alone improved on prior RL "by almost two orders of
  magnitude"). Our cap, not the method, is the ceiling: 500 pieces × 4
  cells ÷ 10 columns = 200 possible lines, and the agent clears ~198
  (~0.396 lines/piece ≈ 99% of the theoretical 0.4).
- Low-end hardware is a non-issue here: CEM needs only rollouts, not
  gradients, and the optimized planner runs a 500-piece episode in ~15 s.
- **vs human: superhuman on per-piece efficiency and consistency.** No
  human sustains ~0.4 lines/piece with near-zero variance; expert humans
  are instead limited by speed and fatigue, which don't exist for the
  planner. If we ever want a more dramatic number for the report, the
  literature says the same agent uncapped would run into the 10⁵+ line
  range — but a single such episode would take hours of wall-clock, which
  is why we report at fixed piece caps instead.
- **Prediction confirmed (2026-07-13).** The claim above ("our cap, not the
  method, is the ceiling") was tested directly by running the promoted weights
  with no piece cap: **10,000 pieces / 3,997 lines, still alive** when the probe
  was stopped by hand — 0.399 lines/piece sustained the whole way, and no
  top-out. A 2,000-piece cap yields 798 lines (seeds 0/1/2 → 797/798/798). The
  wall-clock caveat also held: ~35 pieces/s, so ~5 min of compute per 10,000
  pieces, which is why fixed caps remain the reporting unit. This is the one
  prediction in this document that has been checked end-to-end against the
  agent rather than against the literature.

---

## Summary table

| Track | Method / constraint | Expected here | Achieved so far | vs human level |
| --- | --- | --- | --- | --- |
| 1 | PPO from pixels, ALE, ~3M frames | 0 lines, score ~0 | (final attempt = Night 4) | Far below novice |
| 2 | Decode + search + CEM on ALE | Tens of lines | 37 lines, seeds 0–9 (but the ALE seed does not change the piece sequence — effectively one game) | ≈ casual human |
| 3 | PPO, primitive actions, custom env, 100M steps | Low single-digit mean lines | 1.04 mean lines (max 3) | Well below novice |
| 4 | Placement search + CEM, custom env | Saturates the piece cap | ~198/200 lines, mean score 215,530; uncapped it does not top out (10,000 pieces / 3,997 lines, confirmed 07/13) | Superhuman efficiency/consistency |

The cross-track story the numbers tell: **the action-space abstraction,
not the learning algorithm, is the dominant variable.** Identical engine,
identical compute class — placement-level tools score ~200× more lines
than primitive-action pure RL, matching what two decades of Tetris
research found.

## Sources

- Bairaktaris & Johannssen (2025), "Outsmarting algorithms: A comparative
  battle between Reinforcement Learning and heuristics in Atari Tetris",
  Expert Systems with Applications 277 —
  <https://www.sciencedirect.com/science/article/pii/S0957417425008735>
- Algorta & Şimşek (2019), "The Game of Tetris in Machine Learning" —
  <https://arxiv.org/abs/1905.01652>
- Thiery & Scherrer (2009), "Improvements on Learning Tetris with Cross
  Entropy" (Dellacherie ~660k lines; BCTS ~35M lines) —
  <https://inria.hal.science/inria-00418930/document>
- Szita & Lőrincz (2006), "Learning Tetris Using the Noisy Cross-Entropy
  Method" —
  <https://www.researchgate.net/publication/6743957_Learning_Tetris_Using_the_Noisy_Cross-Entropy_Method>
- Demaine, Hohenberger & Liben-Nowell (2003), "Tetris is Hard, Even to
  Approximate" — <https://arxiv.org/abs/cs/0210020>
- Lindstedt & Gray (2014), "Meta-T: Tetris as an experimental paradigm for
  cognitive skills research" —
  <https://link.springer.com/article/10.3758/s13428-014-0547-y>
- stat_phantom (2025), "I Built the First Purely Learned Frame-by-Frame
  Tetris AI" (flat agents collapse ~1.4M gradient steps; singles dominate) —
  <https://dev.to/stat_phantom/i-built-the-first-purely-learned-frame-by-frame-tetris-ai-then-it-started-cheating-322k>
- Human score context —
  <https://www.playbite.com/q/what-is-an-average-tetris-score>,
  <https://www.quora.com/What-is-considered-a-good-tetris-score-Whats-the-average>
