# Tetris RL Lab — Final Report

Date: 2026-07-13 (project deadline). Results frozen as of this morning.
All numbers below come from the evaluation manifests and training logs
referenced in each section; the caveats that govern which numbers may be
compared are listed in `docs/REPORTING_NOTES.md`.

## 1. Objective and track separation

The project's goal was to build and honestly measure four Tetris agents
that differ along two axes — *environment* (Atari ALE/Tetris-v5 vs a
custom Gymnasium engine) and *method* (pure reinforcement learning vs
tool-assisted planning):

| Track | Environment | Method | Constraint |
| --- | --- | --- | --- |
| 1 | ALE/Tetris-v5 | Pure RL (PPO, pixels) | No board decoding, no planning, no search, no hand-authored heuristics |
| 2 | ALE/Tetris-v5 | Tool-assisted | Anything goes: frame decoding, board reconstruction, search, CEM |
| 3 | Custom engine | Pure RL (PPO, structured obs) | Only the Gymnasium step API; one-piece preview; primitive actions |
| 4 | Custom engine | Tool-assisted | Engine cloning, placement enumeration, queue lookahead, CEM |

The separation exists to answer one question cleanly: **how much of
"playing Tetris well" comes from learning, and how much from the action
abstraction and search that tools provide?** Pairing each environment
with both methods isolates that variable.

## 2. The two environments

**ALE/Tetris-v5** wraps an Atari 2600 Tetris homebrew ROM. Observations
are raw frames; the game has no next-piece preview, coarse piece control,
and speeds up over time. There is no official human baseline (it is not
part of the Atari-57 suite).

**The custom engine** (`packages/tetris_env`) implements standard Tetris
rules: 20×10 board with hidden spawn rows, 7-bag randomizer, SRS-style
rotation kicks, soft/hard drops, line clears, levels, and score-mode
scoring. The Gymnasium wrapper (`tetris_env/gym_env.py`) exposes the
locked board, the active-piece mask, piece state (row/col/rotation),
current-piece and one next-piece identities. Two reward modes: `score`
(raw engine score deltas) and `lines` (`10 × cleared²` per lock, `+0.25`
per placed piece, `−10` on top-out) — the latter became the default after
the score mode's dense drop points were shown to drown the line signal.

Two engine-level bugs were found and fixed during the project, both via
training telemetry: episodes replayed identical piece sequences per env
seed (fixed 7/08), and episodes could be *infinite* — upward SRS kicks
can cancel gravity, so a deterministic policy could hover one piece
forever, which froze a 100M-step run inside its eval callback for 15 h
(fixed 7/09 with a 50-step-per-piece force-lock, mirroring real Tetris'
move-limit lock delay).

## 3. Pure-RL methods (Tracks 1 and 3)

- **Track 1** (`agents/ale/pure_rl_ale_agent.py`): Stable-Baselines3 PPO,
  `CnnPolicy`, standard Atari preprocessing (84×84, frame stack 4,
  sticky actions 0.25), 10M frames, lr 2.5e-4, 4 envs, CPU.
- **Track 3** (`agents/custom/pure_rl_custom_agent.py`): SB3 PPO,
  `MlpPolicy` (2×256), flat 417-float observation, VecNormalize, 8 envs,
  `lines` reward, 500-piece episode cap, eval callback every 1M steps,
  checkpoints every 5M. The promoted run: 100M steps, lr 3e-4, seed 7.

Both tracks act at the *primitive* level — left/right/rotate/drop with
gravity every step (Track 3) or raw joystick actions (Track 1) — which is
the defining difficulty: ~8–9 decisions per piece before any reward is
possible.

## 4. Tool-assisted methods (Tracks 2 and 4), and why they are separate

- **Track 2** (`agents/ale/ale_tetris_agent.py`): decodes frames into a
  board model, enumerates placements in a cloned model, scores them with
  weighted features, and optimizes the weights CEM-style
  (`legacy_model` planner, depth 2, beam 6).
- **Track 4** (`agents/custom/tetris_custom_agent.py`): placement
  enumeration directly on a cloned engine, Dellacherie-style placement
  features, one-piece **queue lookahead** (depth 2, 4 candidates,
  lookahead weight 0.35), CEM optimization with promotion decided on a
  fixed held-out seed set.

These are kept in separate tracks because they change the *decision
level*: the agent picks **where a piece lands** and search does the rest.
Two decades of Tetris research reaches strong play through this abstraction
rather than through the learning algorithm — but that is the literature's claim,
and Track 5 tested it here directly. It found the abstraction to be **necessary
but not sufficient**: worth 5.4× on its own, yet only 2.3% of the gap to Track 4.
What carries the rest is the hand-authored features and the lookahead layered on
top of it (§7.2, and `docs/EXPECTED_PERFORMANCE.md`).

## 5. Experiment timeline

| Date (2026) | Event |
| --- | --- |
| ≤ 07/03 | Initial agents. Track 2 stable 37-line result. Track 1 10M-frame overnight (final in-training eval: reward 0.00). Track 3 200k-step score-mode starter: 0 lines. |
| 07/07 | Overnight pair on the custom env: Track 4 queue lookahead reaches **78.3 lines at the 200-piece cap** (ceiling 80 — saturated); Track 3 20M-step lines-mode run: 0.48 mean lines (pre-seeding-fix data). |
| 07/08 | Code day: Track 4 planner ~5× faster, duplicate rotations removed; CEM promotion on held-out seeds; `lines` reward made default; Track 1 eval-JSON crash fixed. **Night 1:** Track 4 at 500 pieces — new weights **promoted on score** (215,530 vs 213,780; lines tied ~198/200). Track 3 pilot measures ~3,900 fps → Night 2 sized to 100M. Episode-seeding bug found and fixed. |
| 07/08 → 09 | **Night 2 attempt 1 hung** at 4.0M steps: infinite deterministic eval episode (upward-kick hover). Diagnosed with py-spy + source-level logs; fixed with the 50-step force-lock; 19/19 tests. |
| 07/09 | **Night 2 attempt 2: the Track 3 result.** 100M steps, ~7.8 h: mean **1.04 lines** / 25 episodes (max 3; 21/25 clear ≥1) — the project's first pure-RL line clears. Promoted. Deadline set; one-change-at-a-time tuning retired. |
| 07/10 → 11 | **Night 3 (negative result):** lr 1e-4 + top-out penalty 25, 200M steps (~17 h): **0.44 mean lines** — never took off. The lower lr stabilized PPO (approx_kl 0.15→~0.06) but the harsher penalty taught *hovering* (episode steps ~240→~330–370 at unchanged ~28 pieces). Not promoted. Track 2 re-confirmed (37/3700, seeds 0–2). Track 1 non-sticky attempt dropped (deadline triage). |
| 07/13 | Final Track 3 slot (lr 2e-4) was not launched; results frozen. Track 1 25-episode evaluation manifest generated. This report written. |
| 07/13 (after freeze) | Playback added: every track rendered to mp4 (`artifacts/best_plays/`) plus a live viewer. Doing so required running Track 4 past its cap, which produced the §6 addendum — it never tops out. No agent, model, or frozen number changed. |
| 07/13 (after freeze) | **Track 5 built and run** (12M steps, 2.0 h): the single-variable afterstate experiment §7.2 had been missing. It changes only Track 3's action space. Result 5.60 lines — the action abstraction is worth 5.4×, but closes just 2.3% of the Track 3 → Track 4 gap, **refuting** this report's earlier guess that the abstraction was the dominant variable. |

## 6. Final results by track

| Track | Final result | Episodes | Evidence |
| --- | --- | --- | --- |
| 1 — pure RL, ALE | **0 lines** (mean/max native reward 0.0) | 25, seeds 1000+, sticky 0.25 | `artifacts/ale_pure_rl/evaluation.json` (10M-frame model, manifest 07/13) |
| 2 — tools, ALE | **37 lines, score 3,700**, 259 decisions — identical on every seed | seeds 0–9 (+ re-confirm 0–2) | `artifacts/ale_stable_high_score/evaluation.json`, `runs/plan_20260708/track2_confirm.json` |
| 3 — pure RL, custom | **mean 1.04 lines** (max 3, 21/25 ≥1 line), mean score 361, ~28 pieces survived | 25 deterministic, seeds 1000–1024, 500-piece cap | `artifacts/custom_pure_rl/evaluation.json` |
| 4 — tools, custom | **mean 198.1 lines of a 200 ceiling** (max 199), mean score 215,530 | 10, seeds 0–9, 500-piece cap | `artifacts/custom_best/evaluation_500.json` |
| 5 — pure RL, custom, **placement actions** | **mean 5.60 lines** (max 9, min 3, 25/25 ≥1 line), mean score 676, ~48 pieces survived | 25 deterministic, seeds 1000–1024, 500-piece cap | `artifacts/custom_afterstate/evaluation.json` |

Track 5 was added *after* the freeze (2026-07-13) and is the project's one
controlled experiment: it is Track 3's PPO with **only the action space changed**
(placement instead of keypress), still with no hand-authored features and no
lookahead. It does not replace any frozen number above; it explains the gap
between them. See §7.2.

Every one of these is watchable: `artifacts/best_plays/` holds the best episode
of each track as an mp4, with `manifest.json` recording the seed and stats behind
each one (`python tools/render_best_plays.py` regenerates; the mp4s themselves are
gitignored). `artifacts/best_plays/live_play.py` plays an agent live in a window.
Track 1's video is a top-out with no line clears — that is the result, not a
broken render.

**Post-freeze addendum (2026-07-13, after the results above were frozen):
Track 4's "200-line ceiling" is the piece cap, not the agent.** Rendering the
videos required running Track 4 past its cap, which measured, for the first
time, how long it actually survives: **10,000 pieces / 3,997 lines and still
alive** (probe stopped manually, not a top-out), holding 0.399 lines per piece —
99.8% of the theoretical 0.4 maximum — the whole way. A 2,000-piece episode
clears **798 lines** (score 3,370,000; seeds 0/1/2 → 797/798/798). This confirms
empirically what `docs/EXPECTED_PERFORMANCE.md` predicted from the literature
(Dellacherie-class agents run to 10⁵+ lines uncapped). The frozen headline above
is unchanged and remains the reported result; this only establishes that its
ceiling was an artifact of the 500-piece evaluation cap, so Track 4's line count
should be read as "whatever cap you set × 0.4", not as a plateau.

Human-level context (full grounding in `docs/EXPECTED_PERFORMANCE.md`):
Track 1 is far below any human; Track 3 is well below a first-week novice
(~10–30 lines); Track 2 is roughly a casual human on this ROM with
superhuman consistency; Track 4 is superhuman in per-piece efficiency
(~0.396 lines/piece ≈ 99% of the theoretical 0.4) and consistency.

Comparison caveats: Track 3 evaluations before 07/08 used the score
reward and/or predate the seeding and force-lock fixes; Track 4 numbers
from before the 07/08 planner change differ mildly from post-change
numbers; Night 3's raw rewards are on a penalty-25 scale. None of these
may be mixed in one curve or table without a note
(`docs/REPORTING_NOTES.md`).

## 7. Why pure RL (Tracks 1 and 3) is hard here

The two pure-RL tracks failed/underperformed for the same structural
reasons the literature predicts:

1. **Within-piece credit assignment.** Reward arrives only when a line
   clears, dozens of primitive actions after the decisions that caused
   it. Random exploration almost never completes a line from pixels
   (Track 1 never cleared one in 10M frames), and even with structured
   observations it took ~30M steps to reliably find singles (Track 3).
2. **The action abstraction is worth 5.4×, but it is *not* the dominant
   variable — Track 5 measured it (2026-07-13).** On the *same engine and the
   same machine*, placement-level search (Track 4) scores ~190× the lines of
   primitive-action RL (Track 3): 198.1 vs 1.04. But Track 4 changes **four**
   things at once relative to Track 3: the action space (one placement vs ~9
   primitive moves), hand-authored board features, a 2-ply queue lookahead, and
   the optimizer (CEM vs PPO). Earlier drafts of this report guessed that the
   action space carried most of the 190×, on the grounds that essentially all
   strong "RL Tetris" results use afterstate/placement action spaces. **That
   guess was wrong, and Track 5 is the experiment that showed it.**

   Track 5 (`agents/custom/afterstate_custom_agent.py`) changes **exactly one**
   of the four: PPO picks a *placement* (rotation × column, `Discrete(40)`)
   instead of a keypress. Same network (MlpPolicy 2×256), same reward, same
   hyperparameters, same seed, **no hand-authored features, no lookahead** — the
   observation is the raw board plus the current and next piece, so the agent
   must learn board quality itself. Result: **mean 5.60 lines** (min 3, max 9,
   sd 1.50; 25 deterministic episodes, seeds 1000–1024, 500-piece cap;
   `artifacts/custom_afterstate/evaluation.json`).

   | | Track 3 (primitive) | Track 5 (placement) | Track 4 (features+search) |
   | --- | --- | --- | --- |
   | Mean lines | 1.04 | **5.60** | 198.1 |
   | Mean pieces survived | 28.6 | 47.6 | 500 (never tops out) |
   | Lines per piece | 0.036 | 0.118 | 0.399 |
   | Episodes clearing ≥1 line | 21/25 | **25/25** | 25/25 |
   | Training | 100M primitive steps, ~7–8 h | 12M placement steps, **2.0 h** | CEM, minutes |

   **The comparison is fair on experience, not just on steps.** A Track 3 step is
   one keypress; a Track 5 step is a whole piece. The promoted Track 3 policy
   spends a measured **9.11 primitive steps per piece**, so its 100M steps are
   **≈11.0M pieces** of Tetris experience against Track 5's **12.0M** — matched to
   within 9%, while Track 5 trained in a quarter of the wall-clock.

   Two conclusions, and they point in opposite directions:

   - **The abstraction is real.** At matched experience, changing only the action
     space gave **5.4× the lines**, 1.7× the survival, and 3.2× the per-piece
     efficiency, and it clears at least three lines on *every* seed where Track 3
     is shut out on 4 of 25. Primitive-action PPO also *plateaued* at 36M of its
     100M steps, whereas Track 5 was **still improving when its budget ran out**
     (ep_rew_mean 60 → 75 over the final 4M steps). Credit assignment across ~9
     keypresses per piece is a genuine obstacle, and removing it genuinely helps.
   - **The abstraction is nowhere near sufficient.** 5.60 lines closes only
     **2.3%** of the Track 3 → Track 4 gap ((5.60−1.04)/(198.1−1.04)). The other
     ~97.7% is attributable to what Track 5 was forbidden: hand-authored
     Dellacherie features, queue lookahead, and CEM. Track 5 learned to *clear
     lines*; it did not learn to *survive*, topping out around 48 pieces while
     Track 4 never tops out at all.

   **Honest limit on this result.** Track 5's learning curve had not converged at
   12M steps, so **5.60 is a lower bound on what the abstraction alone can do, not
   a ceiling.** This experiment establishes that the action space is not
   *sufficient* at this compute budget; it does not establish that hand-authored
   features are *necessary* in principle. Deciding that would need Track 5 run to
   convergence (§10.1), which the schedule did not allow.
3. **Optimization pathologies at low compute.** The promoted Track 3 run
   ran "hot" (approx_kl ~0.15–0.18, clip_fraction ~0.41–0.44) and
   plateaued at 36M of 100M steps. The one attempt to fix it (Night 3)
   traded the optimizer pathology for a reward-shaping pathology: a 2.5×
   top-out penalty made *delaying death by hovering* the easiest reward
   improvement, and performance halved. Fixing both at once was not
   possible within the remaining schedule.
4. **Sample starvation.** One CPU-night is 100–200M custom-env steps or a
   few million ALE agent steps, with no capacity for hyperparameter sweeps —
   each configuration got exactly one run.

   *Units, stated precisely, because the two are easy to conflate:* Track 1's
   budget was **10M agent steps**, and with frame-skip 4 that is **~40M
   emulator frames** — about **20%** of the canonical 200M-*frame* Atari
   benchmark budget (which is 50M agent steps). So Track 1 was under-trained
   by roughly 5×, not by the ~50× that a naive "10M vs 200M" reading implies.
   That said, more steps would not have rescued it: as §7.1 argues, the reward
   was identically zero for the entire run, so the policy gradient was zero —
   the problem is the *absence* of a gradient, not a shortage of samples to
   estimate it from.

Track 3's 1.04 lines is nonetheless a genuine positive result for
primitive-action RL at this budget: 21/25 deterministic episodes clear at
least one line, which matched the literature-derived expectation of
"low single digits" for this setting.

## 8. Why Track 4 improved strongly with queue lookahead

Track 4's method — placement enumeration + Dellacherie-style features +
CEM — is the recipe that dominates machine Tetris (660k–35M lines
uncapped in published work). Its project-specific gains came from:

- **Queue lookahead** (scoring each placement against real upcoming
  queue pieces, depth 2): lifted the 100-piece-cap mean score from 7,900
  to 9,500 in the first test, then saturated the 200-piece cap
  (78.3 of 80) and the 500-piece cap (198.1 of 200).
- **Honest promotion**: CEM promotion decided on a fixed held-out seed
  set (not the training seeds), which stopped lucky-seed artifacts.
- **A 5× faster planner** (vectorized placement enumeration + duplicate
  symmetric-rotation removal), which made 500-piece optimization runs
  practical on this hardware (~15 s/episode).

At the 500-piece cap the ceiling itself (200 lines = 500 pieces × 4
cells ÷ 10 columns) is the binding constraint: the final promotion was
decided on mean score (215,530 vs 213,780) because lines had saturated
(~198 both). Further Track 4 gains would require longer caps, at
proportionally longer wall-clock per episode.

Measured after the freeze (see §6): raising the cap does exactly that and
nothing else. The agent does not top out — 10,000 pieces produced 3,997
lines with the game still alive — so its line count is a linear function
of the cap (≈0.4 × pieces) rather than a property of the agent that
further tuning could improve. Track 4 is finished in the only sense that
matters: it plays at the theoretical maximum efficiency, and the remaining
"headroom" is just wall-clock (~35 pieces/s, so ~5 minutes of compute per
10,000 pieces).

## 9. Hardware and runtime limitations

All work ran on a single low-end Windows 11 laptop, CPU only:

- Custom env PPO: ~3,300–3,900 steps/s (8 envs) → 100M steps ≈ 7–8 h,
  200M ≈ 17 h. One training slot per day/night.
- ALE PPO: order 100 agent steps/s → a few million agent steps is an
  overnight run; **10M agent steps (~40M emulator frames at frame-skip 4)**
  was the largest Track 1 budget spent — roughly 20% of the canonical
  200M-frame Atari budget (= 50M agent steps). State the unit: earlier
  drafts of this report compared agent steps against frames directly and
  understated the budget.
- Consequences: no hyperparameter search (every config = one run), no
  seeds-replication of training runs, small networks, and a hard trade
  between run length and number of experiments. Two full nights were
  additionally lost to environment bugs (the seeding bug's tainted pilot
  and the 15 h hang).
- Mitigations that mattered: source-level `log.txt`/`progress.csv`
  logging (diagnosed the hang and the hovering), checkpoints every 5M
  steps, eval callbacks every 1M, and py-spy for live stack dumps.

## 10. Next work

1. ~~**An afterstate variant on the custom env — the missing experiment.**~~
   **Done 2026-07-13 — this is now Track 5 (§7.2).** It answered the question:
   the action abstraction is worth 5.4× (1.04 → 5.60 lines at matched
   experience) but accounts for only 2.3% of the Track 3 → Track 4 gap, so the
   hand-authored features and lookahead carry the rest. **The follow-up is to
   run it to convergence:** Track 5's learning curve was still climbing when its
   12M-step budget ended (ep_rew_mean 60 → 75 over the final 4M steps), so 5.60
   is a lower bound, not a ceiling, and the *sufficiency* of the abstraction is
   still open. A 50–100M-step run (~8–16 h at the measured 1,664 steps/s) would
   settle whether placement-level PPO plateaus in the single digits or keeps
   going. This is now the highest-value item in the project.

2. **A features-only ablation to complete the 2×2.** Track 5 isolated the action
   space; nothing isolated the *features*. Feeding the 10 Dellacherie features
   (instead of the raw board) to Track 5's PPO, still with no search, would
   attribute the remaining 97.7% between "better state representation" and
   "lookahead + CEM". Together with Track 5 that would fully decompose the 190×.

Then, to improve Track 3's number specifically:

3. **The never-run lr 2e-4 experiment** (Night 4 in `TRAINING_PLAN.md`):
   Night 2's proven config with only the learning rate halved-ish —
   directly targets the diagnosed 36M plateau without Night 3's penalty
   mistake. 150M steps, ~12–13 h.
4. **Survival shaping without bigger penalties**: raise `--piece-reward`
   (rewarding each lock fights hovering by construction) instead of
   raising the top-out penalty, which Night 3 showed induces hovering.
5. **lr / entropy schedules** in the Track 3 trainer (constant-only
   today): linear lr decay is the standard cure for
   plateau-then-oscillate.
6. **Track 1 non-sticky attempt** (the dropped slot): expected 0 lines,
   but it would complete the documented negative-result pair. Note that the
   ALE seed does not vary the piece sequence (§6 / `docs/REPORTING_NOTES.md`),
   so sticky actions are the *only* stochasticity available in that
   environment — which makes this ablation more interesting than it looked.
7. ~~**Track 4 at longer caps** (1,000+ pieces) if a bigger headline
   number is ever needed~~ — **done 2026-07-13** (§6 addendum): it does not
   top out (10,000 pieces / 3,997 lines, still alive), so a longer cap buys
   a bigger number and no new information. Not worth further time.

## Appendix A — Watching the agents play

Added 2026-07-13 after the freeze. No agent, model, or reported number was
changed by this work; it only makes the existing results observable.

| Track | Video (`artifacts/best_plays/`) | Seed | Result shown | Length |
| --- | --- | --- | --- | --- |
| 1 | `track1_ale_pure_rl.mp4` | 14 | 0 lines — stacks and tops out | 0:30 |
| 2 | `track2_ale_tool.mp4` | 0 | 37 lines / 3,700 | 2:34 |
| 3 | `track3_custom_pure_rl.mp4` | 4 | 2 lines, 33 pieces | 0:24 |
| 4 | `track4_custom_tool.mp4` | 1 | 798 lines at a 2,000-piece cap | 11:10 |

- `python tools/render_best_plays.py` regenerates all four: it plays a batch of
  seeded episodes per track, ranks them by lines then score, and re-runs the
  winner with frame capture. The mp4s are gitignored; `README.md` and
  `manifest.json` in that directory are committed.
- `python artifacts/best_plays/live_play.py` plays Track 4 (or `--track 3`) live
  in a window until closed. Track 4 has no piece cap there — it runs indefinitely.
- **Fidelity note.** Tracks 1–3 act at the primitive level, so their videos are
  simply one frame per environment step. Track 4 does not: its planner commits a
  chosen `Placement` by overwriting the board, so pieces would teleport. To
  animate it honestly, each placement is converted back into primitive engine
  actions (rotate → shift → hard drop) and *verified on a clone* before being
  applied — the sequence is used only if the resulting board is bit-identical to
  the one the planner chose, otherwise the placement is committed directly and a
  synthetic fly-in is drawn (`packages/tetris_env/replay.py`). Across the full
  2,000-piece video this fell back **zero** times, so every piece shown is real
  engine play landing on the planner's exact board. Verification is not optional
  here: `TetrisGame.step` applies gravity after every action and SRS kicks shift
  the piece sideways mid-rotation, so a naive action sequence silently misses.
