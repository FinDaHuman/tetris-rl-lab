# Q&A Preparation

Prep for the 1-on-1 defense. The assignment was: *pick a game from the ALE list,
build an RL agent, score as high as you can.* This repo answers that with four
tracks instead of one agent, and the pure-RL ALE agent scores zero. That is
unusual, so expect to defend it.

Rule for the whole session: **every number in here is real and every weakness in
§6 is real.** Do not oversell. A professor who catches you spinning one weak
point will stop believing the strong ones. The strongest position you have is
that you know exactly where your own work is weak — most students don't.

> **He has said he will also ask about game mechanics, algorithm input, agent
> structure — and then point at a line of code and ask what it does.**
> That is a different skill from defending the project, and it has its own file:
> **[docs/QA_CODE_WALKTHROUGH.md](QA_CODE_WALKTHROUGH.md)**. It teaches the engine,
> the observation spaces, PPO and CEM from scratch, and drills the ~20 lines most
> likely to get pointed at, each with an answer. **Read that one first if you are
> short on time** — this file wins you the argument, but that one gets you through
> the next five minutes.

---

## 1. The opening answer (have this ready verbatim)

If he opens with any version of *"this isn't what I asked for"* or *"where's the
score?"*:

> On ALE Tetris, the two halves of the brief pull against each other. A pure-RL
> agent scores zero — I trained PPO for 10M steps and it never cleared a single
> line, which is what the literature predicts for Tetris. But anything that
> actually scores well on Tetris isn't pure RL; it's placement search over
> hand-designed features. So "an RL agent that scores highest" is, for this game,
> close to a contradiction.
>
> I didn't know which half of that you cared about, and I didn't want to guess.
> So I built both, on two environments: pure RL and tool-assisted, on ALE and on
> a custom engine I wrote. That gives you a real score on the assigned
> environment (37 lines), a real RL result (my custom-env agent does clear
> lines), and — because the four tracks form a 2×2 — an actual answer to *why*
> pure RL fails here, rather than just an excuse.
>
> And then I ran a fifth track to check my own explanation. I'd claimed the gap
> was mostly about the action abstraction. Track 5 changes only that — one
> variable — and it showed I was wrong: the abstraction is worth about 5×, but
> it accounts for only 2% of the gap. The hand-designed features do the rest.

Then stop talking and let him pick which thread to pull.

**Track 5 is your best card.** It's the difference between a student who built
four things and a student who ran an experiment, got a result that contradicted
his own report, and wrote that down. Play it early.

**Be honest about the ordering.** If he asks "did you plan this 2×2 up front as
an experiment?" — the answer is **no**. It started as a hedge against an
ambiguous brief and against a zero score, and it *became* a controlled comparison.
Say that. It is a completely respectable thing to have done, and pretending it was
a grand design from day one will fall apart under one follow-up.

---

## 2. Why four tracks

Three honest reasons, in the order they actually happened:

1. **Pure RL on ALE Tetris scores zero, and I needed to know if that was my bug
   or the problem's nature.** The only way to tell is to change one thing at a
   time and see what recovers. So: keep the method, change the environment
   (Track 3) — it starts clearing lines. Keep the environment, change the method
   (Track 2) — it scores. That isolates the cause.
2. **I didn't know what was allowed.** "Make an RL agent" could mean strictly
   model-free learning from environment interaction, or it could permit search,
   planning, and hand-authored features (a lot of published "RL Tetris" does).
   Rather than pick an interpretation and risk submitting against the wrong one,
   I built both and drew an explicit, enforced boundary between them — that's what
   `AGENTS.md` is: a written rule that the pure-RL tracks may never import board
   decoding, placement enumeration, engine cloning, or search, and CI-style tests
   plus code review keep them clean.
3. **It makes the negative result publishable instead of embarrassing.** "PPO got
   0" is a shrug. "PPO got 0 on pixels, 1.04 lines on a structured observation
   with a shaped reward, and a placement-level planner on the *same engine and the
   same laptop* got ~190× more" is a finding.

The 2×2:

|  | **Pure RL** | **Tool-assisted** |
| --- | --- | --- |
| **ALE/Tetris-v5** | Track 1 — 0 lines | Track 2 — 37 lines |
| **Custom engine** | Track 3 — 1.04 mean lines | Track 4 — 198 lines @ 500 pieces (uncapped: never dies) |

Reading it: **across a row** = what the method buys you. **Down a column** = what
the environment buys you.

---

## 3. The numbers, and what they actually mean

| Track | Result | Evidence |
| --- | --- | --- |
| 1 — pure RL, ALE | **0 lines**, mean/max reward 0.0, 25 episodes | `artifacts/ale_pure_rl/evaluation.json` |
| 2 — tools, ALE | **37 lines**, 259 decisions | `artifacts/ale_stable_high_score/evaluation.json` |
| 3 — pure RL, custom | **mean 1.04 lines** (max 3; 21/25 episodes ≥1 line), ~28 pieces survived | `artifacts/custom_pure_rl/evaluation.json` |
| 4 — tools, custom | **mean 198.1 lines** of a 200 ceiling @ 500-piece cap; uncapped: 10,000 pieces / 3,997 lines and still alive | `artifacts/custom_best/evaluation_500.json` |
| 5 — pure RL, custom, **placement actions** | **mean 5.60 lines** (max 9, min 3; **25/25** episodes ≥1 line), ~48 pieces survived | `artifacts/custom_afterstate/evaluation.json` |

Track 5 is not a fifth "attempt at a high score" — it is the **controlled
experiment** that explains the others (§6.3). It is Track 3 with one variable
changed.

**Two things to say before he reads a number wrong:**

- **"Score 3,700" is my own unit, not the ROM's.** ALE/Tetris-v5's native reward
  *is lines cleared* — my agent gets 37.0 reward = 37 lines. `estimate_atari_score()`
  in `agents/ale/env.py` just multiplies lines × 100 for a friendlier headline. If
  he asks what you scored, **say "37 lines."** Don't quote 3,700 as if the game
  printed it.
- **Track 4's "200-line ceiling" is my piece cap, not the agent's limit.** 500
  pieces × 4 cells ÷ 10 columns = 200 possible lines. Uncapped, it doesn't top out
  at all — I ran it to 10,000 pieces / 3,997 lines and stopped it by hand. Its
  score is `≈ 0.4 × cap`. Never quote a Track 4 line count without its cap.

Scores are **not comparable across tracks** (Track 3's includes drop points,
Track 4's doesn't, Tracks 1–2 use my lines×100 convention). **Compare lines.**

---

## 4. How each track works (be able to draw this on a whiteboard)

### Track 1 — Pure RL on ALE (`agents/ale/pure_rl_ale_agent.py`)
Stable-Baselines3 **PPO, CnnPolicy**. Standard Atari preprocessing: RGB → grayscale,
84×84, frame-skip 4, no-op starts (≤30), 4-frame stack, sticky actions 0.25. 4
parallel envs, CPU. lr 2.5e-4, n_steps 128, batch 256, clip 0.1, ent_coef 0.01,
gamma 0.99. **10M agent steps.** Sees pixels, outputs joystick actions. Nothing else.
**Result: never cleared one line.**

### Track 2 — Tool-assisted on ALE (`agents/ale/ale_tetris_agent.py`)
Not RL. Per piece: **decode the frame** into a 20×10 binary board (background gray
≈ (111,111,111) ± 16 → threshold), identify the falling piece by shape-matching its
cells against a tetromino table, **enumerate every (rotation, column) placement**,
score each with a weighted linear function of board features, pick the best, and
emit the joystick actions to get there. The weights were tuned by **CEM**. It is a
closed-loop planner: it re-reads the board every piece, so it self-corrects.

### Track 3 — Pure RL on my engine (`agents/custom/pure_rl_custom_agent.py`)
SB3 **PPO, MlpPolicy (2×256)**. My own Gymnasium env (`packages/tetris_env`): real
Tetris rules — 20×10 + 2 hidden rows, 7-bag randomizer, SRS kicks, hard/soft drop.
Observation = **417 floats**: locked board (200) + active-piece mask (200) + piece
row/col/rotation (3) + current-piece one-hot (7) + next-piece one-hot (7).
**Primitive actions** (left/right/rotate/soft/hard drop) with gravity every step.
VecNormalize, 8 envs, 100M steps (~7.8 h). **Reward (`lines` mode):**
`10 × cleared² + 0.25 per piece locked − 10 on top-out`.
**Result: mean 1.04 lines — the project's only pure-RL line clears.**

### Track 4 — Tool-assisted on my engine (`agents/custom/tetris_custom_agent.py`)
Also not RL. **Clone the engine**, enumerate all placements for the current piece,
score each with **10 Dellacherie-style features**, and add a **depth-2 lookahead
over the real upcoming queue** (top 4 candidates, future discounted ×0.35). Weights
tuned by **CEM**, with best-weight promotion decided on a **held-out seed set** so
it isn't a lucky-seed artifact. **Result: ~0.4 lines/piece — the theoretical maximum
— indefinitely.**

**The 10 features** (`packages/tetris_env/features.py`): `score_delta, lines,
aggregate_height, holes, bumpiness, wells, row_transitions, col_transitions,
max_height, landing_height`.

**Explaining CEM in one breath:** sample a population of weight vectors from a
Gaussian, play rollouts with each, keep the top-k elite, refit the Gaussian's mean
and variance to the elite, repeat. Derivative-free — you never differentiate through
the game. It's the right tool here because there are only 10 parameters and the
fitness is a noisy rollout.

### Track 5 — Afterstate RL on my engine (`agents/custom/afterstate_custom_agent.py`)
**This one IS reinforcement learning** — it's Track 3's PPO, and the *only* thing
that changed is the action space.

- **Action:** `Discrete(40)` = 4 rotations × 10 columns. **One action = one whole
  piece placed.** (Track 3 emits ~9 keypresses per piece.) The env maps the action
  to the nearest legal placement and commits it.
- **Observation:** raw 20×10 board (200) + current piece one-hot (7) + next piece
  one-hot (7) = **214 floats**. **No holes, no heights, no bumpiness — no features
  at all.** It must learn what a bad board looks like.
- **No lookahead, no search, no engine cloning to plan.** Same one-piece preview
  Track 3 gets.
- **Everything else identical to Track 3:** MlpPolicy 2×256, reward
  `10×cleared² + 0.25/piece − 10 on top-out`, lr 3e-4, γ 0.995, n_steps 512, batch
  256, 8 envs, seed 7.
- **12M steps, 2.0 h. Result: mean 5.60 lines.**

The discipline is the whole point, and it's written down as an enforced rule in
`AGENTS.md`: *if you add a feature to the observation "because it would learn
faster," you have turned Track 5 into a worse Track 4 and destroyed the only
experiment that isolates the action abstraction.* If he asks "how do I know you
didn't sneak features in?" — point at `tests/test_afterstate_env.py::
test_observation_is_raw_board_only_no_hand_features`, which asserts the observation
is exactly board + two one-hots and nothing derived.

---

## 5. Why they perform so differently — the core of your defense

### 5.1 Why Track 1 gets exactly zero (this is the important one)
Not "PPO is bad." **PPO received no learning signal at all.**

ALE Tetris's reward *is* lines cleared. To clear one line you must place ~10 pieces
correctly in a row. A randomly-initialized policy will essentially never do that by
chance — so the reward is **0 for every transition in the entire 10M steps**. With
an all-zero return, the advantage estimate is zero, the policy gradient is zero, and
**PPO has nothing to climb.** It isn't learning slowly; it is not learning at all.
Tetris is the textbook hard-exploration / sparse-reward Atari game — it's not in the
Atari-57 benchmark suite, and standard DQN/PPO/Rainbow results on it are ~0.

*Do not say "it needed more frames."* More zeros is still zeros. What it needed was
a reward gradient before the first line clear.

### 5.2 Why Track 3 escaped zero
Three changes, and you should be able to say which mattered:
1. **Shaped reward** — `+0.25` per piece *locked* gives a non-zero gradient long
   before the first line clear. **This is the one that mattered most.**
2. **Structured observation** — the board is handed over as occupancy, so the net
   doesn't have to solve pixels→board vision *and* Tetris at once.
3. **More samples** — the custom env runs ~3,300–3,900 steps/s vs ALE's ~100 fps,
   so 100M steps was an overnight run.

### 5.3 Why Track 4 crushes Track 3 (~190×)
The honest one-liner: **it is solving a much easier problem.** Four things change at
once between them:

| | Track 3 (pure RL) | Track 4 (tools) |
| --- | --- | --- |
| **Action space** | ~8–9 primitive moves per piece, gravity ticking | **1 decision per piece**: which of ~34 placements |
| **Credit assignment** | reward is dozens of actions after the cause | the decision *is* the outcome |
| **Board evaluation** | must learn what a bad board looks like | hand-given (holes, bumpiness, …) |
| **Model** | model-free | clones the engine and searches 2 pieces ahead |
| **Params to fit** | ~150k network weights, via PPO | **10 weights**, via CEM |

The literature suggests the **action abstraction** is the dominant term — essentially
every strong "RL Tetris" result uses afterstate/placement actions
(Thiery & Scherrer 2009: Dellacherie-style feature agents reach ~660k lines;
BCTS ~35M). **I tested that on my own engine, and it turned out to be wrong.**

**Track 5 isolates the action space** (change it, and *only* it, from Track 3):
1.04 → **5.60 lines**. So the abstraction is worth **5.4×** — real, but it closes only
**2.3%** of the Track 3 → Track 4 gap. The remaining ~97.7% is the hand-authored
features + lookahead + CEM. **The abstraction is necessary but nowhere near
sufficient.** Full answer, including the caveat that Track 5 hadn't converged, in
**§6.3** — read that section before you walk in.

---

## 6. The hard questions — where you are actually vulnerable

Read this section twice. These are the ones that can hurt, and each has an honest
answer that is *better* than a dodge.

### 6.1 "Tracks 2 and 4 aren't reinforcement learning."
**He's right. Concede immediately.** They are derivative-free policy search (CEM)
over a hand-designed, feature-based heuristic with search. There is no value
function, no TD error, no policy gradient, no bootstrapping. The honest framing:
CEM-over-features is a *policy search* method with a real place in the Tetris RL
literature (Szita & Lőrincz 2006 used noisy CEM and beat prior RL by ~two orders of
magnitude), but calling it "an RL agent" would be wrong. **Track 1 and Track 3 are
my RL agents. Tracks 2 and 4 are the tool-assisted comparison group** — that's the
entire point of separating them.

### 6.2 ⚠ "You evaluated on 10 seeds and got 37 lines every single time. Isn't that suspicious?"
**This is the sharpest question in the set, and the answer is a real limitation you
should volunteer before he finds it.**

I checked this directly: **the ALE seed does not change ALE/Tetris-v5's piece
sequence.** Playing NOOP-only from seeds 0, 1, 2 and 42 produces a bit-identical
trajectory. The ROM's piece generator isn't seeded by ALE. So my "10 seeds" is
**not 10 independent samples — it's the same game 10 times.** The effective sample
size for game-to-game variation is **1**, and the zero variance is an artifact of
that, not evidence of robustness.

What *is* genuine: the planner is robust to **action-execution noise** — I re-ran it
with sticky actions at 0.25 and it still produces exactly 37 lines, because it's
closed-loop (it re-reads the board every piece and corrects). That's a real result.
But it has been tested on exactly one piece sequence.

Also worth conceding: **Track 1 was trained/evaluated with sticky 0.25, Track 2 with
sticky 0.0 by default.** Not identical conditions. (It doesn't change the
conclusion — Track 2 scores 37 either way — but he shouldn't have to be the one to
notice.)

The custom engine (Tracks 3/4) *does* have real seed diversity — 7-bag, seeded per
episode — so its variance numbers mean something. The ALE ones don't.

### 6.3 ★ "You claim the action abstraction is what matters. Did you actually show that?"
**This is now your strongest answer in the entire session. Lead with it if he gives
you any opening at all.**

> "I claimed it, then I realised I hadn't shown it, so I ran the experiment — and it
> proved me wrong. That's Track 5."

The problem: Track 4 differs from Track 3 in **four** ways at once (action space,
hand-authored features, search/lookahead, optimizer). The 190× is the effect of the
whole bundle. My report originally guessed the action space carried most of it,
because that's what the literature suggests. **That guess was wrong.**

**Track 5** (`agents/custom/afterstate_custom_agent.py`) changes **exactly one** of
the four. Same PPO, same MlpPolicy 2×256, same reward, same hyperparameters, same
seed. **No hand features, no lookahead** — the observation is the raw 20×10 board
plus current and next piece, so it must learn board quality itself. The only
difference from Track 3: it outputs a **placement** (rotation × column, `Discrete(40)`)
instead of a keypress.

| | Track 3 (keypress) | **Track 5 (placement)** | Track 4 (features+search) |
| --- | --- | --- | --- |
| Mean lines | 1.04 | **5.60** | 198.1 |
| Pieces survived | 28.6 | 47.6 | 500 (never tops out) |
| Lines per piece | 0.036 | 0.118 | 0.399 |
| Episodes clearing ≥1 line | 21/25 | **25/25** | 25/25 |
| Training | 100M steps, ~7–8 h | 12M steps, **2.0 h** | CEM, minutes |

**Two findings, and you must give both:**

1. **The abstraction is real: 5.4× the lines.** And the comparison is fair on
   *experience*, not just step count — which he will absolutely probe. A Track 3
   step is one keypress; a Track 5 step is a whole piece. I measured Track 3 at
   **9.11 primitive steps per piece**, so its 100M steps ≈ **11.0M pieces** against
   Track 5's **12.0M** — matched within 9%, in a quarter of the wall-clock.
2. **But it is nowhere near sufficient: it closes only 2.3% of the gap.**
   (5.60−1.04)/(198.1−1.04) = 2.3%. The other **97.7%** is the hand-authored
   features + lookahead + CEM. Track 5 learned to *clear lines*; it never learned to
   *survive* — it tops out around 48 pieces while Track 4 never tops out at all.

**Conclusion to say out loud:** *"The action abstraction is necessary but nowhere
near sufficient. I had it backwards, and the experiment is what told me."*

**The caveat you must volunteer** (he will find it otherwise): Track 5's learning
curve **had not converged** — ep_rew_mean was still climbing, 60 → 75, over the final
4M steps. So **5.60 is a lower bound on the abstraction, not a ceiling.** What I've
shown is that the action space isn't sufficient *at this compute budget*; I have
**not** shown that hand-authored features are necessary in principle. Settling that
needs Track 5 run to convergence, and that's my #1 next step.

Note the asymmetry, because it's a genuine point in the abstraction's favour: Track 3
**plateaued** at 36M of its 100M steps. Track 5 was **still improving** when the
budget ran out.

**"So what would you do next?"** Two things, in order: (a) run Track 5 to convergence
(50–100M steps, ~8–16 h) to find its actual ceiling; (b) a **features-only** ablation
— feed the 10 Dellacherie features to Track 5's PPO with still no search — which
would split the remaining 97.7% between "better state representation" and "lookahead
+ CEM" and fully decompose the 190×.

### 6.4 "10M steps is tiny. The Atari benchmark is 200M frames."
The units are a trap — **know this cold, because he will**:

> My budget was **10M agent steps**. With frame-skip 4 that's **~40M emulator
> frames**. The canonical 200M-*frame* benchmark is 50M agent steps. So I trained to
> about **20%** of the standard budget — under-trained by ~5×, not the ~50× that a
> naive "10M vs 200M" reading suggests.

Then immediately take the argument away from him: **more frames would not have fixed
it** (§5.1). The reward was identically zero for the entire run, so the gradient was
zero. You cannot fix a *missing* signal with more samples of it.

(This unit error was in my own earlier drafts — they compared agent steps to frames
and understated the budget as "1.5–3%". It's corrected in `docs/REPORT.md` §9 and
`docs/REPORTING_NOTES.md` now. Use the numbers above.)

### 6.5 "Why PPO? Did you try DQN / Rainbow?"
No — one algorithm was all the compute allowed (every configuration got exactly one
run; no hyperparameter search, no seed replication). PPO because it's the SB3 Atari
default, on-policy and stable, and its diagnostics (`approx_kl`, `clip_fraction`) are
readable, which is how I diagnosed Track 3. **Honest caveat: for a sparse-reward
exploration problem, a value-based method with a replay buffer (DQN/Rainbow) is
arguably the better prior**, and I can't claim I ruled it out. What would *actually*
have moved the needle is not the algorithm but intrinsic motivation / count-based
exploration, or the afterstate abstraction from §6.3.

### 6.6 "Your bumpiness weight is positive. Explain."
Learned weights: `lines +505`, `holes −36.5`, `col_transitions −14.8`,
`row_transitions −6.6`, `landing_height −4.4`, `wells −4.1`, `max_height −3.6`,
`aggregate_height −2.0`, `score_delta −2.0`, **`bumpiness +4.9`**.

Everything is signed as Tetris intuition says *except bumpiness*, which is positive —
CEM decided a *bumpier* board is mildly better. **I ablated it** (5 seeds × 500 pieces
each, `runs/qa_prep/bumpiness_ablation.json`):

| bumpiness weight | mean lines | mean score |
| --- | --- | --- |
| **+4.876** (as promoted) | 197.6 | 214,660 |
| **0** (feature switched off) | 197.4 | 215,100 |
| **−4.876** (the "intuitive" sign) | 199.0 | 214,920 |

**Answer: the feature does essentially nothing.** Switching it off entirely moves the
result by 0.2 lines — noise. So the positive weight is not a strategy; it's a
*redundant* feature sitting in a flat region of the fitness landscape. `row_transitions`
and `col_transitions` already penalize a jagged surface, and they carry much larger
weights (−6.6 and −14.8), so CEM had no gradient to push bumpiness in either direction
and parked it arbitrarily. Flipping it to the intuitive negative sign is very slightly
better (199.0, and dead-consistent), which is exactly what you'd expect from a variable
that barely matters.

**State the caveat before he does:** all three variants sit near the 200-line cap, so
this ablation has limited power to separate them — it can show the feature is *not
important*, but not finely rank the three. A sharper test would need a much longer cap.

### 6.7 "Track 3 clears one line. That's terrible."
Agreed — it's far below a novice human (~10–30 lines). But it is a **positive result
for the setting**: 21 of 25 deterministic episodes clear at least one line with
primitive actions, no search, and no domain knowledge, which matches the
literature-derived expectation of "low single digits" (`docs/EXPECTED_PERFORMANCE.md`).
And I can tell you exactly what limits it: **survival, not line-finding** — it tops
out after ~28 pieces of a 500-piece cap.

---

## 7. Questions he'll ask about the code and the engine

**"You wrote your own Tetris engine? How do I know it's correct?"**
7-bag randomizer, SRS-style rotation kicks (proper kick tables for I and JLSTZ),
hidden spawn rows, hard/soft drop, standard line scoring. It's tested
(`python -m pytest`, 28 tests), including an **oracle test** that proves the fast
vectorized placement enumeration returns exactly the same placement set as the naive
implementation it replaced.

**"Did you find any bugs?"** — Yes, two, and both are good stories. Lead with them;
they show you read your own telemetry.
1. **The env replayed the same piece sequence every episode** for a given env seed
   (fixed 7/08). Training was effectively on `n_envs` fixed games.
2. **Episodes could be infinite.** Upward SRS rotation kicks can cancel gravity, so a
   *deterministic* policy could hover one piece forever. This **hung a 100M-step run
   for 15 hours** inside its eval callback. Diagnosed with py-spy. Fixed with a
   50-step-per-piece force-lock — which is exactly what real Tetris does (move-limit
   lock delay).

**"Show me a negative result."** — Night 3. I changed lr to 1e-4 *and* raised the
top-out penalty from 10 to 25, ran 200M steps (~17 h), and got **0.44 lines — worse
than the 1.04 baseline.** The diagnosis is the interesting part: the lower lr did fix
PPO's optimizer pathology (approx_kl 0.15 → 0.06), but the bigger death penalty taught
the agent to **hover** — episode length grew from ~240 to ~330–370 steps at the *same*
~28 pieces placed. It learned to delay dying rather than to stack better. Classic
reward-shaping own-goal, and it's bounded only by the force-lock from bug #2.
**Concede the methodology error too:** I changed two variables in one run, so the
attribution is confounded. I did that knowingly, because the deadline left one slot.

**"What do approx_kl / clip_fraction tell you?"** — They're PPO's trust-region
diagnostics. Healthy is `approx_kl` ~0.01–0.03; my promoted Track 3 run sat at
**0.15–0.18** with `clip_fraction` 0.41–0.44 — i.e. the policy was moving far too far
per update and most samples were being clipped. The run plateaued at ~36M of 100M
steps. That's *why* the next experiment was a lower learning rate.

**"Why the squared term in `10 × cleared²`?"** — Tetris scoring is superlinear (a
4-line Tetris is worth far more than 4 singles), so squaring encourages the agent to
build for multi-line clears instead of chipping single rows.

**"Why `+0.25` per piece? Doesn't that reward stalling?"** — No, and this is a subtle
point worth making: it rewards **locking** a piece, not surviving a step. Hovering
earns nothing. That's precisely why raising the *top-out penalty* backfired (Night 3)
while the piece reward didn't — penalty-avoidance rewards *delay*, whereas the piece
reward requires actually placing something.

**"Why did the original `score` reward fail?"** — The engine pays drop points (2/cell
hard drop). That's ~30+ dense points per piece just for dropping fast, which
completely drowns the sparse ~100-point line clear. PPO correctly learned to drop
pieces as fast as possible and never clear a line. It was maximizing exactly what I
asked for. Good cautionary tale about reward design.

---

## 8. Things to say "I don't know" to

Say it plainly. It costs you nothing and buys credibility for everything else.

- **Where Track 5 actually converges.** It was still improving at 12M steps, so
  5.60 lines is a lower bound and I don't know the ceiling of the action
  abstraction on its own (§6.3). This is the most important "I don't know" in the
  list — it's the limit of my strongest result, and it's my #1 next step.
- Whether hand-authored features are *necessary* in principle, or just necessary at
  my compute budget. Track 5 can't distinguish those.
- Whether DQN/Rainbow would beat PPO on Track 1 — never tried.
- Whether Track 3 would improve at 500M+ steps — never had the compute.
- What the ROM's *actual* on-screen score is vs. my lines×100 convention.
- Whether Track 4's weights transfer to a different board size — never tested.

(Note: the positive bumpiness weight **is** ablated — see §6.6. Don't say "I don't
know" to that one.)

---

## 9. Cheat sheet

| | Track 1 | Track 2 | Track 3 | Track 4 | **Track 5** |
| --- | --- | --- | --- | --- | --- |
| Env | ALE | ALE | custom | custom | custom |
| Method | PPO CnnPolicy | decode + search + CEM | PPO MlpPolicy | enumerate + features + lookahead + CEM | **PPO MlpPolicy** |
| Learns? | yes (RL) | no (policy search) | yes (RL) | no (policy search) | **yes (RL)** |
| Actions | joystick | placement → joystick | primitive | placement | **placement** |
| Hand features? | no | yes | no | yes | **no** |
| Lookahead? | no | yes | no | yes (depth 2) | **no** |
| Budget | 10M steps | CEM gens | 100M steps (~7.8 h) | CEM, 12 gens × 24 pop | **12M steps (2.0 h)** |
| Result | **0 lines** | **37 lines** | **1.04 mean lines** | **198 @ 500 cap; never tops out** | **5.60 mean lines** |

**The three numbers that carry the whole defense:** `0 → 1.04 → 5.60 → 198.1`.
Pixels to structured obs (0 → 1.04): the reward signal appears. Keypress to placement
(1.04 → 5.60, **the one controlled change**): 5.4×, but only 2.3% of the gap. Adding
hand features + lookahead + CEM (5.60 → 198.1): everything else.

**Files:** `agents/ale/pure_rl_ale_agent.py` · `agents/ale/ale_tetris_agent.py` ·
`agents/custom/pure_rl_custom_agent.py` · `agents/custom/tetris_custom_agent.py` ·
`agents/custom/afterstate_custom_agent.py` · engine in `packages/tetris_env/` ·
boundaries in `AGENTS.md` · full write-up in `docs/REPORT.md`.

**Live demo — have this ready to run:** `python artifacts/best_plays/live_play.py`
plays Track 4 in real time, indefinitely. `--track 3` shows the RL agent. Videos of
all five are in `artifacts/best_plays/`. Showing him Track 1 stacking pieces into a
tower and dying, next to Track 4 playing forever, makes the whole argument in 30
seconds without you saying a word. **Then show Track 3 next to Track 5** — same
algorithm, same network, one changed action space, visibly flatter stack — and that
makes the *scientific* argument just as fast.
