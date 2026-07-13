# Code Walkthrough — for the "explain this line" part of the Q&A

Companion to `docs/QA_PREP.md`. That file defends the *project*. This file teaches
you the *machinery*, from zero, so that when he points at a line you can say what it
does and why it's there.

Read §0 first. Then §1–§4 are the four things he said he'd ask. §5 is the drill:
the lines most likely to get pointed at, each with an answer.

---

## 0. How to survive "what does this line do?"

**The technique, in order. Use it every single time — even when you know the answer,
because it keeps your pace even and stops you blurting something wrong.**

1. **Read the line out loud.** This is not stalling; it is what engineers actually
   do. It buys you three seconds and it often *tells you the answer* — the variable
   names in this repo are honest.
2. **Say what it does mechanically.** "This takes the board, compares every cell to
   zero, and gives me a boolean grid." Mechanical description is almost always safe:
   it's just reading.
3. **Then say why it's there.** "…and I need booleans because the next line counts
   holes, and `cumsum` on a boolean is how I find cells that have something above
   them."
4. **If you don't know: say the true thing you *do* know.** *"I'd have to trace that
   one. What I can tell you is the function it's in — `count_holes` — returns the
   number of empty cells with at least one filled cell above them, and that's the
   feature with the biggest weight in Track 4."* That is a **good** answer. It shows
   you know the architecture even if you don't have that line memorised.

**Never do this:** invent a specific reason. "It's for performance" or "that's a
standard trick" when you don't know is the one move that will actually sink you —
he'll ask one follow-up and it collapses. *"I don't remember why I wrote that"* costs
you nothing. A bluff that unravels costs you everything.

**The framing that saves you:** you did not write every line from a blank page. You
built a system, used a standard library (Stable-Baselines3), and adapted known
algorithms (PPO, CEM, SRS kick tables, Dellacherie features). **That is what real
engineering is.** If he asks "did you write this yourself?", the honest answer is:
*"I wrote the engine, the environments, the feature functions and the training
scripts. PPO comes from Stable-Baselines3 — I configured it, I didn't reimplement it.
The kick tables are the published SRS tables. The feature set is Dellacherie's, from
Thiery & Scherrer."* Knowing **where your work ends and the library begins** is a
sign of maturity, not weakness.

---

## 1. Game mechanics — your custom engine (Tracks 3, 4, 5)

> **Know which track runs on which game. Do not mix these up in the room — the whole
> 2×2 design collapses if you do.**
>
> | Track | Environment |
> | --- | --- |
> | **1, 2** | **ALE `ALE/Tetris-v5`** — the Atari 2600 ROM. Mechanics in **§1B**. |
> | **3, 4, 5** | **Your custom engine** — `engine.py`. Mechanics in this section. |
>
> Track 5 is a *controlled* comparison against Track 3 **only because they share the
> custom engine.** If they were different games, it would prove nothing.

**File: `packages/tetris_env/tetris_env/engine.py`.** You wrote this. It is a
standard "guideline" Tetris.

### The board

```python
VISIBLE_ROWS = 20        # engine.py:10
HIDDEN_ROWS  = 2         # two spawn rows above the visible field
ROWS = 22
COLS = 10
self.board = np.zeros((ROWS, COLS), dtype=np.uint8)   # engine.py:129
```

The board is a **22×10 grid of 0s and 1s**. Not colours — just "filled or empty".
Row 0 is the **top**. The two hidden rows are where pieces spawn, so a piece can
exist before it's visible. `visible_board` (line 155) returns `board[2:]` — the 20×10
the player sees, and the 200 numbers the agent sees.

**Say this if asked why the board is binary:** once a piece locks, its colour is
irrelevant to every decision — line clears, holes, heights all depend only on
filled/empty. Keeping it `uint8` binary makes the board cheap to copy, and Track 4
copies the board thousands of times per second to search placements.

### The seven pieces and rotation

`PIECES` (line 26) is a dict: piece name → 4 rotations → 4 cells each, as
`(row, col)` offsets. So `PIECES["T"][0]` is the T piece in its spawn rotation.

`cells()` (line 163) adds the piece's current `(row, col)` to those offsets to get its
four **absolute** board positions. `collides()` (line 169) then asks: is any of those
four cells off the side, below the floor, or already filled? That single function is
the whole physics of the game.

### Gravity — the most important subtlety in the file

```python
# engine.py:207-209
if action not in (Action.SOFT_DROP, Action.HARD_DROP) and not self.game_over:
    if not self._try_shift(1, 0):
        reward += self._lock_piece()
```

**After *every* action that isn't a drop, the piece falls one row.** Move left →
it also falls. Rotate → it also falls. If it *can't* fall (something's under it), it
**locks in place** immediately.

This is the single fact that explains why Track 3 is hard, so understand it cold:
the agent doesn't get to line the piece up at leisure. Every keypress costs one row
of altitude. It has ~9 actions per piece before the piece hits the stack, and it must
have finished rotating and shifting by then.

### 7-bag randomiser

```python
# engine.py:266-270
def _fill_queue(self):
    while len(self.queue) < 7:
        bag = list(PIECES)
        self.rng.shuffle(bag)
        self.queue.extend(bag)
```

Not uniformly random. Take all 7 pieces, shuffle, deal them out; when you run low,
shuffle a fresh bag. **This is the modern Tetris standard** — it guarantees you never
go more than 12 pieces without an I-piece, and it's why you can't be starved to death
by bad luck. If he asks "is your randomiser fair?" — this is the answer, and it is
the *correct* one.

### SRS wall kicks

```python
# engine.py:234-238
for drow, dcol in kicks[(old, new)]:
    rotated = PieceState(..., self.current.row + drow, self.current.col + dcol, new)
    if not self.collides(rotated):
        self.current = rotated
        return True
```

When you rotate against a wall or the stack, real Tetris doesn't just refuse — it
tries to **nudge the piece** into a nearby legal spot. `JLSTZ_KICKS` and `I_KICKS`
(lines 71–91) are lookup tables keyed by `(from_rotation, to_rotation)`, each giving
**five candidate offsets to try in order**. First one that doesn't collide wins. The
first is always `(0,0)` — "try rotating in place".

These are the **published Super Rotation System tables**. You did not invent them.
Say that.

*(This also causes a real quirk you should know about: a rotation can move the piece
sideways. That's why `replay.py` has to verify that its reconstructed action sequence
actually lands on the intended board — see §5.)*

### Locking, line clears, top-out

```python
# engine.py:241-255
def _lock_piece(self):
    for row, col in self.cells():
        if row < 0:
            self.game_over = True
            return -1000.0
        self.board[row, col] = 1
    self.pieces += 1
    cleared = self._clear_lines()
    line_score = LINE_SCORES[cleared] * (self.level + 1)
    ...
```

Stamp the four cells into the board, count the piece, clear any full rows, score.
`LINE_SCORES = (0, 100, 300, 500, 800)` (line 93) — 1/2/3/4 lines. Note it's
**superlinear**: a Tetris (4 lines at once) pays 800, not 400. That's the real game's
design and it's why "should I hold out for a Tetris?" is a real strategic question.

```python
# engine.py:257-264
def _clear_lines(self):
    full = np.all(self.board == 1, axis=1)   # which rows are completely filled?
    cleared = int(full.sum())
    if cleared:
        kept = self.board[~full]             # keep the rows that aren't full
        self.board[:] = 0
        self.board[ROWS - len(kept):] = kept  # push them all down to the bottom
    return cleared
```

That's the whole line-clear. No loops — pick the surviving rows, drop them to the
bottom, zero everything above.

**Top-out** happens two ways: `spawn()` (line 181) creates a piece and it immediately
collides → game over; or `_lock_piece` locks a cell above the board.

---

## 1B. Game mechanics — ALE Tetris (Tracks 1 and 2)

**This is a different game, and you did not write it.** `ALE/Tetris-v5` is an
Atari 2600 ROM running under emulation (Stella, via the Arcade Learning
Environment). You cannot read its internals — the only things you get are **the
screen, an action, and a reward.** Everything below is verified from the live
environment, not from memory.

### The interface

```python
# agents/ale/env.py:19-26
gym.make("ALE/Tetris-v5",
         obs_type="rgb",                    # 210 x 160 x 3 uint8 screen
         repeat_action_probability=sticky,  # sticky actions
         frameskip=4,
         full_action_space=False)           # -> minimal action set
```

| | ALE Tetris | Your engine |
| --- | --- | --- |
| **Observation** | `210 × 160 × 3` RGB **screen** | structured board arrays |
| **Actions** | **5** | 7 |
| **Reward** | **lines cleared, nothing else** | shaped (see §2) |
| **Playfield** | 20 × 10 (measured by your decoder) | 20 × 10 visible |

### The action set — and the thing that surprises people

```
['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN']     # Discrete(5)
```

- **`FIRE` = rotate.** One direction only. There is no counter-clockwise rotate.
- **`DOWN` = soft drop.** It nudges the piece down one step.
- **There is no hard drop.** This is the big one. Your custom engine has
  `HARD_DROP` — one action, piece slams to the bottom, done. **ALE has no such
  action.** To get a piece down you must either wait for gravity or hold `DOWN`
  repeatedly.

That is not a theoretical point — it's visible in your own Track 2 code:

```python
# ale_tetris_agent.py:546
actions = [FIRE] * fires + [move] * moves + [DOWN] * 90
```

> **Rotate a few times, shift sideways a few times, then press DOWN ninety times.**
> That is how Track 2 executes one placement, and it's a direct consequence of the
> ROM having no hard-drop action.

### The reward — and why Track 1 scores exactly zero

**ALE Tetris's native reward *is* the number of lines cleared.** Nothing else. No
points for surviving, none for dropping, none for placing a piece. Your agent's
reward of `37.0` literally means **37 lines**.

This is the single most important mechanical fact in your whole project, because it
is *why Track 1 gets 0 and it isn't PPO's fault*:

- To clear one line, a random policy must place ~10 pieces correctly in a row.
- It essentially never does this by chance.
- So the reward is **0 on every single transition for the entire 10M steps.**
- Zero reward → zero advantage → **zero policy gradient.**

PPO didn't fail to learn. **It had nothing to learn from.** You cannot fix a
*missing* signal by collecting more samples of it. Your custom engine exists
precisely to inject a denser reward (`+0.25` per piece) so a gradient exists at all —
and Track 3 immediately escapes zero. That contrast *is* your experiment.

*(Sanity check, run live: NOOP-only from seed 0 tops out after 416 agent steps with
total reward `0.0`. Gravity does all the work, nothing clears.)*

### Frame-skip 4

The agent acts once, and the emulator runs that action for **4 frames**. So "10M
agent steps" = **40M emulator frames**. Say the unit out loud — the canonical Atari
benchmark is 200M *frames* (= 50M agent steps), so your budget was ~20% of it, not
the ~2% a naive "10M vs 200M" reading suggests. He may well test you on this.

### Sticky actions

`repeat_action_probability=0.25`: with 25% chance the emulator **ignores your new
action and repeats the previous one** (Machado et al. 2018). It exists to stop agents
memorising one fixed button sequence — it forces genuine responsiveness to the screen.

**And it matters more here than anywhere else in your project, for a reason you must
volunteer:** you verified that **the ALE seed does not change Tetris's piece
sequence** — seeds 0, 1, 2, 42 all produce a bit-identical trajectory. The ROM's piece
generator isn't driven by ALE's seed. So sticky actions are the **only** source of
randomness in that environment, and your "10 seeds → 37 lines every time" is
**one game played ten times**, not ten independent samples. See `QA_PREP.md` §6.2.

### What ALE does *not* give you

- **No 7-bag guarantee.** Your engine's `_fill_queue` shuffles all seven pieces so you
  can't be starved. The ROM has its own generator and you don't control it.
- **No next-piece preview available to your planner.** Track 2 decodes only the
  playfield, so it is **depth-1 greedy** — it scores the current piece's placements and
  picks the best. Track 4, on the custom engine, reads the real queue and searches
  **2 ply**. If he asks why Track 2 is so much weaker than Track 4, this is a large
  part of the answer (along with 5 clumsy actions vs a clean placement commit).
- **No board array.** Track 2 has to *reconstruct* the board from pixels — that's
  `decode_board` (§5), sampling the centre of each of the 200 cells and asking "is
  this neither background-grey nor black?"

### The one-paragraph version, if he asks "how does the Atari game work?"

> It's the Atari 2600 ROM under emulation, so I only get the 210×160 screen, five
> actions — NOOP, FIRE to rotate, LEFT, RIGHT, and DOWN to soft-drop — and a reward
> that is exactly the number of lines cleared. Notably there's **no hard drop**, so my
> planner has to press DOWN about ninety times to seat a piece. And because the reward
> is *only* line clears, a randomly-initialised policy sees zero reward for its entire
> training run, which is why my pure-RL agent on ALE scores exactly zero — the policy
> gradient is zero, so there's nothing to descend. That's the finding, not a bug.

---

## 2. Algorithm input — what each agent actually sees

This is the "algorithm input" question. **Know the exact numbers.** He may well ask
"how many inputs does your network have?" and a confident, correct number is worth a
lot.

| Track | Input | Size | Type |
| --- | --- | --- | --- |
| 1 (ALE pure RL) | 4 stacked grayscale game screens | **84 × 84 × 4** | pixels |
| 2 (ALE tools) | decoded board + features | 20×10 grid → 9 features | decoded |
| 3 (custom pure RL) | board + active piece + position + 2 piece IDs | **417 floats** | structured |
| 4 (custom tools) | 10 features **per candidate placement** | 10 floats × ~34 | engineered |
| 5 (custom afterstate) | board + 2 piece IDs | **214 floats** | structured |

### Track 3's 417 — be able to break this down

```python
# pure_rl_custom_agent.py:29
size = VISIBLE_ROWS * COLS * 2 + PIECE_COUNT * 2 + 3
```

- `20 × 10 = 200` — the **locked stack** (`board`)
- `20 × 10 = 200` — the **falling piece**, as its own separate grid (`active`)
- `7` — one-hot of the current piece, `7` — one-hot of the next piece
- `3` — the piece's `row/22`, `col/10`, `rotation/3` (normalised to 0–1)

**= 417 floats.** Flat vector → that's why it's an `MlpPolicy` (a plain fully-connected
net) and not a CNN.

**Why board and piece are two separate 200-cell grids** rather than one grid with the
piece drawn in: so the network can tell "this cell is a falling piece I can still
move" apart from "this cell is locked forever". Superimposing them would destroy that
distinction. Good answer if he asks.

**One-hot** = a length-7 vector that is all zeros except a single 1 at the index of
the piece. You use it instead of the number 0–6 because the network would otherwise
think piece 6 is "bigger than" piece 1, which is meaningless.

### Track 5's 214 — and the point worth making

```python
# placement_env.py:43
OBS_SIZE = VISIBLE_ROWS * COLS + PIECE_COUNT * 2   # 200 + 7 + 7 = 214
```

Track 5 **drops** the `active` grid and the 3 position numbers. Why? Because when one
action places a whole piece, "where is the piece right now, mid-flight" is a question
that no longer exists. There is no mid-flight.

**Say this if you get the chance — it's a genuinely sharp observation:** Track 5 sees
*less* than Track 3 (214 vs 417 inputs) and performs **5.4× better**. The extra 203
numbers Track 3 gets aren't helping it; they're describing a problem — piloting a
piece through the air — that Track 5 doesn't have to solve.

### Track 1's pixels

`AtariPreprocessing` (`pure_rl_ale_agent.py:67`) does the standard Atari pipeline:
RGB → grayscale, resize to 84×84, **frame-skip 4** (act once, repeat that action for 4
emulator frames), and up to 30 random no-ops at reset.

```python
env = VecFrameStack(env, n_stack=4, channels_order="last")   # pure_rl_ale_agent.py:86
```

**Why stack 4 frames:** one still image cannot tell you which way a piece is *moving*,
or how fast. Stacking the last 4 gives the network motion. This is the standard trick
from the 2015 DQN Atari paper — cite that if asked.

**Sticky actions** (`repeat_action_probability=0.25`, `env.py:23`): with 25% chance the
emulator ignores your new action and repeats your previous one. It exists to stop
agents memorising one fixed winning button sequence — it forces them to actually
respond to the screen. (Machado et al. 2018.)

### The reward — and the best story in your project

```python
# gym_env.py:103
reward = self.line_reward * float(cleared * cleared) + self.piece_reward * float(placed)
if terminated:
    reward -= self.top_out_penalty
```

`10 × cleared² + 0.25 per piece − 10 on top-out`.

- **Squared** so a Tetris (4 at once) pays 16× a single, not 4×. Matches the game's
  own superlinear scoring.
- **+0.25 per piece** is the survival term: a small, *dense* signal so the agent gets
  feedback on almost every step instead of only on the rare line clear.
- **−10 on top-out** for dying.

**And now the story he will love.** The env has a *second* reward mode, `score`
(line 98), which just passes through the engine's own score. It failed, and the
comment at `gym_env.py:21-24` says why: the engine pays **+1 per cell of soft drop and
+2 per cell of hard drop** (`engine.py:200, 216`). Those points are **dense** — every
single piece pays them. A line clear pays 100 but is **rare**. So PPO correctly
learned to **slam pieces down as fast as possible and never clear a line.** It was
maximising exactly what you asked for.

That is a textbook **reward-misspecification** story, you found it yourself, and you
fixed it by removing drop points entirely. Have it ready.

---

## 3. Structure of the agent

Two completely different things. Know which is which.

### PPO (Tracks 1, 3, 5) — this IS reinforcement learning

**PPO = Proximal Policy Optimization.** It's the default modern policy-gradient
algorithm. Here's the whole thing, honestly, in plain terms:

**The two networks (the "actor-critic"):**

```python
# pure_rl_custom_agent.py:166-171
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],   # the ACTOR: state -> which action to take
        "vf": [256, 256],   # the CRITIC: state -> how good is this state?
    }
}
```

- **Actor** (`pi`): takes the 417 numbers, passes them through two hidden layers of
  256 neurons, and outputs a **probability for each of the 7 actions**. That's the
  policy, π(a|s).
- **Critic** (`vf`): same input, but outputs **one number** — "from this board, how
  much total future reward do I expect?" That's the value function, V(s).

**The training loop:**

1. **Collect.** Run the current policy for `n_steps=512` steps in each of **8 parallel
   environments** → 4,096 transitions of `(state, action, reward, next state)`.
2. **Score each action taken, with hindsight.** Compute the **advantage**: did this
   action turn out better or worse than the critic *predicted*? `A = actual return −
   V(s)`. Positive advantage = "that was better than expected, do it more."
   (The exact recipe is **GAE**, `gae_lambda=0.95` — a weighted blend of short- and
   long-horizon estimates, to trade off bias and variance.)
3. **Update.** Nudge the actor to make positive-advantage actions more likely and
   negative-advantage ones less likely. Nudge the critic to predict returns better.
4. Repeat, 10 times over the same batch (`n_epochs=10`, minibatches of 256), then
   throw the data away and collect fresh.

**What the "Proximal" means — this is the one thing that makes it PPO and he may well
ask:** if you take a big greedy step, you can destroy a working policy in one update
and never recover. So PPO **clips** the update: it computes the ratio
`r = π_new(a|s) / π_old(a|s)` and refuses to let a single update change any action's
probability by more than `clip_range = 0.2` (±20%) in a way that helps the objective.
It keeps the new policy **proximal** — close — to the old one. That's the whole idea,
and it's the whole name.

**The other hyperparameters, and what to say:**

| Param | Value | What it means |
| --- | --- | --- |
| `learning_rate` | 3e-4 | Adam step size. The standard default. |
| `gamma` (γ) | 0.995 | **Discount.** How much future reward is worth now. Horizon ≈ 1/(1−γ) = **200 steps** ≈ 22 pieces. Set high *because* Tetris rewards are delayed. |
| `gae_lambda` | 0.95 | Bias/variance trade-off in the advantage estimate. |
| `clip_range` | 0.2 | The ±20% trust region above. |
| `ent_coef` | 0.01 | **Entropy bonus.** Pays the agent to keep its action distribution uncertain, so it keeps exploring instead of collapsing to one action early. |
| `vf_coef` | 0.5 | How much the critic's error counts in the total loss. |
| `n_steps` × `n_envs` | 512 × 8 | 4,096 transitions per update. |

**VecNormalize** (`pure_rl_custom_agent.py:122`) keeps a running mean and standard
deviation of the observations and rescales them to roughly mean 0, variance 1. Neural
nets train badly on inputs at wildly different scales. **This is why
`vec_normalize.pkl` sits next to every model checkpoint and why the model is useless
without it** — load the policy with the wrong input statistics and you get garbage.
If he asks "what's this .pkl file?", that's the answer.

**If he asks "why PPO and not DQN?"** — honest answer: PPO is the robust default, it
handles this discrete action space fine, and SB3's implementation is well-tested.
You did not try DQN or Rainbow. Say so; it's on the "I don't know" list in
`QA_PREP.md` §8.

### CEM (Tracks 2, 4) — this is NOT reinforcement learning

**Concede that immediately if he raises it.** No value function, no TD error, no
gradient, no bootstrapping. It's **derivative-free black-box optimisation** — you
never differentiate through the game.

The agent is just **10 numbers** (`features.py:8-19`), one weight per board feature.
To choose a move: enumerate every legal placement, score each with a dot product,
take the best.

```python
# tetris_custom_agent.py:19-20
def _placement_value(weights, placement):
    return float(np.dot(weights, placement_features(placement)))
```

**That dot product is the entire policy.** Ten multiplications and an addition.

**CEM = Cross-Entropy Method.** How you find good weights:

```python
# tetris_custom_agent.py:214
population = mean[None, :] + rng.standard_normal((args.population, len(mean))) * std[None, :]
# ...play games with each, get a fitness score for each...
# tetris_custom_agent.py:231-233
elite = population[np.argsort(fitness)[-elite_count:]]   # keep the top ~25%
mean  = elite.mean(axis=0)                                # re-centre on the winners
std   = elite.std(axis=0) + args.noise_floor              # re-spread around them
```

In English: **keep a bell curve over the 10 weights. Sample ~30 candidate weight-sets
from it. Play games with each. Throw away all but the best few. Recompute the bell
curve from those survivors. Repeat.** It's evolution — survival of the fittest weight
vector.

`noise_floor` is the one non-obvious bit: without it, `std` shrinks toward zero as the
elites converge, the search freezes, and you're stuck in a local optimum. Adding a
small constant keeps it exploring. That's **"noisy CEM"** — Szita & Lőrincz (2006),
who used exactly this on Tetris and beat the prior RL results by two orders of
magnitude.

**And the lookahead** (`tetris_custom_agent.py:111`):

```python
return immediate + lookahead_weight * future_value
```

Score a placement by *this* board plus 0.35 × the best it could do *next* piece
(it knows the next piece from the queue, and searches depth 2 over the top 4
candidates). 0.35 discounts the future because the plan may not survive contact.

---

## 4. The comparison, in one breath

If he asks you to tie it together:

> Tracks 3 and 5 are the same algorithm — same PPO, same 2×256 network, same reward,
> same hyperparameters. The only difference is that Track 3 outputs a **keypress** and
> Track 5 outputs a **placement**. That took it from 1.04 lines to 5.60. Track 4 then
> adds hand-authored features, a 2-ply lookahead, and CEM instead of PPO, and gets
> 198. So the action abstraction is worth about 5×, and the features and search are
> worth the other 190×. I had assumed the opposite, and Track 5 is what corrected me.

---

## 5. The drill — lines he is most likely to point at

Cover the right-hand column and work down. If you can do these fifteen, you can
handle the code portion.

### `engine.py:207-209`
```python
if action not in (Action.SOFT_DROP, Action.HARD_DROP) and not self.game_over:
    if not self._try_shift(1, 0):
        reward += self._lock_piece()
```
> **Gravity.** After any action that isn't a drop, the piece falls one row. If it
> can't fall, it locks. This is why the pure-RL agent has only ~9 actions per piece
> and why the credit-assignment problem is hard for it.

### `engine.py:234-238` (the kick loop)
> **SRS wall kicks.** When a rotation collides, try five published offsets in order
> and take the first that fits — that's how a piece "kicks" off a wall. Standard Super
> Rotation System tables; I didn't invent them. Side effect: a rotation can shift the
> piece sideways, which is why my placement-replay code has to verify its output.

### `engine.py:258` — `full = np.all(self.board == 1, axis=1)`
> Gives me a boolean per row: is that row completely filled? `axis=1` collapses across
> the 10 columns. `full.sum()` is then the number of lines cleared.

### `engine.py:261-263`
```python
kept = self.board[~full]
self.board[:] = 0
self.board[ROWS - len(kept):] = kept
```
> The line clear. Keep the rows that *aren't* full, blank the board, and drop the
> survivors to the bottom. No loop needed.

### `engine.py:244-246`
```python
if row < 0:
    self.game_over = True
    return -1000.0
```
> Top-out: the piece locked above the top of the board. The −1000 is the penalty in
> the legacy `score` reward mode. **In the mode I actually trained on (`lines`) this
> value is ignored** — the environment computes its own reward.

### `engine.py:266-270` — `_fill_queue`
> **7-bag randomiser.** Shuffle all seven pieces and deal them out, then reshuffle.
> The modern Tetris standard — it guarantees you can't be starved of a piece.

### `engine.py:319` — `landing_row = min(int(tops[c + col]) - 1 - b for c, b in bottoms)`
> Fast hard-drop for the planner. `tops` is the height of the first filled cell in
> each column, precomputed once. For each column the piece occupies, work out how far
> its lowest cell in that column can fall; the answer is the *smallest* of those. It's
> a vectorised drop, so I don't simulate the fall cell-by-cell for all ~34 placements.
> **The `if landing_row < start_row` branch right below is the fallback** for when the
> piece spawns inside a covered pocket, where that shortcut is wrong — then I do the
> slow exact drop.

### `engine.py:307-308`
```python
footprint = tuple(sorted((r - min_row, c + col) for r, c in shape))
if footprint in seen: continue
```
> Deduplication. The O-piece is identical in all four rotations, and I/S/Z repeat every
> two, so different `(rotation, column)` pairs can produce **the same final shape**.
> I hash the resulting footprint and enumerate each distinct one once. Roughly halves
> the search for those pieces.

### `gym_env.py:92-96`
```python
if not terminated and self._steps_since_lock >= self.max_steps_per_piece:
    forced = self.game.step(Action.HARD_DROP)
```
> A **lock delay / hover guard**, and it fixes a real bug I hit. Rotation kicks can
> push a piece *upward*, so a deterministic policy could rotate forever, cancel
> gravity, and the episode would never end — training hung. After 50 non-locking steps
> on one piece I force a hard drop. Real Tetris has the same idea (move-limit lock
> delay).

### `gym_env.py:103` — the reward
```python
reward = self.line_reward * float(cleared * cleared) + self.piece_reward * float(placed)
```
> `10 × lines² + 0.25 per piece`, minus 10 on top-out. Squared so a Tetris is worth
> 16× a single, not 4×. The per-piece term is a dense survival signal — without it the
> reward is almost always zero and the policy gradient has nothing to work with.

### `gym_env.py:73-74`
```python
game_seed = int(self.np_random.integers(0, 2**31 - 1))
self.game = TetrisGame(seed=game_seed)
```
> Every episode draws a **fresh piece sequence** from the env's seeded RNG. So episodes
> differ, but the whole run is reproducible from one seed. (This is genuine seed
> diversity — worth contrasting with ALE, where the seed does **not** change the piece
> sequence at all. See `QA_PREP.md` §6.2.)

### `placement_env.py:112-115` — `_resolve`
```python
want_rot, want_col = divmod(action, COLS)
same_rot = [p for p in placements if p.rotation == want_rot]
pool = same_rot if same_rot else placements
return min(pool, key=lambda p: (abs(p.col - want_col), abs(p.rotation - want_rot)))
```
> Track 5's action is a number 0–39; `divmod` by 10 splits it into **(rotation,
> column)**. But not all 40 are legal — symmetric pieces have fewer distinct rotations,
> and wide pieces can't reach every column. Rather than action-masking (SB3's PPO
> doesn't support it), an illegal request **snaps to the nearest legal placement** —
> matching rotation first, then column. It's deterministic, so the policy can learn
> the mapping.

### `placement_env.py:88-93`
```python
self.game.board = placement.board.copy()
self.game.lines += placement.lines
...
self.game.spawn()
```
> Committing a placement. `enumerate_placements` already computed the resulting board
> including any line clears, so the env just **adopts that board** and spawns the next
> piece. That's what makes one step = one piece. `tests/test_render_and_replay.py`
> verifies this teleport lands on exactly the same board the primitive actions would
> have reached.

### `features.py:32-35` — `count_holes`
```python
visible = board[HIDDEN_ROWS:] != 0
covered = np.cumsum(visible, axis=0) > 0
return int((~visible & covered).sum())
```
> **A hole is an empty cell with something above it** — the thing you can't fill
> without clearing the rows on top of it. `cumsum` down each column is nonzero from
> the first filled cell onward, so `covered` marks "has something above". Empty AND
> covered = hole. **This is the most heavily weighted feature in Track 4 (−36.5).**

### `features.py:54-60` — `wells`
```python
depth = np.minimum(left, right) - heights
return int((depth * (depth + 1) // 2).sum())
```
> A well is a column much lower than **both** its neighbours — a deep narrow gap only
> an I-piece can fill. `depth*(depth+1)/2` is the triangular number: it makes deep
> wells *disproportionately* worse than shallow ones, which is the standard Dellacherie
> formulation.

### `features.py:102` — `return raw / FEATURE_SCALE`
> Normalisation. The raw features live on wildly different scales — `score_delta` can
> be 2000, `lines` is 0–4. Without dividing by a per-feature constant, CEM's single
> shared standard deviation would take absurdly large steps on one weight and
> useless tiny ones on another.

### `tetris_custom_agent.py:20` — `np.dot(weights, placement_features(placement))`
> **The entire Track 4 policy.** Ten features, ten weights, one dot product per
> candidate placement; pick the highest. No neural network anywhere in this track.

### `tetris_custom_agent.py:231-233`
```python
elite = population[np.argsort(fitness)[-elite_count:]]
mean  = elite.mean(axis=0)
std   = elite.std(axis=0) + args.noise_floor
```
> The CEM update. Sort the candidate weight-sets by how well they played, keep the top
> ~25%, and recompute the mean and spread of the bell curve from just those survivors.
> `noise_floor` stops the spread collapsing to zero and freezing the search.

### `pure_rl_custom_agent.py:166-171` — `net_arch`
> Actor-critic. `pi` is the policy head (417 inputs → 256 → 256 → 7 action
> probabilities); `vf` is the value head (same input → 256 → 256 → one number, the
> expected future reward from this state). PPO trains both together.

### `ale/env.py:29-32` — `estimate_atari_score`
```python
def estimate_atari_score(lines: int) -> int:
    return int(lines * 100)
```
> ⚠ **Volunteer the truth here.** This is **my own convention, not the ROM's score.**
> ALE Tetris's native reward *is lines cleared* — my agent gets reward 37.0, meaning
> 37 lines. This function just multiplies by 100 for a friendlier headline. **If he
> asks what you scored, say "37 lines"** — never quote 3,700 as if the game printed it.

### `ale_tetris_agent.py:81-86` — `decode_board`
```python
patches = frame[_row_idx, _col_idx]
med = np.median(patches, axis=2).astype(np.int16)
is_gray = np.all(np.abs(med - BG_GRAY) <= GRAY_TOL, axis=-1)
is_black = np.all(med <= 12, axis=-1)
return (~(is_gray | is_black)).astype(np.uint8)
```
> Track 2's eyes. I precomputed the pixel coordinate of the **centre of each of the
> 200 board cells**, sample a 3×3 patch at each, take the **median** (robust to a
> stray pixel), and call the cell filled if it's neither background grey nor black.
> Result: the 210×160 RGB screen becomes a 20×10 binary grid — the same representation
> my custom engine uses, which is what lets the same planner drive both.

### `ale_tetris_agent.py:17` — the ALE action set
```python
NOOP, FIRE, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4
```
> The Atari ROM's **entire** action set — five actions. `FIRE` rotates (one direction
> only, no counter-clockwise). `DOWN` is a **soft** drop. **There is no hard drop** —
> unlike my custom engine, which has one. That absence shapes the whole track.

### `ale_tetris_agent.py:546` — how Track 2 seats a piece
```python
actions = [FIRE] * fires + [move] * moves + [DOWN] * 90
```
> Executing one chosen placement on ALE: rotate `n` times, shift sideways `n` times,
> then **press DOWN ninety times** to force the piece to the bottom. It looks absurd
> until you know there's no hard-drop action in the ROM — so "get this piece down" is
> literally spam-the-soft-drop. It's also why Track 2 is open-loop *within* a piece but
> closed-loop *across* pieces: it re-decodes the screen every piece and re-plans, which
> is why sticky actions at 0.25 don't hurt it.

---

## 6. Vocabulary — so nothing he says lands as a foreign word

| He says | It means |
| --- | --- |
| **Policy** | The function state → action. Your actor network (or, in Track 4, the 10 weights). |
| **Value function / critic** | Estimate of total future reward from a state. |
| **Return** | Sum of (discounted) future rewards. |
| **Advantage** | How much better an action was than the critic expected. Positive → reinforce it. |
| **On-policy** | Learns only from data the *current* policy generated (PPO). Opposite: off-policy (DQN, replay buffers). |
| **Model-free** | Doesn't simulate the environment to plan. Tracks 1/3/5 are model-free. Track 4 clones the engine, so it's **model-based** search. |
| **Afterstate** | The board *after* your piece lands, before the next spawns. Track 5's action space is defined over these. |
| **Credit assignment** | Figuring out *which* of your past actions caused a reward that arrived much later. The core difficulty in Track 3. |
| **Reward shaping** | Adding extra reward terms (your +0.25/piece) to give a denser learning signal. |
| **Sparse reward** | Reward almost always zero. ALE Tetris. The reason Track 1 gets 0. |
| **Rollout** | One run of the policy through the env to collect data. |
| **Entropy bonus** | Reward for keeping the action distribution random, to sustain exploration. |
| **Sticky actions** | 25% chance the emulator repeats your last action. Anti-memorisation. |
| **Frame stacking** | Feeding the last 4 screens so the net can perceive motion. |

---

## 7. The three traps

1. **"What score did you get?"** → **"37 lines."** Not 3,700 (§5, `estimate_atari_score`).
2. **"You tested 10 seeds?"** → the ALE seed **does not change the piece sequence**.
   It's one game, ten times. Volunteer it (`QA_PREP.md` §6.2).
3. **"Track 4 hits a 200-line ceiling?"** → that's the **500-piece cap**, not the
   agent. Uncapped it never tops out — 10,000 pieces, 3,997 lines, still alive.

---

## 8. Ninety-second self-test

If you can answer these without looking, you're ready:

0. **Which tracks run on ALE, and which on your engine?** *(ALE: **1 and 2**. Custom
   engine: **3, 4, 5**. Get this wrong and the 2×2 design falls apart — Track 5 only
   isolates the action space **because** it shares the custom engine with Track 3.)*
1. How big is the board, and why are there 22 rows and not 20? *(Custom engine.)*
2. What happens to the falling piece when the agent presses LEFT? *(It moves left
   **and falls one row.** Custom engine.)*
2b. **How many actions does ALE Tetris have, and which one is missing that your engine
   has?** *(Five — NOOP, FIRE=rotate, LEFT, RIGHT, DOWN=soft drop. **No hard drop.**)*
2c. **What is ALE Tetris's reward?** *(Lines cleared. Nothing else — which is exactly
   why Track 1's policy gradient is zero and it scores 0.)*
3. How many numbers go into the Track 3 network? What are they? *(417.)*
4. What does the critic output? *(One number: expected future reward from this state.)*
5. What does the "clip" in PPO clip, and why? *(The policy-update ratio, to ±20%, so
   one bad update can't destroy a working policy.)*
6. What is a hole, and how do you count it in one line of numpy?
7. What are the three steps of CEM? *(Sample a population from a Gaussian → keep the
   elite by fitness → refit the Gaussian to the elite.)*
8. Why did the `score` reward mode fail? *(Drop points are dense, line clears are
   rare; PPO learned to drop fast and never clear.)*
9. What is the single difference between Track 3 and Track 5? *(Keypress vs placement.
   Nothing else.)*
10. What did Track 5 prove? *(The action abstraction is worth 5.4×, but only 2.3% of
    the gap — the hand-authored features carry the rest. It refuted my own claim.)*
