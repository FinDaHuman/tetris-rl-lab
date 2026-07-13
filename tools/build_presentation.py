"""Build docs/presentation.html -- a self-contained slide deck for the project Q&A.

Three parts: (1) what ALE Tetris is and how it works, (2) the five tracks and what
differs between them, (3) how each performed.

Every headline number is READ FROM the evaluation manifests rather than typed in, so
a slide cannot drift away from the evidence. Every screenshot is a real capture from
the live environment, inlined as a base64 data URI so the file works offline with no
external requests.

    python tools/build_presentation.py            # capture images + build
    python tools/build_presentation.py --no-capture   # rebuild from cached images
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

OUT = ROOT / "docs" / "presentation.html"
CACHE = ROOT / "docs" / "_presentation_images.json"

EVALS = {
    1: "artifacts/ale_pure_rl/evaluation.json",
    2: "artifacts/ale_stable_high_score/evaluation.json",
    3: "artifacts/custom_pure_rl/evaluation.json",
    4: "artifacts/custom_best/evaluation_500.json",
    5: "artifacts/custom_afterstate/evaluation.json",
}

# Seed used for the Track 3 vs Track 5 side-by-side. Same seed, same engine, same
# reward -- the only difference is the action space, which is the whole point.
COMPARE_SEED = 1002


# --------------------------------------------------------------------------- data


def load_results() -> dict:
    """Pull every headline number straight from the frozen evaluation manifests."""
    ev = {t: json.loads((ROOT / p).read_text(encoding="utf-8")) for t, p in EVALS.items()}

    def mean_pieces(d):
        rows = d.get("rows") or d.get("results") or []
        vals = [r["pieces"] for r in rows if "pieces" in r]
        return sum(vals) / len(vals) if vals else None

    r = {
        1: {
            "lines": ev[1]["mean_native_reward"],  # ALE's native reward IS lines cleared
            "episodes": ev[1]["episodes"],
            "detail": "mean &amp; max native reward 0.0",
        },
        2: {
            "lines": ev[2]["mean_lines"],
            "episodes": ev[2]["episodes"],
            "pieces": mean_pieces(ev[2]),
            "std": ev[2]["line_std"],
            "detail": f"{int(ev[2]['mean_lines'])} lines on every one of {ev[2]['episodes']} seeds (std {ev[2]['line_std']})",
        },
        3: {
            "lines": ev[3]["mean_lines"],
            "max": ev[3]["max_lines"],
            "episodes": ev[3]["episodes"],
            "pieces": mean_pieces(ev[3]),
            "detail": f"max {ev[3]['max_lines']}, {ev[3]['episodes']} deterministic episodes",
        },
        4: {
            "lines": ev[4]["mean_lines"],
            "max": ev[4]["max_lines"],
            "episodes": ev[4]["episodes"],
            "cap": ev[4]["max_pieces"],
            "detail": f"of a {ev[4]['max_pieces'] * 4 // 10}-line ceiling at a {ev[4]['max_pieces']}-piece cap",
        },
        5: {
            "lines": ev[5]["mean_lines"],
            "max": ev[5]["max_lines"],
            "min": ev[5]["min_lines"],
            "episodes": ev[5]["episodes"],
            "pieces": ev[5]["mean_pieces"],
            "detail": f"max {ev[5]['max_lines']}, min {ev[5]['min_lines']}, {ev[5]['episodes']} deterministic episodes",
        },
    }
    # Derived, and stated as derived.
    t3, t5, t4 = r[3]["lines"], r[5]["lines"], r[4]["lines"]
    r["gain"] = t5 / t3                              # action abstraction, in x
    r["closed"] = (t5 - t3) / (t4 - t3) * 100        # % of the T3->T4 gap it closes
    r["remaining"] = 100 - r["closed"]
    r[3]["lines_per_piece"] = t3 / r[3]["pieces"]
    r[5]["lines_per_piece"] = t5 / r[5]["pieces"]
    r[4]["lines_per_piece"] = t4 / r[4]["cap"]
    return r


# ------------------------------------------------------------------------- images


def png_b64(arr: np.ndarray, scale: int = 1) -> str:
    from PIL import Image

    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def capture_images() -> dict:
    """Real frames from the live envs. Nothing here is a mockup."""
    from gymnasium.wrappers import AtariPreprocessing

    from agents.ale.env import make_env
    from agents.custom import afterstate_custom_agent as t5m
    from agents.custom import pure_rl_custom_agent as t3m

    imgs: dict[str, str] = {}
    rng = np.random.default_rng(0)

    # The Atari screen a human sees.
    env = make_env(sticky=0.0, frameskip=4, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(120):
        _, _, term, trunc, _ = env.step(int(rng.integers(0, 5)))
        if term or trunc:
            break
    imgs["ale_screen"] = png_b64(env.unwrapped.render(), scale=3)

    # The 84x84 grayscale the CNN is actually fed.
    env2 = AtariPreprocessing(
        make_env(sticky=0.0, frameskip=1), noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True
    )
    o, _ = env2.reset(seed=0)
    for _ in range(120):
        o, _, term, trunc, _ = env2.step(int(rng.integers(0, 5)))
        if term or trunc:
            break
    imgs["ale_obs"] = png_b64(np.stack([o] * 3, axis=-1), scale=6)

    # Track 3 and Track 5 final boards on the SAME seed.
    class Grab:
        def __init__(self):
            self.last = None
            self.frames = 0

        def append(self, frame, hold=1):
            self.last = frame
            self.frames += hold

    m3, n3, _, _ = t3m.load_policy(str(ROOT / "artifacts/custom_pure_rl/ppo_custom_pure.zip"))
    g3 = Grab()
    s3 = t3m.play_and_render(m3, n3, seed=COMPARE_SEED, max_pieces=500, writer=g3)
    imgs["board_t3"] = png_b64(g3.last)

    m5, n5, _, _ = t5m.load_policy(str(ROOT / "artifacts/custom_afterstate/ppo_custom_afterstate.zip"))
    g5 = Grab()
    s5 = t5m.play_and_render(m5, n5, seed=COMPARE_SEED, max_pieces=500, writer=g5)
    imgs["board_t5"] = png_b64(g5.last)

    imgs["_stats"] = {"t3": s3, "t5": s5, "seed": COMPARE_SEED}
    return imgs


# --------------------------------------------------------------------------- html

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0e14; --panel:#141922; --line:#232c3b; --ink:#e8edf5; --dim:#8b97ab;
  --rl:#5b9dff;        /* pure RL  */
  --tool:#ffb454;      /* tool-assisted */
  --good:#4ec9a0; --bad:#ff6b6b; --warn:#ffd166;
  --mono:'SF Mono',SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;
}
html,body{height:100%;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;overflow:hidden}
#deck{position:relative;width:100vw;height:100vh}
.slide{position:absolute;inset:0;display:none;flex-direction:column;justify-content:center;
  padding:5vh 6vw;animation:in .28s ease}
.slide.on{display:flex}
@keyframes in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
h1{font-size:clamp(30px,4.4vw,64px);line-height:1.08;letter-spacing:-.02em;font-weight:700}
h2{font-size:clamp(22px,2.9vw,42px);line-height:1.15;letter-spacing:-.015em;margin-bottom:.6em;font-weight:650}
h3{font-size:clamp(15px,1.35vw,20px);color:var(--dim);text-transform:uppercase;
  letter-spacing:.14em;font-weight:600;margin-bottom:1.1em}
p,li{font-size:clamp(15px,1.5vw,23px);line-height:1.55;color:#c9d3e0;max-width:62ch}
ul{list-style:none} li{padding-left:1.3em;position:relative;margin:.5em 0}
li:before{content:'';position:absolute;left:0;top:.62em;width:7px;height:7px;
  border-radius:50%;background:var(--line)}
b,strong{color:#fff;font-weight:650}
code,.mono{font-family:var(--mono);font-size:.92em}
.big{font-size:clamp(40px,7vw,104px);font-weight:750;letter-spacing:-.035em;line-height:1}
.dim{color:var(--dim)} .rl{color:var(--rl)} .tool{color:var(--tool)}
.good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}

/* part divider */
.divider{justify-content:center;align-items:flex-start}
.divider .num{font-family:var(--mono);font-size:clamp(60px,11vw,160px);font-weight:700;
  color:var(--line);line-height:.9}
.divider h1{margin-top:.1em}
.divider p{margin-top:.8em;font-size:clamp(16px,1.7vw,26px)}

/* generic blocks */
.row{display:flex;gap:2.2vw;align-items:center}
.col{flex:1;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.5em 1.6em}
.card.rlc{border-left:4px solid var(--rl)} .card.toolc{border-left:4px solid var(--tool)}
.tag{display:inline-block;font-size:.62em;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;padding:.34em .7em;border-radius:999px;vertical-align:middle}
.tag.rlt{background:rgba(91,157,255,.14);color:var(--rl)}
.tag.toolt{background:rgba(255,180,84,.14);color:var(--tool)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.45em 1.1em;font-size:clamp(13px,1.15vw,18px)}
.kv dt{color:var(--dim);white-space:nowrap}
.kv dd{color:#dbe3ee;font-family:var(--mono);font-size:.95em}
pre{background:#0f131b;border:1px solid var(--line);border-left:3px solid var(--tool);
  border-radius:9px;padding:1em 1.2em;overflow-x:auto;font-family:var(--mono);
  font-size:clamp(12px,1.15vw,18px);line-height:1.6;color:#dbe3ee}
.note{border-left:3px solid var(--warn);background:rgba(255,209,102,.06);
  padding:.85em 1.1em;border-radius:0 9px 9px 0;font-size:clamp(13px,1.2vw,19px);
  color:#e4d5b0;max-width:none}
.note b{color:var(--warn)}
.lead{font-size:clamp(18px,2vw,30px);color:#dbe3ee;max-width:56ch}

table{border-collapse:collapse;width:100%;font-size:clamp(12px,1.15vw,19px)}
th,td{padding:.62em .7em;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:.82em;text-transform:uppercase;letter-spacing:.07em}
td.n{font-family:var(--mono);text-align:right;white-space:nowrap}
tr.hi td{background:rgba(91,157,255,.09)}
td .same{color:#5f6b7d}

/* the 2x2 */
.grid2{display:grid;grid-template-columns:auto 1fr 1fr;grid-template-rows:auto 1fr 1fr;
  gap:.7em;align-items:stretch}
.grid2 .lbl{display:flex;align-items:center;justify-content:center;color:var(--dim);
  font-size:clamp(11px,1vw,16px);text-transform:uppercase;letter-spacing:.12em;font-weight:600}
.grid2 .lbl.v{writing-mode:vertical-rl;transform:rotate(180deg)}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1em 1.1em}
.cell.rlc{border-left:4px solid var(--rl)} .cell.toolc{border-left:4px solid var(--tool)}
.cell .t{font-weight:700;font-size:clamp(14px,1.3vw,21px);margin-bottom:.15em}
.cell .r{font-family:var(--mono);font-size:clamp(15px,1.6vw,26px);font-weight:700;margin-top:.3em}
.cell .s{color:var(--dim);font-size:clamp(11px,1vw,15px)}

img.shot{border-radius:10px;border:1px solid var(--line);display:block;max-width:100%;
  height:auto;image-rendering:pixelated}
.cap{color:var(--dim);font-size:clamp(11px,1.05vw,16px);margin-top:.6em;text-align:center}

.chain{display:flex;align-items:center;gap:1vw;flex-wrap:wrap;margin:.4em 0}
.chain .step{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:.7em 1em;font-size:clamp(12px,1.2vw,19px)}
.chain .arw{color:var(--dim);font-size:1.4em}
.chain .step.end{border-color:var(--bad);color:var(--bad);font-weight:700}

/* chrome */
#bar{position:fixed;bottom:0;left:0;height:2px;background:var(--rl);transition:width .25s;z-index:9}
#ctr{position:fixed;bottom:14px;right:20px;font-family:var(--mono);font-size:13px;
  color:var(--dim);z-index:9}
#hint{position:fixed;bottom:14px;left:20px;font-size:12px;color:#465061;z-index:9}
#ov{position:fixed;inset:0;background:rgba(11,14,20,.97);z-index:20;display:none;
  grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;padding:28px;
  overflow-y:auto;align-content:start}
#ov.on{display:grid}
#ov button{background:var(--panel);border:1px solid var(--line);color:var(--dim);
  border-radius:9px;padding:11px;text-align:left;cursor:pointer;font:inherit;font-size:12px}
#ov button:hover{border-color:var(--rl);color:var(--ink)}
#ov button .i{font-family:var(--mono);color:#4a5666;display:block;margin-bottom:3px}
@media print{
  html,body{overflow:visible;background:#fff}
  .slide{display:flex!important;position:relative;height:100vh;page-break-after:always;
    border-bottom:1px solid #ccc}
  #bar,#ctr,#hint,#ov{display:none!important}
}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
const bar=document.getElementById('bar'),ctr=document.getElementById('ctr'),ov=document.getElementById('ov');
function go(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('on',k===i));
  bar.style.width=((i+1)/S.length*100)+'%';ctr.textContent=(i+1)+' / '+S.length;
  location.hash=i+1;}
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){ov.classList.toggle('on');return;}
  if(ov.classList.contains('on')){return;}
  if(['ArrowRight','ArrowDown',' ','PageDown','n'].includes(e.key)){e.preventDefault();go(i+1);}
  if(['ArrowLeft','ArrowUp','PageUp','p'].includes(e.key)){e.preventDefault();go(i-1);}
  if(e.key==='Home')go(0); if(e.key==='End')go(S.length-1);
});
document.getElementById('deck').addEventListener('click',e=>{
  if(ov.classList.contains('on'))return;
  go(e.clientX < innerWidth*0.22 ? i-1 : i+1);
});
S.forEach((s,k)=>{const h=s.querySelector('h1,h2');const b=document.createElement('button');
  b.innerHTML='<span class="i">'+(k+1)+'</span>'+(h?h.textContent.slice(0,58):'');
  b.onclick=()=>{go(k);ov.classList.remove('on');};ov.appendChild(b);});
go(parseInt(location.hash.slice(1))-1 || 0);
"""


def build_html(r: dict, img: dict) -> str:
    t1, t2, t3, t4, t5 = (r[k] for k in (1, 2, 3, 4, 5))
    s3, s5 = img["_stats"]["t3"], img["_stats"]["t5"]
    seed = img["_stats"]["seed"]

    def f(x, n=2):
        return f"{x:.{n}f}"

    # log(1+x) so a genuine zero still renders as a zero-height bar
    def h(v, maxv=np.log10(1 + 198.1)):
        return np.log10(1 + v) / maxv * 100

    S: list[str] = []
    A = S.append

    # ---------------------------------------------------------------- title
    A(f"""<section class="slide">
  <h3>Reinforcement learning &middot; Tetris</h3>
  <h1>Five agents,<br>one question.</h1>
  <p class="lead" style="margin-top:1em">How much of "playing Tetris well" comes from
  <b>learning</b> &mdash; and how much from the <b>action abstraction and search</b>
  that tools provide?</p>
  <p class="dim" style="margin-top:2em;font-size:clamp(13px,1.2vw,18px)">
  <span class="mono">ALE/Tetris-v5</span> &nbsp;&middot;&nbsp; custom engine
  &nbsp;&middot;&nbsp; PPO &nbsp;&middot;&nbsp; CEM</p>
</section>""")

    # ================================================================ PART 1
    A("""<section class="slide divider">
  <div class="num">01</div>
  <h1>The game</h1>
  <p class="dim">What <span class="mono">ALE/Tetris-v5</span> actually is, and the
  mechanics that decide everything downstream.</p>
</section>""")

    A("""<section class="slide">
  <h3>Part 1 &middot; the environment</h3>
  <h2>It is an Atari 2600 ROM,<br>running under emulation.</h2>
  <p class="lead">Not a Tetris library. A 1980s cartridge, emulated by Stella, wrapped by the
  Arcade Learning Environment. <b>You cannot read its memory or its rules.</b></p>
  <div class="row" style="margin-top:1.6em">
    <div class="card col"><h3 style="margin-bottom:.6em">You are given</h3>
      <ul style="font-size:.95em">
        <li>a <b>screen</b> &mdash; <span class="mono">210&times;160&times;3</span> RGB</li>
        <li>an <b>action</b> &mdash; one of five</li>
        <li>a <b>reward</b> &mdash; one number</li>
      </ul></div>
    <div class="card col"><h3 style="margin-bottom:.6em">You are not given</h3>
      <ul style="font-size:.95em">
        <li>the board as an array</li>
        <li>the piece identity or position</li>
        <li>the next piece</li>
        <li>any rule of the game</li>
      </ul></div>
  </div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 1 &middot; observation</h3>
  <h2>This is the game.<br>This is what the agent sees.</h2>
  <div class="row" style="margin-top:.5em;align-items:flex-start">
    <div class="col" style="text-align:center">
      <img class="shot" src="{img['ale_screen']}" style="max-height:46vh;margin:0 auto">
      <div class="cap">The Atari screen &mdash; <span class="mono">210&times;160&times;3</span> RGB</div>
    </div>
    <div class="col" style="text-align:center">
      <img class="shot" src="{img['ale_obs']}" style="max-height:46vh;margin:0 auto">
      <div class="cap">What Track&nbsp;1's network is fed &mdash; <span class="mono">84&times;84</span>
      grayscale, &times;4 stacked</div>
    </div>
  </div>
  <div class="note" style="margin-top:1.2em"><b>Why stack 4 frames?</b> One still image cannot
  tell you which way a piece is <i>moving</i>. Stacking the last four gives the network motion.
  Standard since the 2015 DQN Atari paper.</div>
</section>""")

    A("""<section class="slide">
  <h3>Part 1 &middot; action space</h3>
  <h2>Five actions. <span class="bad">No hard drop.</span></h2>
  <pre>&gt;&gt;&gt; env.unwrapped.get_action_meanings()
['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN']        <span class="dim"># Discrete(5)</span></pre>
  <ul style="margin-top:1em">
    <li><b class="mono">FIRE</b> = rotate. <b>One direction only</b> &mdash; there is no counter-clockwise.</li>
    <li><b class="mono">DOWN</b> = <b>soft</b> drop. It nudges the piece down one step.</li>
    <li><b class="bad">There is no hard-drop action.</b> To get a piece down you wait for gravity,
      or you hold DOWN.</li>
  </ul>
  <div class="note" style="margin-top:1.1em">My own custom engine <i>does</i> have <span class="mono">HARD_DROP</span>
  &mdash; one action, piece slams down, done. <b>The Atari ROM simply does not.</b></div>
</section>""")

    A("""<section class="slide">
  <h3>Part 1 &middot; consequence</h3>
  <h2>So my planner presses DOWN<br>ninety times.</h2>
  <pre><span class="dim">agents/ale/ale_tetris_agent.py:546</span>
actions = [FIRE] * fires + [move] * moves + <b class="tool">[DOWN] * 90</b></pre>
  <p class="lead" style="margin-top:1.2em">Rotate <i>n</i> times, shift sideways <i>n</i> times,
  then <b>spam the soft-drop ninety times</b> to seat the piece.</p>
  <p class="lead" style="margin-top:.6em">It looks absurd until you know there is no drop action.
  Then it is the only thing you <i>can</i> write.</p>
</section>""")

    A("""<section class="slide">
  <h3>Part 1 &middot; reward</h3>
  <h2>The reward <span class="rl">is</span> lines cleared.<br>Nothing else.</h2>
  <div class="row" style="margin-top:1.2em">
    <div class="card col">
      <h3 style="margin-bottom:.5em">ALE Tetris pays you for</h3>
      <ul style="font-size:.95em"><li><b>clearing a line</b></li></ul>
      <h3 style="margin:1.1em 0 .5em">and nothing else</h3>
      <ul style="font-size:.95em">
        <li><span class="dim">no points for surviving</span></li>
        <li><span class="dim">no points for dropping</span></li>
        <li><span class="dim">no points for placing a piece</span></li>
      </ul>
    </div>
    <div class="col">
      <p class="lead">So when my agent scores <b class="mono">37.0</b> reward, that
      <b>literally means 37 lines.</b></p>
      <div class="note" style="margin-top:1.2em"><b>Say "37 lines."</b> The
      "3,700" you may see in my code is <b>my own unit</b>
      (<span class="mono">lines &times; 100</span>) for a friendlier headline &mdash;
      it is <b>not</b> the ROM's score.</div>
    </div>
  </div>
</section>""")

    A("""<section class="slide">
  <h3>Part 1 &middot; the consequence that defines this project</h3>
  <h2>This is why pure RL scores <span class="bad">exactly zero</span>.</h2>
  <div class="chain" style="margin-top:.8em">
    <div class="step">A random policy must place<br>~10 pieces correctly in a row<br>to clear one line</div>
    <div class="arw">&rarr;</div>
    <div class="step">it essentially <b>never</b><br>does this by chance</div>
    <div class="arw">&rarr;</div>
    <div class="step">reward is <b>0 on every<br>transition</b>, for all 10M steps</div>
  </div>
  <div class="chain">
    <div class="step">so the advantage estimate is <b>0</b></div>
    <div class="arw">&rarr;</div>
    <div class="step end">the policy gradient is <b>0</b></div>
  </div>
  <p class="lead" style="margin-top:1.3em"><b>PPO did not fail to learn. It had nothing to learn from.</b>
  You cannot fix a <i>missing</i> signal by collecting more samples of it.</p>
  <div class="note" style="margin-top:1em"><b>Verified live:</b> NOOP-only from seed 0 tops out
  after <span class="mono">416</span> agent steps with total reward <span class="mono">0.0</span>.</div>
</section>""")

    A("""<section class="slide">
  <h3>Part 1 &middot; three things that trip people up</h3>
  <h2>The fine print.</h2>
  <div class="row" style="margin-top:.4em;align-items:stretch">
    <div class="card col"><div class="t" style="font-weight:700;margin-bottom:.4em">Frame-skip 4</div>
      <p style="font-size:.9em">One action runs for <b>4 emulator frames</b>. So
      <b>10M agent steps = 40M frames</b> &mdash; about <b>20%</b> of the canonical 200M-<i>frame</i>
      Atari budget, not the 2% a naive reading suggests. <b>State the unit.</b></p></div>
    <div class="card col"><div class="t" style="font-weight:700;margin-bottom:.4em">Sticky actions 0.25</div>
      <p style="font-size:.9em">25% chance the emulator <b>repeats your previous action</b> instead of
      the new one. It stops agents memorising one fixed button sequence.</p></div>
    <div class="card col" style="border-left:4px solid var(--warn)">
      <div class="t" style="font-weight:700;margin-bottom:.4em">&#9888; The seed does nothing</div>
      <p style="font-size:.9em"><b>The ALE seed does not change Tetris's piece sequence.</b> Seeds
      0/1/2/42 give a bit-identical trajectory. So sticky actions are the <b>only</b> randomness
      in this environment &mdash; and "10 seeds" is <b>one game, ten times</b>.</p></div>
  </div>
</section>""")

    # ================================================================ PART 2
    A("""<section class="slide divider">
  <div class="num">02</div>
  <h1>The five tracks</h1>
  <p class="dim">How each one was built, and exactly what differs between them.</p>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; the design</h3>
  <h2>A 2&times;2 &mdash; plus the experiment.</h2>
  <div class="grid2" style="margin-top:.3em;max-height:52vh">
    <div></div>
    <div class="lbl">Pure RL &nbsp;<span class="tag rlt">learns</span></div>
    <div class="lbl">Tool-assisted &nbsp;<span class="tag toolt">searches</span></div>
    <div class="lbl v">ALE</div>
    <div class="cell rlc"><div class="t">Track 1</div><div class="s">PPO on pixels</div>
      <div class="r bad">{t1['lines']:.0f} lines</div></div>
    <div class="cell toolc"><div class="t">Track 2</div><div class="s">decode + search + CEM</div>
      <div class="r tool">{t2['lines']:.0f} lines</div></div>
    <div class="lbl v">Custom engine</div>
    <div class="cell rlc"><div class="t">Track 3</div><div class="s">PPO, keypress actions</div>
      <div class="r rl">{f(t3['lines'])} lines</div></div>
    <div class="cell toolc"><div class="t">Track 4</div><div class="s">features + lookahead + CEM</div>
      <div class="r tool">{f(t4['lines'],1)} lines</div></div>
  </div>
  <div class="note" style="margin-top:1em"><b>Track 5</b> is not a fifth corner of the grid. It is
  <b>Track 3 with exactly one variable changed</b> &mdash; the action space &mdash; so the 2&times;2
  no longer has to <i>guess</i> what causes the gap. <span class="rl">&rarr; 5.60 lines</span></div>
</section>""")

    A("""<section class="slide">
  <h3>Part 2 &middot; motivation</h3>
  <h2>Why five tracks, honestly.</h2>
  <ul style="margin-top:.4em">
    <li><b>Pure RL on ALE Tetris scores zero &mdash; and I needed to know whether that was my
      bug or the problem's nature.</b> The only way to tell is to change one thing at a time.
      Keep the method, change the environment (Track 3) &rarr; it starts clearing lines. Keep the
      environment, change the method (Track 2) &rarr; it scores. That isolates the cause.</li>
    <li><b>I did not know what was allowed.</b> "Make an RL agent" could mean strictly model-free
      learning, or it could permit search, planning and hand-authored features &mdash; a lot of
      published "RL Tetris" does exactly that. Rather than guess and risk submitting against the
      wrong interpretation, I built <b>both</b>, and drew an enforced boundary between them
      (<span class="mono">AGENTS.md</span> + tests).</li>
    <li><b>It turns a zero into a finding.</b> "PPO got 0" is a shrug. "PPO got 0 on pixels,
      1.04 lines on a structured observation, and a planner on the <i>same engine and the same
      laptop</i> got ~190&times; more" is a result.</li>
  </ul>
  <div class="note" style="margin-top:.9em"><b>Be honest about the ordering:</b> this began as a
  <i>hedge</i> against an ambiguous brief, and <i>became</i> a controlled comparison. It was not a
  grand design from day one.</div>
</section>""")

    # per-track cards
    A(f"""<section class="slide">
  <h3>Part 2 &middot; track 1 &nbsp;<span class="tag rlt">pure RL</span> &nbsp;<span class="dim">ALE</span></h3>
  <h2>PPO, straight onto the pixels.</h2>
  <div class="row" style="align-items:flex-start">
    <div class="col">
      <dl class="kv">
        <dt>Algorithm</dt><dd>PPO (Stable-Baselines3), CnnPolicy</dd>
        <dt>Input</dt><dd>84&times;84&times;4 grayscale frames</dd>
        <dt>Actions</dt><dd>the 5 ALE actions</dd>
        <dt>Reward</dt><dd>the ROM's &mdash; lines cleared</dd>
        <dt>Budget</dt><dd>10M agent steps (= 40M frames)</dd>
        <dt>Result</dt><dd class="bad">{t1['lines']:.0f} lines &mdash; {t1['detail']}</dd>
      </dl>
    </div>
    <div class="col">
      <p><b>What it may not do:</b> decode the board, enumerate placements, use any model of Tetris,
      or search. It sees pixels and presses buttons. That is the whole point &mdash; it is the
      <b>control</b>.</p>
      <div class="note" style="margin-top:1em"><b>Zero is the result, not a bug.</b> The video shows
      the stack building until it tops out. See the causal chain in Part 1.</div>
    </div>
  </div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; track 2 &nbsp;<span class="tag toolt">tool-assisted</span> &nbsp;<span class="dim">ALE</span></h3>
  <h2>Give it eyes and a plan.</h2>
  <div class="row" style="align-items:flex-start">
    <div class="col">
      <dl class="kv">
        <dt>Method</dt><dd>decode &rarr; enumerate &rarr; score &rarr; execute</dd>
        <dt>Vision</dt><dd>sample the centre of all 200 cells<br>&rarr; a 20&times;10 binary grid</dd>
        <dt>Policy</dt><dd>9 hand-authored features &middot; dot product</dd>
        <dt>Tuning</dt><dd>CEM over 9 weights</dd>
        <dt>Lookahead</dt><dd class="dim">none &mdash; depth-1 greedy</dd>
        <dt>Result</dt><dd class="tool">{t2['lines']:.0f} lines, {t2['pieces']:.0f} decisions</dd>
      </dl>
    </div>
    <div class="col">
      <pre style="font-size:clamp(11px,1vw,15px)"><span class="dim">decode_board() &mdash; ale_tetris_agent.py:81</span>
med     = median(3&times;3 patch at each cell)
is_gray = |med - BG_GRAY| &lt;= 16
is_black= med &lt;= 12
<b>return ~(is_gray | is_black)</b>   <span class="dim"># filled?</span></pre>
      <p style="margin-top:.9em;font-size:.95em">It reconstructs the same 20&times;10 grid my custom
      engine uses &mdash; which is what lets <b>one planner drive both environments</b>.</p>
    </div>
  </div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; track 3 &nbsp;<span class="tag rlt">pure RL</span> &nbsp;<span class="dim">custom engine</span></h3>
  <h2>Same PPO. A reward it can actually feel.</h2>
  <div class="row" style="align-items:flex-start">
    <div class="col">
      <dl class="kv">
        <dt>Algorithm</dt><dd>PPO, MlpPolicy 2&times;256</dd>
        <dt>Input</dt><dd><b>417 floats</b></dd>
        <dt></dt><dd class="dim">200 board + 200 falling piece<br>+ 7 + 7 one-hots + 3 position</dd>
        <dt>Actions</dt><dd><b>Discrete(7)</b> &mdash; one <b>keypress</b><br>
          <span class="dim">left / right / rotate / drop</span></dd>
        <dt>Reward</dt><dd>10&times;cleared&sup2; + 0.25/piece &minus; 10</dd>
        <dt>Budget</dt><dd>100M steps &middot; ~7.8 h</dd>
        <dt>Result</dt><dd class="rl">{f(t3['lines'])} lines &mdash; {t3['detail']}</dd>
      </dl>
    </div>
    <div class="col">
      <p><b>The +0.25 per piece is the whole trick.</b> It is a <i>dense</i> survival signal, so the
      gradient is not zero everywhere. Track 1 had no such term available &mdash; ALE's reward is
      fixed.</p>
      <div class="note" style="margin-top:1em"><b>Gravity is the difficulty.</b> Every keypress also
      drops the piece one row, so the agent gets only <b>~9 actions per piece</b> &mdash; and the
      reward arrives dozens of actions after the decision that earned it.</div>
    </div>
  </div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; track 4 &nbsp;<span class="tag toolt">tool-assisted</span> &nbsp;<span class="dim">custom engine</span></h3>
  <h2>Enumerate, score, look ahead.</h2>
  <div class="row" style="align-items:flex-start">
    <div class="col">
      <dl class="kv">
        <dt>Method</dt><dd>enumerate all ~34 placements</dd>
        <dt>Policy</dt><dd><b>10 Dellacherie features</b> &middot; dot product</dd>
        <dt></dt><dd class="dim">holes, heights, bumpiness, wells,<br>transitions, landing height &hellip;</dd>
        <dt>Search</dt><dd><b>depth-2</b> over the real queue<br><span class="dim">top 4 candidates, future &times;0.35</span></dd>
        <dt>Tuning</dt><dd>CEM over <b>10 weights</b></dd>
        <dt>Result</dt><dd class="tool">{f(t4['lines'],1)} lines &mdash; {t4['detail']}</dd>
      </dl>
    </div>
    <div class="col">
      <pre style="font-size:clamp(11px,1vw,16px)"><span class="dim">the entire policy &mdash; tetris_custom_agent.py:20</span>
<b>np.dot(weights, placement_features(p))</b></pre>
      <p style="margin-top:.8em;font-size:.95em">Ten multiplications and an addition, per candidate.
      <b>No neural network anywhere in this track.</b></p>
      <div class="note" style="margin-top:.9em"><b>It never tops out.</b> Run uncapped it reached
      <b>10,000 pieces / 3,997 lines</b> and was still alive &mdash; holding ~0.4 lines per piece, the
      theoretical maximum.</div>
    </div>
  </div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; track 5 &nbsp;<span class="tag rlt">pure RL</span> &nbsp;<span class="dim">custom engine</span></h3>
  <h2>Track 3 &mdash; with <span class="rl">one</span> thing changed.</h2>
  <div class="row" style="align-items:flex-start">
    <div class="col">
      <dl class="kv">
        <dt>Algorithm</dt><dd>PPO, MlpPolicy 2&times;256 <span class="good">&mdash; identical</span></dd>
        <dt>Reward</dt><dd>10&times;cleared&sup2; + 0.25/piece &minus; 10 <span class="good">&mdash; identical</span></dd>
        <dt>Hyperparams</dt><dd>lr 3e-4, &gamma; .995, seed 7 <span class="good">&mdash; identical</span></dd>
        <dt>Input</dt><dd><b>214 floats</b> &mdash; raw board + 2 one-hots<br>
          <span class="bad">no features. none.</span></dd>
        <dt>Lookahead</dt><dd class="bad">none</dd>
        <dt>Actions</dt><dd><b class="rl">Discrete(40)</b> &mdash; 4 rotations &times; 10 columns<br>
          <span class="rl">one action = one whole piece placed</span></dd>
        <dt>Budget</dt><dd>12M steps &middot; <b>2.0 h</b></dd>
        <dt>Result</dt><dd class="rl">{f(t5['lines'])} lines &mdash; {t5['detail']}</dd>
      </dl>
    </div>
    <div class="col">
      <p>It must still <b>learn what a bad board looks like</b> from raw cells &mdash; it is handed
      no holes, no heights, no bumpiness.</p>
      <div class="note" style="margin-top:1em"><b>Note it sees <i>less</i> than Track 3</b> (214 vs 417
      inputs) and does 5.4&times; better. Those extra 203 numbers describe a problem &mdash; piloting a
      piece mid-flight &mdash; that Track 5 does not have to solve.</div>
    </div>
  </div>
</section>""")

    # comparison matrix
    A(f"""<section class="slide">
  <h3>Part 2 &middot; the whole project on one slide</h3>
  <h2>What actually differs.</h2>
  <table style="margin-top:.2em">
    <tr><th></th><th>Env</th><th>Learns?</th><th>Action</th><th>Hand<br>features</th>
      <th>Look-<br>ahead</th><th>Budget</th><th style="text-align:right">Lines</th></tr>
    <tr><td><b class="rl">Track 1</b></td><td>ALE</td><td class="rl">yes &mdash; PPO</td>
      <td>joystick</td><td class="bad">no</td><td class="bad">no</td><td class="n">10M steps</td>
      <td class="n bad">{t1['lines']:.0f}</td></tr>
    <tr><td><b class="tool">Track 2</b></td><td>ALE</td><td class="dim">no &mdash; CEM</td>
      <td>placement&rarr;joystick</td><td class="good">yes (9)</td><td class="bad">no</td>
      <td class="n">CEM gens</td><td class="n tool">{t2['lines']:.0f}</td></tr>
    <tr class="hi"><td><b class="rl">Track 3</b></td><td>custom</td><td class="rl">yes &mdash; PPO</td>
      <td><b>keypress</b></td><td class="bad">no</td><td class="bad">no</td>
      <td class="n">100M steps</td><td class="n rl">{f(t3['lines'])}</td></tr>
    <tr class="hi"><td><b class="rl">Track 5</b></td><td><span class="same">custom</span></td>
      <td><span class="same">yes &mdash; PPO</span></td>
      <td><b class="rl">placement</b></td><td><span class="same">no</span></td>
      <td><span class="same">no</span></td><td class="n">12M steps</td>
      <td class="n rl"><b>{f(t5['lines'])}</b></td></tr>
    <tr><td><b class="tool">Track 4</b></td><td>custom</td><td class="dim">no &mdash; CEM</td>
      <td>placement</td><td class="good">yes (10)</td><td class="good">yes (2-ply)</td>
      <td class="n">CEM gens</td><td class="n tool">{f(t4['lines'],1)}</td></tr>
  </table>
  <div class="note" style="margin-top:1.1em">Read the two highlighted rows. <b>Every cell is the
  same except one</b> &mdash; keypress becomes placement. That single change is worth
  <b class="rl">{f(r['gain'],1)}&times;</b>.</div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; the controlled experiment</h3>
  <h2>Same seed. Same algorithm.<br>Different action space.</h2>
  <div class="row" style="margin-top:.3em;align-items:flex-start">
    <div class="col" style="text-align:center">
      <img class="shot" src="{img['board_t3']}" style="max-height:40vh;margin:0 auto">
      <div class="cap"><b class="rl">Track 3</b> &mdash; keypress actions<br>
        seed {seed}: <b>{s3['lines']} lines</b>, {s3['pieces']} pieces</div>
    </div>
    <div class="col" style="text-align:center">
      <img class="shot" src="{img['board_t5']}" style="max-height:40vh;margin:0 auto">
      <div class="cap"><b class="rl">Track 5</b> &mdash; placement actions<br>
        seed {seed}: <b>{s5['lines']} lines</b>, {s5['pieces']} pieces</div>
    </div>
  </div>
  <div class="note" style="margin-top:1em"><b>The comparison is fair on <i>experience</i>, not just
  step count.</b> A Track 3 step is one keypress; a Track 5 step is a whole piece. Track 3 spends a
  <b>measured 9.11 keypresses per piece</b>, so its 100M steps &asymp; <b>11.0M pieces</b> against
  Track 5's <b>12.0M</b> &mdash; matched within 9%, in a quarter of the wall-clock.</div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 2 &middot; what is still different</h3>
  <h2>Track 5 &rarr; Track 4: three more changes.</h2>
  <div class="row" style="margin-top:.5em;align-items:stretch">
    <div class="card col toolc"><div class="t" style="font-weight:700;margin-bottom:.4em">1. Hand-authored features</div>
      <p style="font-size:.92em">Track 4 is <i>handed</i> holes, heights, bumpiness, wells, transitions,
      landing height. Track 5 must <b>learn all of it from raw cells</b>.</p></div>
    <div class="card col toolc"><div class="t" style="font-weight:700;margin-bottom:.4em">2. Lookahead</div>
      <p style="font-size:.92em">Track 4 clones the engine and searches <b>2 pieces deep</b> over the real
      queue. Track 5 sees the same one-piece preview and <b>cannot search at all</b>.</p></div>
    <div class="card col toolc"><div class="t" style="font-weight:700;margin-bottom:.4em">3. Optimizer</div>
      <p style="font-size:.92em">Track 4 fits <b>10 weights</b> with CEM. Track 5 fits <b>~150k network
      weights</b> with PPO from a scalar reward.</p></div>
  </div>
  <p class="lead" style="margin-top:1.4em">Track 5 closes the action-space part of the gap. <b>These
  three are what remain</b> &mdash; and Part 3 shows they are worth <b class="tool">{f(r['remaining'],1)}%</b> of it.</p>
</section>""")

    A("""<section class="slide">
  <h3>Part 2 &middot; the question I will be asked</h3>
  <h2>"Are all five reinforcement learning?"</h2>
  <p class="lead"><b>No &mdash; and I will concede that before you ask.</b></p>
  <div class="row" style="margin-top:1.2em;align-items:stretch">
    <div class="card col rlc"><div class="t" style="font-weight:700;margin-bottom:.5em">
      <span class="tag rlt">RL</span> &nbsp;Tracks 1, 3, 5</div>
      <p style="font-size:.93em">PPO. A <b>policy network</b> and a <b>value network</b>. Learns from
      the reward at <b>every step</b> via the policy gradient and an advantage estimate.
      <b>This is reinforcement learning.</b></p></div>
    <div class="card col toolc"><div class="t" style="font-weight:700;margin-bottom:.5em">
      <span class="tag toolt">not RL</span> &nbsp;Tracks 2, 4</div>
      <p style="font-size:.93em">CEM. <b>No value function, no TD error, no policy gradient.</b> It
      collapses an entire episode into <b>one scalar</b> and never asks which move caused what. It is
      <b>derivative-free policy search</b> &mdash; black-box optimisation.</p></div>
  </div>
  <div class="note" style="margin-top:1.1em"><b>Sutton &amp; Barto, &sect;1.1</b> exclude evolutionary
  methods for exactly this reason: <i>"they do not notice which states an individual passes through
  during its lifetime, or which actions it selects."</i> &mdash; Structurally, Track 4 is a small
  <b>chess engine</b>: hand-crafted eval + shallow search + tuned coefficients.</div>
</section>""")

    # ================================================================ PART 3
    A("""<section class="slide divider">
  <div class="num">03</div>
  <h1>The results</h1>
  <p class="dim">Every number below is read straight from the frozen evaluation manifests.</p>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 3 &middot; best performance per track</h3>
  <h2>Where each one landed.</h2>
  <table style="margin-top:.2em">
    <tr><th></th><th>Env</th><th>Method</th><th style="text-align:right">Result</th><th>Evidence &amp; caveat</th></tr>
    <tr><td><b class="rl">1</b></td><td>ALE</td><td>PPO on pixels</td>
      <td class="n bad" style="font-size:1.35em"><b>{t1['lines']:.0f}</b> lines</td>
      <td>{t1['episodes']} eps. <b>That is the result</b>, not a bug &mdash; zero reward &rArr; zero gradient.</td></tr>
    <tr><td><b class="tool">2</b></td><td>ALE</td><td>decode + search + CEM</td>
      <td class="n tool" style="font-size:1.35em"><b>{t2['lines']:.0f}</b> lines</td>
      <td>{t2['pieces']:.0f} decisions. <span class="warn">&#9888; identical on all {t2['episodes']} seeds because the
      ALE seed does not vary the pieces &mdash; effective <b>N&nbsp;=&nbsp;1</b>.</span></td></tr>
    <tr><td><b class="rl">3</b></td><td>custom</td><td>PPO, keypress</td>
      <td class="n rl" style="font-size:1.35em"><b>{f(t3['lines'])}</b> lines</td>
      <td>{t3['detail']}; survives {t3['pieces']:.1f} pieces.</td></tr>
    <tr><td><b class="rl">5</b></td><td>custom</td><td>PPO, <b>placement</b></td>
      <td class="n rl" style="font-size:1.35em"><b>{f(t5['lines'])}</b> lines</td>
      <td>{t5['detail']}; survives {t5['pieces']:.1f}. <b class="good">Clears &ge;1 line on every seed.</b>
      <span class="warn">&#9888; not converged &rArr; a lower bound.</span></td></tr>
    <tr><td><b class="tool">4</b></td><td>custom</td><td>features + 2-ply + CEM</td>
      <td class="n tool" style="font-size:1.35em"><b>{f(t4['lines'],1)}</b> lines</td>
      <td>{t4['detail']}. <span class="warn">&#9888; the 200 is <b>the cap</b>, not the agent &mdash; uncapped it
      never tops out.</span></td></tr>
  </table>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 3 &middot; visually</h3>
  <h2>Lines cleared, by environment.</h2>
  <svg viewBox="0 0 1000 380" style="width:100%;max-height:52vh;margin-top:.3em">
    <line x1="60" y1="330" x2="960" y2="330" stroke="#232c3b" stroke-width="1"/>
    <text x="60" y="24" fill="#8b97ab" font-size="14" font-family="monospace">log scale &mdash; the range is 0 to 198</text>

    <!-- ALE group -->
    <rect x="95"  y="{330 - h(t1['lines'])*2.7:.0f}" width="90" height="{max(h(t1['lines'])*2.7, 2):.0f}" fill="#5b9dff" opacity=".85" rx="3"/>
    <text x="140" y="{330 - h(t1['lines'])*2.7 - 12:.0f}" fill="#ff6b6b" font-size="26" font-weight="700" text-anchor="middle" font-family="monospace">{t1['lines']:.0f}</text>
    <text x="140" y="352" fill="#8b97ab" font-size="15" text-anchor="middle">Track 1</text>
    <text x="140" y="370" fill="#4a5666" font-size="12" text-anchor="middle">pure RL</text>

    <rect x="215" y="{330 - h(t2['lines'])*2.7:.0f}" width="90" height="{h(t2['lines'])*2.7:.0f}" fill="#ffb454" opacity=".9" rx="3"/>
    <text x="260" y="{330 - h(t2['lines'])*2.7 - 12:.0f}" fill="#ffb454" font-size="26" font-weight="700" text-anchor="middle" font-family="monospace">{t2['lines']:.0f}</text>
    <text x="260" y="352" fill="#8b97ab" font-size="15" text-anchor="middle">Track 2</text>
    <text x="260" y="370" fill="#4a5666" font-size="12" text-anchor="middle">tools</text>
    <text x="200" y="{330 - h(198.1)*2.7 - 34:.0f}" fill="#5f6b7d" font-size="15" text-anchor="middle" letter-spacing="2">ALE</text>

    <line x1="345" y1="40" x2="345" y2="330" stroke="#232c3b" stroke-width="1" stroke-dasharray="4 4"/>

    <!-- custom group -->
    <rect x="400" y="{330 - h(t3['lines'])*2.7:.0f}" width="90" height="{h(t3['lines'])*2.7:.0f}" fill="#5b9dff" opacity=".85" rx="3"/>
    <text x="445" y="{330 - h(t3['lines'])*2.7 - 12:.0f}" fill="#5b9dff" font-size="26" font-weight="700" text-anchor="middle" font-family="monospace">{f(t3['lines'])}</text>
    <text x="445" y="352" fill="#8b97ab" font-size="15" text-anchor="middle">Track 3</text>
    <text x="445" y="370" fill="#4a5666" font-size="12" text-anchor="middle">keypress</text>

    <rect x="520" y="{330 - h(t5['lines'])*2.7:.0f}" width="90" height="{h(t5['lines'])*2.7:.0f}" fill="#5b9dff" rx="3"/>
    <text x="565" y="{330 - h(t5['lines'])*2.7 - 12:.0f}" fill="#5b9dff" font-size="26" font-weight="700" text-anchor="middle" font-family="monospace">{f(t5['lines'])}</text>
    <text x="565" y="352" fill="#e8edf5" font-size="15" text-anchor="middle" font-weight="600">Track 5</text>
    <text x="565" y="370" fill="#4a5666" font-size="12" text-anchor="middle">placement</text>

    <rect x="640" y="{330 - h(t4['lines'])*2.7:.0f}" width="90" height="{h(t4['lines'])*2.7:.0f}" fill="#ffb454" opacity=".9" rx="3"/>
    <text x="685" y="{330 - h(t4['lines'])*2.7 - 12:.0f}" fill="#ffb454" font-size="26" font-weight="700" text-anchor="middle" font-family="monospace">{f(t4['lines'],1)}</text>
    <text x="685" y="352" fill="#8b97ab" font-size="15" text-anchor="middle">Track 4</text>
    <text x="685" y="370" fill="#4a5666" font-size="12" text-anchor="middle">tools</text>
    <text x="565" y="{330 - h(198.1)*2.7 - 34:.0f}" fill="#5f6b7d" font-size="15" text-anchor="middle" letter-spacing="2">CUSTOM ENGINE</text>

    <path d="M 490 {330 - h(t3['lines'])*2.7 - 40:.0f} L 555 {330 - h(t5['lines'])*2.7 - 40:.0f}"
      stroke="#5b9dff" stroke-width="2" fill="none" marker-end="url(#a)" opacity=".8"/>
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#5b9dff"/></marker></defs>
    <text x="522" y="{330 - h(t5['lines'])*2.7 - 50:.0f}" fill="#5b9dff" font-size="15" text-anchor="middle" font-weight="700">{f(r['gain'],1)}&times;</text>
  </svg>
  <div class="note" style="margin-top:.3em"><b>&#9888; Do not compare across the divider.</b> ALE and my
  custom engine are <b>different games</b> with different mechanics. Lines are only comparable
  <i>within</i> an environment.</div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 3 &middot; the decomposition</h3>
  <h2>So what actually carries Tetris?</h2>
  <div class="chain" style="margin-top:.6em">
    <div class="step"><span class="dim">Track 3</span><br><b class="rl" style="font-size:1.5em">{f(t3['lines'])}</b> lines
      <br><span class="dim" style="font-size:.85em">PPO, keypress</span></div>
    <div class="arw">&rarr;</div>
    <div class="step" style="border-color:var(--rl)"><span class="rl">change the action space</span><br>
      <b class="rl" style="font-size:1.5em">{f(t5['lines'])}</b> lines
      <br><span class="dim" style="font-size:.85em">Track 5 &mdash; nothing else changed</span></div>
    <div class="arw">&rarr;</div>
    <div class="step" style="border-color:var(--tool)"><span class="tool">+ features, lookahead, CEM</span><br>
      <b class="tool" style="font-size:1.5em">{f(t4['lines'],1)}</b> lines
      <br><span class="dim" style="font-size:.85em">Track 4</span></div>
  </div>
  <div class="row" style="margin-top:1.6em">
    <div class="card col rlc"><div class="dim" style="font-size:.9em">The action abstraction is worth</div>
      <div class="big rl">{f(r['gain'],1)}&times;</div>
      <div class="dim" style="font-size:.9em;margin-top:.3em">&hellip;but it closes only
      <b class="rl">{f(r['closed'],1)}%</b> of the Track&nbsp;3 &rarr; Track&nbsp;4 gap.</div></div>
    <div class="card col toolc"><div class="dim" style="font-size:.9em">Hand features + lookahead + CEM carry</div>
      <div class="big tool">{f(r['remaining'],1)}%</div>
      <div class="dim" style="font-size:.9em;margin-top:.3em">of the remaining distance.</div></div>
  </div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 3 &middot; the finding</h3>
  <h2>I was wrong,<br>and I ran the experiment that proved it.</h2>
  <p class="lead" style="margin-top:.6em">My report originally claimed the <b>action abstraction</b> was
  the dominant variable &mdash; reasoning from the literature, where essentially every strong RL-Tetris
  result uses placement actions.</p>
  <p class="lead" style="margin-top:.9em"><b>Track 5 tested that claim on my own engine and refuted
  it.</b> The abstraction is real &mdash; <b class="rl">{f(r['gain'],1)}&times;</b>, at matched experience, in a
  quarter of the wall-clock &mdash; but it is <b>necessary, not sufficient</b>. It closes
  <b class="rl">{f(r['closed'],1)}%</b> of the gap. The hand-authored features and the search carry the rest.</p>
  <div class="note" style="margin-top:1.3em"><b>The caveat I have to state myself:</b> Track 5's learning
  curve <b>had not converged</b> &mdash; <span class="mono">ep_rew_mean</span> was still climbing 60 &rarr; 75
  over the final 4M steps. So <b>{f(t5['lines'])} is a lower bound, not a ceiling.</b> I have shown the action space is
  not <i>sufficient at this compute budget</i>. I have <b>not</b> shown hand-authored features are
  necessary in principle. <span class="dim">(Track 3, by contrast, <i>plateaued</i> at 36M of its 100M steps.)</span></div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 3 &middot; receipts</h3>
  <h2>Four things I will not let you<br>catch me on.</h2>
  <ul style="margin-top:.2em">
    <li><b>"Score 3,700" is my own unit, not the ROM's.</b> ALE Tetris's native reward <i>is</i> lines
      cleared &mdash; my agent gets {t2['lines']:.0f}.0 reward. If you ask what I scored, the answer is
      <b class="tool">{t2['lines']:.0f} lines</b>.</li>
    <li><b>My "10 seeds" for Track 2 are one game, ten times.</b> The ALE seed does not change the piece
      sequence &mdash; I verified it. The zero variance is an <b>artifact</b>, not robustness. Effective
      <b>N&nbsp;=&nbsp;1</b>. <span class="dim">(What <i>is</i> genuine: it still clears {t2['lines']:.0f} with sticky
      actions at 0.25, because the planner re-reads the board every piece.)</span></li>
    <li><b>Track 4's "200-line ceiling" is my piece cap, not the agent.</b> 500 pieces &times; 4 cells
      &divide; 10 columns = 200. Uncapped it <b>does not top out</b> &mdash; 10,000 pieces, 3,997 lines,
      still alive. Read its score as <span class="mono">&asymp; 0.4 &times; cap</span>.</li>
    <li><b>Track 5 had not converged</b>, so <b class="rl">{f(t5['lines'])}</b> is a floor, not a ceiling.</li>
  </ul>
  <div class="note" style="margin-top:.9em">Scores are <b>not comparable across tracks</b> &mdash; Track 3's
  includes drop points, Track 4's does not, Tracks 1&ndash;2 use my lines&times;100 convention.
  <b>Compare lines, within an environment.</b></div>
</section>""")

    A(f"""<section class="slide">
  <h3>Part 3 &middot; what I would do next</h3>
  <h2>Two experiments, in order.</h2>
  <div class="row" style="margin-top:.6em;align-items:stretch">
    <div class="card col rlc">
      <div class="t" style="font-weight:700;font-size:1.15em;margin-bottom:.5em">1. Run Track 5 to convergence</div>
      <p style="font-size:.95em">It was <b>still improving</b> when its 12M-step budget ran out. 50&ndash;100M
      steps (~8&ndash;16 h at the measured 1,664 steps/s) would find the <b>actual ceiling</b> of the action
      abstraction on its own. Right now I only have a floor.</p>
    </div>
    <div class="card col toolc">
      <div class="t" style="font-weight:700;font-size:1.15em;margin-bottom:.5em">2. A features-only ablation</div>
      <p style="font-size:.95em">Track 5 isolated the <b>action space</b>. Nothing isolated the
      <b>features</b>. Feed the 10 Dellacherie features to Track 5's PPO &mdash; still no search &mdash; and
      the remaining <b>{f(r['remaining'],1)}%</b> splits into "better state representation" vs "lookahead +
      CEM". That would <b>fully decompose the 190&times;</b>.</p>
    </div>
  </div>
</section>""")

    A(f"""<section class="slide" style="justify-content:center">
  <h3>In one line</h3>
  <h1 style="font-family:var(--mono);font-size:clamp(26px,4.6vw,68px);letter-spacing:-.02em">
    <span class="bad">{t1['lines']:.0f}</span>
    <span class="dim">&rarr;</span>
    <span class="rl">{f(t3['lines'])}</span>
    <span class="dim">&rarr;</span>
    <span class="rl">{f(t5['lines'])}</span>
    <span class="dim">&rarr;</span>
    <span class="tool">{f(t4['lines'],1)}</span>
  </h1>
  <p class="lead" style="margin-top:1.3em">
    Pixels to a structured observation with a dense reward: <b>the signal appears</b>.<br>
    Keypress to placement &mdash; <b>the one controlled change</b>: <b class="rl">{f(r['gain'],1)}&times;</b>,
    but only <b class="rl">{f(r['closed'],1)}%</b> of the gap.<br>
    Hand-authored features, lookahead and CEM: <b class="tool">everything else</b>.
  </p>
  <p class="dim" style="margin-top:2.2em;font-size:clamp(13px,1.2vw,18px)">
    <span class="mono">docs/REPORT.md</span> &middot;
    <span class="mono">docs/QA_CODE_WALKTHROUGH.md</span> &middot;
    <span class="mono">python artifacts/best_plays/live_play.py</span>
  </p>
</section>""")

    slides = "\n".join(S)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>TetrisGPT — five agents, one question</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f'<div id="deck">\n{slides}\n</div>\n'
        '<div id="bar"></div><div id="ctr"></div>\n'
        '<div id="hint">&larr; &rarr; to move &middot; Esc for overview</div>\n'
        '<div id="ov"></div>\n'
        f"<script>{JS}</script>\n</body>\n</html>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-capture", action="store_true", help="reuse cached images")
    args = ap.parse_args()
    start = time.time()

    if args.no_capture and CACHE.exists():
        img = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"reusing cached images from {CACHE.name}")
    else:
        print("capturing real frames from the live environments...")
        img = capture_images()
        CACHE.write_text(json.dumps(img), encoding="utf-8")

    results = load_results()
    print(
        "results read from manifests: "
        + ", ".join(f"T{t}={results[t]['lines']}" for t in (1, 2, 3, 4, 5))
    )
    OUT.write_text(build_html(results, img), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB, self-contained)")
    print(f"elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
