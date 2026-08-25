"""The loop every notebook in `notebooks/` runs: propose, score, repair, repeat.

The four loops differ only in what proposes and what scores. Everything that is
the same -- shelling into a pixi environment, timing a round, holding the
baseline beside the number, refusing to call an unmet precondition a pass --
lives here so a fix lands once.

Three rules from CLAUDE.md are load-bearing in this file and are named where
they apply:

* A number without a baseline is not a measurement, so `run` will not return a
  history without one.
* A silent skip reads exactly like a pass, so a missing environment raises.
* A check that passes on known-broken input certifies the defect, so
  `test_weft_loop.py` beside this file drives each guard with input that must
  fail.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence

CORPUS = Path(r"C:\weftspun-keypoint\6-datasource\anny-render-corpus")

# CLAUDE.md's anchors, in millimetres. A measurement is reported with one of
# these beside it, because "4.3 mm" does not tell a reader whether an error
# matters and "about three stacked pennies" does.
ANCHORS = (
    ("credit card", 0.76),
    ("penny", 1.52),
    ("pencil", 7.0),
    ("AAA battery", 10.5),
    ("AA battery", 14.5),
    ("nickel", 21.2),
    ("golf ball", 42.7),
    ("adult wrist", 57.0),
    ("soda can", 66.0),
)


class PreconditionFailed(RuntimeError):
    """An unmet precondition. Raised, never returned, never logged and skipped."""


@dataclass
class Round:
    index: int
    artifact: str
    score: float
    delta: float
    seconds: float
    provenance: dict = field(default_factory=dict)


@dataclass
class History:
    baseline: float
    control: str
    rounds: list[Round] = field(default_factory=list)

    @property
    def best(self) -> Round | None:
        return max(self.rounds, key=lambda r: r.score, default=None)

    def table(self) -> str:
        head = f"baseline {self.baseline:.3f} on {Path(self.control).name}"
        rows = [
            f"  {r.index:>2}  {r.score:6.3f}  {r.delta:+6.3f}  {r.seconds:6.1f}s  {Path(r.artifact).name}"
            for r in self.rounds
        ]
        return "\n".join([head, "  ##   score   delta    time  artifact", *rows])

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def household(mm: float) -> str:
    """Name the closest household object, with how many of them it takes."""
    name, size = min(ANCHORS, key=lambda a: abs(mm - a[1]))
    n = mm / size
    if n < 1.5:
        return f"about one {name}"
    return f"about {n:.1f} {name}s stacked"


def pixi_run(env: str, args: Sequence[str], cwd: Path | str = CORPUS, timeout: int = 3600):
    """Run `pixi run -e <env> python <args>` and raise on anything but success.

    The environments are the ones `6-datasource/anny-render-corpus/pixi.toml`
    already declares. Nothing here builds a second one: OmniGen2 pins
    torch==2.6.0+cu124 and EditScore pins cu128, and one embedded interpreter
    cannot hold both.
    """
    if shutil.which("pixi") is None:
        raise PreconditionFailed("pixi is not on PATH; the model environments are unreachable")
    cwd = Path(cwd)
    if not (cwd / "pixi.toml").is_file():
        raise PreconditionFailed(f"no pixi.toml under {cwd}")
    proc = subprocess.run(
        ["pixi", "run", "-e", env, "python", *map(str, args)],
        cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise PreconditionFailed(
            f"pixi -e {env} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
        )
    return proc


def run(
    propose: Callable[[int], str],
    score: Callable[[str], float],
    control: str,
    rounds: int = 3,
    target: float | None = None,
) -> History:
    """Propose, score, repeat. `control` is what the baseline is measured on.

    The baseline is taken first and on every call. A run that could not measure
    its floor returns no rounds, because a score with nothing to compare it to
    is not a measurement of anything.
    """
    baseline = float(score(control))
    history = History(baseline=baseline, control=str(control))

    for i in range(1, rounds + 1):
        started = time.monotonic()
        artifact = propose(i)
        if artifact is None or not Path(artifact).exists():
            raise PreconditionFailed(f"round {i} proposed no artifact at {artifact!r}")
        value = float(score(artifact))
        history.rounds.append(
            Round(
                index=i,
                artifact=str(artifact),
                score=value,
                delta=value - baseline,
                seconds=time.monotonic() - started,
            )
        )
        if target is not None and value >= target:
            break
    return history


def plot(history: History):
    """Score per round, with the baseline drawn as the floor it is compared to."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = [r.index for r in history.rounds]
    ys = [r.score for r in history.rounds]
    ax.plot(xs, ys, marker="o", label="score")
    ax.axhline(history.baseline, linestyle="--", label=f"baseline {history.baseline:.3f}")
    ax.set_xlabel("round")
    ax.set_ylabel("EditScore overall")
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    return fig


def require_whole_sequence(indices, n_views, band=None):
    """Every view of the sequence, or a declared band. Never a hand-picked set or a prefix.

    `check_view_selection.check` in the corpus repository is the implementation and this is
    the gate: that function returns problems and `gen_posed_from_reference.py` prints them,
    which is a warning nobody has to obey. Here they raise.

    The measurement behind the rule is in that file's docstring. At n=8 the sequence's
    pitches are -90, -30, 0, 9.6, 19.5, 30, 41.8 and 56.4 degrees, so they are not uniform:
    dropping index 0 alone removes the only view below -30, and any subset silently changes
    the pitch distribution the corpus is trained on. A prefix is the polite version of the
    same defect, which is why it is refused here rather than reported.
    """
    if str(CORPUS) not in sys.path:
        sys.path.insert(0, str(CORPUS))
    import check_view_selection

    problems = check_view_selection.check(list(indices), int(n_views), band=band)
    if problems:
        raise PreconditionFailed(
            "the view selection is not the whole sequence: " + "; ".join(problems)
        )
    return True
