"""The loop every notebook in `notebooks/` runs: propose, score, repair, repeat.

The four loops differ only in what proposes and what scores. Everything that is
the same -- importing a corpus script into this interpreter, timing a round,
holding the baseline beside the number, refusing to call an unmet precondition
a pass -- lives here so a fix lands once.

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


# THE CAMERA VOCABULARY IS SOMEBODY ELSE'S, ON PURPOSE.
# These are the phrases fal's Qwen-Image-Edit-2511-Multiple-Angles-LoRA is conditioned on,
# copied rather than invented, so a view named here reads the same way to a generator, to a
# tagger and to a person. "pitch -90, yaw 0" tells a reader nothing about what the picture
# shows. "front view, low-angle shot" does.
#
#   https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
#
# WHAT TRAVELS IS THE PHRASE TABLE, NOT THE MODEL. Qwen-Image-Edit 2509 and 2511 are
# blocklisted here -- 20.4B runs on this desk only quantised, and quantised it corrupts --
# and that LoRA's weights are gated on top of it. Nothing in this project loads either. A
# vocabulary is text, it carries no licence into a corpus, and naming views the way the
# ecosystem already names them is the whole benefit.
#
# The vocabulary is coarse and this file neither widens it nor rounds into it. Azimuth has
# eight exact sectors and elevation four exact steps. A camera that sits between two of them
# has no phrase, and `describe_view` returns the refusal instead of the nearest neighbour.
# Five of the eight views in the render sequence land between steps, which is a fact about
# `sphere_hammersley_sequence` rather than a fault in either.
AZIMUTHS = (
    (0, "front view"),
    (45, "front-right quarter view"),
    (90, "right side view"),
    (135, "back-right quarter view"),
    (180, "back view"),
    (225, "back-left quarter view"),
    (270, "left side view"),
    (315, "front-left quarter view"),
)
ELEVATIONS = (
    (-30, "low-angle shot"),
    (0, "eye-level shot"),
    (30, "elevated shot"),
    (60, "high-angle shot"),
)
DISTANCES = (
    (0.6, "close-up"),
    (1.0, "medium shot"),
    (1.8, "wide shot"),
)


def _exact(table, value, unit, tolerance):
    """The phrase for a value the table names, or a refusal listing what it names.

    THE TABLE IS AN EXACT CONVERSION AND SNAPPING TO IT IS A LIE. A first version of this
    function took the nearest row, so a camera 9.6 degrees above the horizon came back as an
    eye-level shot and a camera 90 degrees below it came back as a low-angle shot. The name
    then described a picture nobody rendered. There is no phrase for those angles, and
    saying so is the honest answer.

    `tolerance` exists for float arithmetic, not for rounding a camera into a neighbouring
    sector. It defaults to a thousandth of a degree.
    """
    for known, phrase in table:
        if abs(known - value) <= tolerance:
            return phrase
    known = ", ".join(str(row[0]) for row in table)
    raise ValueError(f"{value} {unit} has no phrase; the table names {known}")


def view_name(azimuth_deg: float, elevation_deg: float, distance_factor: float = 1.0,
              tolerance: float = 1e-3) -> str:
    """The camera as the LoRA's prompt says it, in its own order, or a refusal.

    Azimuth wraps, because 360 and 0 are one direction. Nothing else is adjusted.
    """
    azimuth = _exact(AZIMUTHS, azimuth_deg % 360, "deg azimuth", tolerance)
    elevation = _exact(ELEVATIONS, elevation_deg, "deg elevation", tolerance)
    distance = _exact(DISTANCES, distance_factor, "x distance", tolerance)
    return f"{azimuth} {elevation} {distance}"


def describe_view(azimuth_deg: float, elevation_deg: float, distance_factor: float = 1.0):
    """(phrase, note). One of the two is None, and never both.

    A table of eight views needs a cell for the views the vocabulary cannot say, and an
    exception is not a cell. The note carries the exact degrees so the row still reports the
    camera it rendered.
    """
    try:
        return view_name(azimuth_deg, elevation_deg, distance_factor), None
    except ValueError as refusal:
        return None, str(refusal)


def _bands(table, wrap=None, geometric=False):
    """Each phrase's half-open interval, with the boundary midway to its neighbours.

    A PHRASE IS A BAND AND A NUMBER IS A POINT, WHICH IS WHY THE INVERSE IS NOT A FUNCTION.
    "eye-level shot" does not mean exactly 0 degrees, it means the sector around 0 that the
    next phrases do not claim. So the exact conversion above is the forward direction, and
    everything that reads a phrase back reads it as the band it stands for.

    Azimuth wraps, so its outer bands close on each other. Elevation and distance do not, so
    the first and last bands extend by half a step and no further: outside that the
    vocabulary says nothing, and `describe_view` refuses rather than reaching.

    Distance boundaries are geometric, because 0.6, 1.0 and 1.8 are multipliers. The
    arithmetic midpoint of 0.6 and 1.0 is 0.8 and the geometric one is 0.775, and the second
    is the one that sits midway in the quantity being scaled.
    """
    rows = sorted(table)
    mid = ((lambda a, b: (a * b) ** 0.5) if geometric else (lambda a, b: (a + b) / 2))
    out = {}
    for i, (value, phrase) in enumerate(rows):
        if i == 0:
            low = (mid(rows[-1][0] - wrap, value) if wrap
                   else value - (rows[1][0] - value) / 2)
        else:
            low = mid(rows[i - 1][0], value)
        if i == len(rows) - 1:
            high = (mid(value, rows[0][0] + wrap) if wrap
                    else value + (value - rows[i - 1][0]) / 2)
        else:
            high = mid(value, rows[i + 1][0])
        out[phrase] = ((low % wrap, high % wrap) if wrap else (low, high))
    return out


def in_band(value: float, band, wrap=None) -> bool:
    """Whether a value falls in a half-open band, including one that wraps through zero.

    THE FRONT VIEW'S BAND WRAPS AND THE FIRST VERSION OF THIS DID NOT KNOW IT. Its band runs
    from 337.5 to 22.5 degrees, so a camera at 343.5 is a front view and a plain low <= v <
    high says it is in no band at all. The enumeration in `test_weft_loop.py` found 36 such
    cameras, all of them front views drawn below zero.
    """
    low, high = band
    if wrap and low > high:
        return value >= low or value < high
    return low <= value < high


AZIMUTH_BANDS = _bands(AZIMUTHS, wrap=360)
ELEVATION_BANDS = _bands(ELEVATIONS)
DISTANCE_BANDS = _bands(DISTANCES, geometric=True)


def camera_ranges(prompt: str) -> dict:
    """A prompt back to the three intervals it stands for.

        >>> camera_ranges("<sks> front view eye-level shot medium shot")["elevation_deg"]
        (-15.0, 15.0)

    The front view's azimuth band wraps, so it reads (337.5, 22.5) with the low above the
    high. `in_band` knows that and a bare comparison does not.

    This is the inverse worth having. `camera_parameters` returns the table's own value,
    which is the centre, and a caller that wants to know what the phrase permits needs the
    edges.
    """
    azimuth, elevation, distance = camera_parameters(prompt)
    return {
        "azimuth_deg": AZIMUTH_BANDS[_exact(AZIMUTHS, azimuth, "deg azimuth", 1e-3)],
        "elevation_deg": ELEVATION_BANDS[_exact(ELEVATIONS, elevation, "deg elevation", 1e-3)],
        "distance_factor": DISTANCE_BANDS[_exact(DISTANCES, distance, "x distance", 1e-3)],
    }


def sample_camera(prompt: str, seed: int) -> tuple[float, float, float]:
    """One camera drawn uniformly inside the prompt's bands, reproducibly.

    The seed is required rather than defaulted. A corpus built from unseeded draws cannot be
    regenerated, and this is the function that would build one.
    """
    import random

    rng = random.Random(seed)
    ranges = camera_ranges(prompt)
    drawn = []
    for key in ("azimuth_deg", "elevation_deg", "distance_factor"):
        low, high = ranges[key]
        if low > high:  # the front view's band wraps through zero
            drawn.append(rng.uniform(low, high + 360) % 360)
        else:
            drawn.append(rng.uniform(low, high))
    return tuple(drawn)


def describe_camera(azimuth_deg: float, elevation_deg: float, distance_factor: float = 1.0):
    """Any camera to (phrase, note), by which band holds it. One of the two is None.

    This is the lossy direction and it says so. `camera_prompt` converts the table's own
    values and refuses anything else; this converts a real camera by finding the phrase whose
    band contains it, and refuses when no band does. The -90 degree view that opens the
    render sequence is the second case: there is no phrase for a camera underneath a body.
    """
    azimuth = azimuth_deg % 360
    found = []
    for value, key, bands in ((azimuth, "azimuth", AZIMUTH_BANDS),
                              (elevation_deg, "elevation", ELEVATION_BANDS),
                              (distance_factor, "distance", DISTANCE_BANDS)):
        phrase = next((name for name, band in bands.items()
                       if in_band(value, band, wrap=360 if key == "azimuth" else None)), None)
        if phrase is None:
            edges = ", ".join(f"{name} [{low:g}, {high:g})" for name, (low, high) in bands.items())
            return None, f"{value:g} is in no {key} band; the bands are {edges}"
        found.append(phrase)
    return f"{TRIGGER} {' '.join(found)}", None


def grid_cameras():
    """The 96 poses, enumerated: 8 azimuths by 4 elevations by 3 distances.

    TWO GENERATORS, TWO JOBS, AND NEITHER IS THE OTHER'S APPROXIMATION.
    `sphere_hammersley_sequence` covers a sphere without any view being chosen, which is what
    a measurement needs. This enumerates a grid whose cells are exactly the phrases the
    camera vocabulary names, which is what a labelled pair needs. Asking the sequence to
    reproduce the grid costs 1536 renders to land every cell within 5 degrees and never lands
    on one; the grid is 96 lines and every label is exact by construction.

    So they run independently. Nothing here samples the sequence and nothing in the sequence
    knows about this.

    Yields (azimuth_deg, elevation_deg, distance_factor, prompt), azimuth fastest, so a
    partial run is a partial sweep of one elevation rather than a partial sweep of one
    azimuth at every elevation.
    """
    for distance, _ in DISTANCES:
        for elevation, _ in ELEVATIONS:
            for azimuth, _ in AZIMUTHS:
                yield azimuth, elevation, distance, camera_prompt(azimuth, elevation, distance)


def angle_between(azimuth_a: float, elevation_a: float,
                  azimuth_b: float, elevation_b: float) -> float:
    """Degrees between two camera directions on the sphere.

    Not the difference of the two azimuths and not the difference of the two elevations. A
    camera 45 degrees round and 45 degrees up is not 90 degrees from the front, and either
    of those proxies would say it is.
    """
    import math

    def unit(azimuth, elevation):
        a, e = math.radians(azimuth), math.radians(elevation)
        return (math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))

    dot = sum(x * y for x, y in zip(unit(azimuth_a, elevation_a), unit(azimuth_b, elevation_b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def nearest_view(directions, azimuth_deg: float, elevation_deg: float):
    """(index, degrees) of the direction closest to a target, or a refusal if there are none.

    THE COMPARISON VIEW IS CHOSEN, NOT INHERITED FROM THE LOOP INDEX. Loop 1 scored view 0
    of its sequence for three rounds, and view 0 is a camera underneath the body. Which view
    a sequence starts on is an accident of the sequence.
    """
    if not directions:
        raise PreconditionFailed("no directions to choose from, so there is no nearest view")
    scored = [(angle_between(a, e, azimuth_deg, elevation_deg), i)
              for i, (a, e) in enumerate(directions)]
    degrees, index = min(scored)
    return index, degrees


TRIGGER = "<sks>"


def camera_prompt(azimuth_deg: float, elevation_deg: float, distance_factor: float = 1.0,
                  tolerance: float = 1e-3) -> str:
    """Camera parameters in, the LoRA's prompt out.

        >>> camera_prompt(0, -30, 0.6)
        '<sks> front view low-angle shot close-up'

    The trigger token is part of the prompt rather than something the caller remembers to
    add. A prompt missing it is not a shorter prompt, it is a different one.
    """
    return f"{TRIGGER} {view_name(azimuth_deg, elevation_deg, distance_factor, tolerance)}"


def camera_parameters(prompt: str) -> tuple[float, float, float]:
    """The prompt back to numbers, which is the direction that catches a typo.

        >>> camera_parameters("<sks> front view low-angle shot close-up")
        (0, -30, 0.6)

    Longest phrase first, because "front view" is a prefix of nothing but "back view" sits
    inside "back-right quarter view", and a shortest-first scan would read that as a back
    view with a stray remainder. Every phrase must be found exactly once, so a prompt naming
    two elevations is a refusal rather than whichever one was matched first.
    """
    text = prompt.strip()
    if text.startswith(TRIGGER):
        text = text[len(TRIGGER):].strip()
    found = []
    for table, unit in ((AZIMUTHS, "azimuth"), (ELEVATIONS, "elevation"), (DISTANCES, "distance")):
        hits = [(value, phrase) for value, phrase in sorted(table, key=lambda r: -len(r[1]))
                if phrase in text]
        # A phrase inside a longer one is not a second hit: "back view" is a substring of
        # nothing here, but "medium shot" and "high-angle shot" both end in "shot", so the
        # match is on the whole phrase and the containment check runs longest first.
        hits = [hit for hit in hits
                if not any(hit[1] in other[1] and hit[1] != other[1] for other in hits)]
        if len(hits) != 1:
            names = [phrase for _, phrase in hits]
            raise ValueError(f"{unit} matched {len(hits)} phrases in {prompt!r}: {names}")
        found.append(hits[0][0])
    return tuple(found)


def corpus_module(name: str):
    """Import a script from the corpus repository into THIS interpreter.

    A `pixi_run` helper stood here and shelled out to a second environment manager. It is
    gone, and `check_pixi_free.py` fails the build if it comes back. The reason is that a second
    environment manager describes a second environment that the notebook does not: the setup
    cell would list three packages while the loop depended on five environments in another
    repository, and the notebook would then run on one desk and nowhere else.

    So the corpus repository's scripts are imported as modules and their functions are
    called directly. The packages they need are declared in the notebook's own
    `pyproject.toml` cell, where pythonx and uv can rebuild them from the file.
    """
    if not (CORPUS / f"{name}.py").is_file():
        raise PreconditionFailed(f"{CORPUS / (name + '.py')} does not exist")
    if str(CORPUS) not in sys.path:
        sys.path.insert(0, str(CORPUS))
    import importlib

    return importlib.import_module(name)


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
