"""Negative controls for weft_loop.

Every guard is driven with input that must fail. A check that passes on
known-broken input certifies the defect instead of catching it, so each case
here asserts a rejection, and the two positive cases at the end exist so that a
file rejecting everything cannot pass either.

    python priv/python/test_weft_loop.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from weft_loop import (AZIMUTHS, DISTANCES, ELEVATIONS, History, PreconditionFailed, Round,
                       camera_parameters, camera_prompt, camera_ranges, corpus_module,
                       angle_between, describe_camera, grid_cameras, household, in_band,
                       nearest_view, run,
                       sample_camera,
                       view_name)  # noqa: E402

FAILS: list[str] = []


def expect_raise(label, fn, exc=PreconditionFailed):
    try:
        fn()
    except exc as error:
        print(f"  ok  {label}: rejected ({str(error).splitlines()[0][:70]})")
        return
    except Exception as error:  # noqa: BLE001
        FAILS.append(label)
        print(f"  BAD {label}: raised {type(error).__name__}, wanted {exc.__name__}")
        return
    FAILS.append(label)
    print(f"  BAD {label}: accepted")


def expect_ok(label, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        FAILS.append(label)
        print(f"  BAD {label}: raised {type(error).__name__}: {error}")
        return
    print(f"  ok  {label}: accepted")


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    control = tmp / "control.png"
    control.write_bytes(b"\x89PNG\r\n\x1a\n")
    good = tmp / "round.png"
    good.write_bytes(b"\x89PNG\r\n\x1a\n")

    print("negative controls: each known-bad input must be rejected")

    expect_raise(
        "a round that writes no artifact is a failed round, not a zero",
        lambda: run(lambda i: str(tmp / "never-written.png"), lambda p: 1.0, control),
    )
    expect_raise(
        "a propose that returns None fails rather than skipping the round",
        lambda: run(lambda i: None, lambda p: 1.0, control),
    )
    expect_raise(
        "a corpus script that does not exist is a FAIL, not an empty result",
        lambda: corpus_module("no_such_corpus_script"),
    )
    expect_raise(
        "a score that cannot be measured on the control stops the run",
        lambda: run(lambda i: str(good), lambda p: (_ for _ in ()).throw(ValueError("no scorer")), control),
        exc=ValueError,
    )

    print("the camera vocabulary, against fal's own table")
    table = {
        (0, 0): "front view eye-level shot medium shot",
        (45, 30): "front-right quarter view elevated shot medium shot",
        (90, 60): "right side view high-angle shot medium shot",
        (180, -30): "back view low-angle shot medium shot",
        (270, 0): "left side view eye-level shot medium shot",
        (315, 0): "front-left quarter view eye-level shot medium shot",
    }
    for (azimuth, elevation), wanted in table.items():
        got = view_name(azimuth, elevation)
        if got == wanted:
            print(f"  ok  {azimuth:>3} deg, {elevation:>3} deg: {got}")
        else:
            FAILS.append(f"view_name({azimuth}, {elevation})")
            print(f"  BAD {azimuth} deg, {elevation} deg: {got!r}, wanted {wanted!r}")

    # THE POPULATION IS FIXED AT 96, SO IT IS ENUMERATED RATHER THAN SAMPLED. A sampled
    # check sees defects larger than about 3/n, and there is no reason to accept that floor
    # for eight azimuths, four elevations and three distances.
    cells = [(a, e, d) for a, _ in AZIMUTHS for e, _ in ELEVATIONS for d, _ in DISTANCES]
    bad = [c for c in cells if camera_parameters(camera_prompt(*c)) != c]
    if bad:
        FAILS.append("prompt round trip")
        print(f"  BAD {len(bad)} of {len(cells)} prompts did not survive the round trip: {bad[:3]}")
    else:
        print(f"  ok  all {len(cells)} prompts round trip through camera_parameters")

    keys = ("azimuth_deg", "elevation_deg", "distance_factor")
    outside = []
    for cell in cells:
        bands = camera_ranges(camera_prompt(*cell))
        if not all(in_band(value, bands[key], wrap=360 if key == "azimuth_deg" else None)
                   for value, key in zip(cell, keys)):
            outside.append(cell)
    if outside:
        FAILS.append("band contains its own value")
        print(f"  BAD {len(outside)} table values sit outside their own band: {outside[:3]}")
    else:
        print(f"  ok  every table value sits inside the band its phrase names")

    # A DRAW INSIDE A BAND MUST STILL NAME THAT BAND, or the band and the phrase disagree.
    drifted = []
    for cell in cells:
        prompt = camera_prompt(*cell)
        for seed in range(5):
            phrase, note = describe_camera(*sample_camera(prompt, seed))
            if phrase != prompt:
                drifted.append((prompt, seed, phrase or note))
    if drifted:
        FAILS.append("sampled camera names its own band")
        print(f"  BAD {len(drifted)} sampled cameras named another band: {drifted[:2]}")
    else:
        print(f"  ok  {len(cells) * 5} sampled cameras each name the band they came from")

    if sample_camera(camera_prompt(0, 0, 1.0), 7) == sample_camera(camera_prompt(0, 0, 1.0), 7):
        print("  ok  a seeded draw repeats, so a corpus built from it can be regenerated")
    else:
        FAILS.append("seeded draw repeats")
        print("  BAD the same seed gave two different cameras")

    print("negative controls for the camera vocabulary")
    expect_raise("an azimuth the table does not name",
                 lambda: camera_prompt(9.6, 0), exc=ValueError)
    expect_raise("an elevation the table does not name",
                 lambda: camera_prompt(0, 9.6), exc=ValueError)
    expect_raise("a distance the table does not name",
                 lambda: camera_prompt(0, 0, 1.23), exc=ValueError)
    expect_raise("a prompt naming two elevations",
                 lambda: camera_parameters("<sks> front view eye-level shot low-angle shot medium shot"), exc=ValueError)
    expect_raise("a prompt naming no azimuth",
                 lambda: camera_parameters("<sks> eye-level shot medium shot"), exc=ValueError)

    phrase, note = describe_camera(0, -90)
    if phrase is None and "elevation band" in note:
        print("  ok  a camera under the body has no phrase, and the note says which bands exist")
    else:
        FAILS.append("uncovered elevation")
        print(f"  BAD -90 deg elevation was named {phrase!r}")
    phrase, note = describe_camera(0, 9.6)
    if phrase == "<sks> front view eye-level shot medium shot":
        print("  ok  a camera between two steps takes the band that contains it")
    else:
        FAILS.append("band containment")
        print(f"  BAD 9.6 deg elevation was named {phrase!r} with note {note!r}")

    print("the 96-pose grid, which is not the sequence")
    grid = list(grid_cameras())
    if len(grid) == 96:
        print(f"  ok  the grid enumerates {len(grid)} poses")
    else:
        FAILS.append("grid size")
        print(f"  BAD the grid enumerated {len(grid)} poses, wanted 96")
    cells = {(a, e, d) for a, e, d, _ in grid}
    if len(cells) == len(grid):
        print("  ok  every pose is distinct")
    else:
        FAILS.append("grid duplicates")
        print(f"  BAD {len(grid) - len(cells)} poses repeat")
    wanted = {(a, e, d) for a, _ in AZIMUTHS for e, _ in ELEVATIONS for d, _ in DISTANCES}
    if cells == wanted:
        print("  ok  the grid is exactly the cross product, with nothing added or dropped")
    else:
        FAILS.append("grid cross product")
        print(f"  BAD the grid differs from the cross product by {cells ^ wanted}")
    mismatched = [p for a, e, d, p in grid if camera_parameters(p) != (a, e, d)]
    if mismatched:
        FAILS.append("grid prompt")
        print(f"  BAD {len(mismatched)} grid prompts do not read back: {mismatched[:2]}")
    else:
        print("  ok  every grid prompt reads back to the pose that made it")
    if [p[0] for p in grid[:9]] == [0, 45, 90, 135, 180, 225, 270, 315, 0]:
        print("  ok  azimuth runs fastest, so a partial run is a partial sweep")
    else:
        FAILS.append("grid order")
        print(f"  BAD the grid order starts {[p[0] for p in grid[:9]]}")

    print("choosing a comparison view")
    if abs(angle_between(0, 0, 0, 0)) < 1e-9:
        print("  ok  a direction is zero degrees from itself")
    else:
        FAILS.append("angle to self")
        print("  BAD a direction was not zero degrees from itself")
    if abs(angle_between(0, 0, 180, 0) - 180) < 1e-9:
        print("  ok  front to back is 180 degrees")
    else:
        FAILS.append("front to back")
        print(f"  BAD front to back measured {angle_between(0, 0, 180, 0)}")
    # THE PROXY IS THE FAILURE THIS CONTROL EXISTS FOR. Adding the azimuth and elevation
    # differences gives 90 for this pair, and the great-circle answer is 60.
    diagonal = angle_between(0, 0, 45, 45)
    if abs(diagonal - 60) < 0.5:
        print(f"  ok  45 round and 45 up is {diagonal:.1f} degrees, not 90")
    else:
        FAILS.append("diagonal angle")
        print(f"  BAD 45 and 45 measured {diagonal:.2f}, wanted about 60")

    index, degrees = nearest_view([(180, 0), (90, 0), (10, 5), (0, -90)], 0, 0)
    if index == 2 and degrees < 12:
        print(f"  ok  the nearest view to the front is index 2 at {degrees:.1f} degrees")
    else:
        FAILS.append("nearest view")
        print(f"  BAD nearest view chose index {index} at {degrees:.1f} degrees")
    expect_raise("choosing a view from no views", lambda: nearest_view([], 0, 0))

    print("positive controls: a file that rejected everything would pass the above")

    def three_rounds():
        history = run(lambda i: str(good), lambda p: 0.5 if p == control else 0.5 + i_of(p), control, rounds=3)
        assert len(history.rounds) == 3, history.rounds
        assert history.baseline == 0.5
        assert history.rounds[-1].delta > 0, "a rising score must show a positive delta"
        assert history.best.index == 3

    counter = {"n": 0}

    def i_of(_path):
        counter["n"] += 1
        return 0.1 * counter["n"]

    expect_ok("three rising rounds keep their baseline and deltas", three_rounds)
    expect_ok(
        "target stops the loop early",
        lambda: _assert_len(run(lambda i: str(good), lambda p: 9.0, control, rounds=5, target=1.0), 1),
    )
    expect_ok(
        "household names an anchor and a count",
        lambda: _assert_in("penny", household(3.0)),
    )
    expect_ok("a history renders a table with its baseline", lambda: _assert_in("baseline", _table()))

    print(f"\n{len(FAILS)} failed")
    return 1 if FAILS else 0


def _assert_len(history, n):
    assert len(history.rounds) == n, f"wanted {n} rounds, got {len(history.rounds)}"


def _assert_in(needle, haystack):
    assert needle in haystack, f"{needle!r} not in {haystack!r}"


def _table():
    h = History(baseline=0.4, control="c.png")
    h.rounds.append(Round(index=1, artifact="a.png", score=0.6, delta=0.2, seconds=1.0))
    return h.table()


if __name__ == "__main__":
    raise SystemExit(main())
