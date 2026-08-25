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

from weft_loop import History, PreconditionFailed, Round, household, pixi_run, run  # noqa: E402

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
        "an environment name pixi does not have is a FAIL, not an empty result",
        lambda: pixi_run("no-such-environment-here", ["-c", "pass"]),
    )
    expect_raise(
        "a working directory with no pixi.toml is a FAIL",
        lambda: pixi_run("editscore", ["-c", "pass"], cwd=tmp),
    )
    expect_raise(
        "a score that cannot be measured on the control stops the run",
        lambda: run(lambda i: str(good), lambda p: (_ for _ in ()).throw(ValueError("no scorer")), control),
        exc=ValueError,
    )

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
