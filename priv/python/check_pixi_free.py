"""No code in this project may call pixi. The Python it runs is pythonx's.

WHY A GATE AND NOT AN AGREEMENT. This project is an Elixir application that embeds a
Python interpreter through `pythonx`. Its setup cell declares the packages, `uv` resolves
them, and the pins travel in the notebook file. That is one environment, described in one
place, rebuildable by anybody who opens the document.

A `pixi run` beside it is a second environment that the notebook does not describe. The
notebook then works on the desk where those environments happen to be installed and
nowhere else, and the failure is not visible in the file: the setup cell still lists three
packages while the loop quietly depends on five environments in another repository. That
is the same objection the `uv` blocklist entry makes, one level up.

So the rule is: an Elixir project reaches Python through pythonx, and through nothing
else. `6-datasource/anny-render-corpus` keeps its pixi environments and its own scripts
use them. What is forbidden is this project calling into them.

ONE ENVIRONMENT PER APP, BECAUSE PYTHONX CANNOT HOLD TWO. An embedded interpreter is
initialised once, with one dependency set. Two notebooks in one app that ask for different
packages are asking for two interpreters, and the second request is the one that loses.
The environments this workspace already separates say why: OmniGen2 pins torch 2.6.0+cu124
and EditScore pins cu128, and no interpreter holds both.

So the replacement for an environment manager is not one app with many environments. It is
one app per environment. A second dependency set means a second Elixir application, with
its own mix project, its own setup cell, and its own runtime.

WHAT IS CHECKED.

1. NO CALLS OUT. Every `.py`, `.ex`, `.exs` and `.livemd` under the project, minus this
   file, for a call to pixi: `pixi_run(`, a command line, a `"pixi"` argument vector, or a
   `which("pixi")` probe.

2. ONE DEPENDENCY SET. Every `pyproject.toml` cell in every notebook of this app must
   declare the same packages. A notebook that needs a different set names the app it should
   move to, rather than being served by a runtime that cannot satisfy it.

WHAT IS NOT CHECKED. Prose. A paragraph that names pixi, including this one, is not a
call, and a gate that could not tell them apart would make the rule unwritable.

A missing project directory is a FAIL, not a skip.

    python priv/python/check_pixi_free.py [--self-test]
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SUFFIXES = (".py", ".ex", ".exs", ".livemd")
SKIP_DIRS = {"_build", "deps", ".git", "__pycache__", ".pixi", "node_modules"}

# Each pattern is a CALL, never a mention. `pixi_run(` is the harness function, the two
# command forms are what a shell-out looks like in Python and in Elixir, and the `which`
# probe is how code asks whether pixi is installed before using it.
CALLS = (
    (re.compile(r"\bpixi_run\s*\("), "calls pixi_run"),
    (re.compile(r"""["']pixi["']\s*,"""), "passes pixi as an argument vector element"),
    (re.compile(r"\bpixi\s+run\b"), "shells out to pixi run"),
    (re.compile(r"""which\s*\(\s*["']pixi["']"""), "probes for pixi on PATH"),
)


def sources(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if set(path.parts) & SKIP_DIRS:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        yield path


def offences(path: Path, text: str) -> list[str]:
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        for pattern, what in CALLS:
            if pattern.search(line):
                out.append(f"{path.name}:{number}: {what}: {line.strip()[:90]}")
    return out


DEPS_FENCE = re.compile(r"^```pyproject\.toml\s*$")
DEP_LINE = re.compile(r"""^\s*["']([A-Za-z0-9._-]+)""")


def declared_packages(text: str) -> list[str] | None:
    """The package names in a notebook's `pyproject.toml` cell, or None if it has none.

    Names only, not pins: a version difference is a version difference, and what cannot be
    shared is a package set. Comments inside the cell are skipped, because the cells here
    carry their reasoning beside the pins.
    """
    lines = text.splitlines()
    fence = next((i for i, line in enumerate(lines) if DEPS_FENCE.match(line)), None)
    if fence is None:
        return None
    body = []
    for line in lines[fence + 1:]:
        if line.strip() == "```":
            break
        body.append(line)
    inside = False
    names = []
    for line in body:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("dependencies"):
            inside = True
            continue
        if inside:
            if stripped.startswith("]"):
                break
            match = DEP_LINE.match(line)
            if match:
                names.append(match.group(1))
    return sorted(set(names))


def check_one_environment(root: Path) -> list[str]:
    notebooks = sorted((root / "notebooks").glob("*.livemd"))
    if not notebooks:
        return []
    sets = {}
    for path in notebooks:
        names = declared_packages(path.read_text(encoding="utf-8", errors="replace"))
        if names is None:
            return [f"{path.name}: declares no pyproject.toml cell, so its environment is undescribed"]
        sets[path.name] = names
    distinct = {tuple(v) for v in sets.values()}
    print(f"{len(notebooks)} notebook(s), {len(distinct)} distinct dependency set(s)")
    if len(distinct) == 1:
        return []
    problems = []
    first = sets[notebooks[0].name]
    for name, names in sets.items():
        if names == first:
            continue
        extra = sorted(set(names) - set(first))
        missing = sorted(set(first) - set(names))
        problems.append(
            f"{name}: declares a different environment than {notebooks[0].name} "
            f"(adds {extra or 'nothing'}, drops {missing or 'nothing'}). "
            "pythonx holds one interpreter, so this notebook belongs in its own Elixir app."
        )
    return problems


def check(root: Path) -> list[str]:
    if not root.is_dir():
        return [f"{root} is not a directory, so nothing was checked, which is a FAIL"]
    problems = []
    checked = 0
    for path in sources(root):
        checked += 1
        problems += offences(path, path.read_text(encoding="utf-8", errors="replace"))
    if checked == 0:
        return [f"no source files found under {root}, which is never correct here"]
    print(f"{checked} source file(s) read under {root.name}")
    return problems + check_one_environment(root)


def self_test() -> int:
    """Each pattern must reject its own known-bad line, and prose must survive.

    A gate that passes on broken input certifies the defect, so the rejections are driven
    here rather than asserted in a comment.
    """
    fails = []
    bad = {
        "a Python harness call": 'result = pixi_run("anny", [str(script)])',
        "a Python argument vector": 'subprocess.run(["pixi", "run", "-e", "anny", "python"])',
        "an Elixir shell-out": 'System.cmd("pixi", ["run", "-e", "anny", "python"])',
        "a shell command line": '    pixi run -e editscore python weft_score.py',
        "a PATH probe": 'if shutil.which("pixi") is None:',
    }
    good = {
        "prose naming pixi": "# The corpus repository keeps its pixi environments, and this does not call them.",
        "a word inside another": "PIXI_LIKE_NAME = 'pixiedust'",
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, line in bad.items():
            path = root / "sample.py"
            path.write_text(line + "\n", encoding="utf-8")
            if offences(path, line):
                print(f"  ok  {label}: rejected")
            else:
                fails.append(label)
                print(f"  BAD {label}: accepted")
        for label, line in good.items():
            path = root / "sample.py"
            path.write_text(line + "\n", encoding="utf-8")
            if offences(path, line):
                fails.append(label)
                print(f"  BAD {label}: rejected, and it is not a call")
            else:
                print(f"  ok  {label}: accepted")
        # ONE ENVIRONMENT PER APP, driven both ways.
        NL = chr(10)

        def notebook(*packages):
            body = [f'  "{name}",' for name in packages]
            return NL.join(["```pyproject.toml", "[project]", "dependencies = ["]
                           + body + ["]", "```", ""])

        app = root / "app"
        (app / "notebooks").mkdir(parents=True)
        (app / "notebooks" / "a.livemd").write_text(notebook("torch", "numpy"), encoding="utf-8")
        (app / "notebooks" / "b.livemd").write_text(notebook("torch", "numpy"), encoding="utf-8")
        if check_one_environment(app):
            fails.append("two notebooks with one environment")
            print("  BAD two notebooks with the same dependency set: rejected")
        else:
            print("  ok  two notebooks with the same dependency set: accepted")

        (app / "notebooks" / "b.livemd").write_text(notebook("torch", "omnigen2"), encoding="utf-8")
        if check_one_environment(app):
            print("  ok  a notebook asking for a second environment: rejected")
        else:
            fails.append("a second environment in one app")
            print("  BAD a notebook asking for a second environment: accepted")

        (app / "notebooks" / "b.livemd").write_text("# no setup cell here" + NL, encoding="utf-8")
        if check_one_environment(app):
            print("  ok  a notebook with no pyproject.toml cell: rejected")
        else:
            fails.append("a notebook with no environment")
            print("  BAD a notebook with no pyproject.toml cell: accepted")

        empty = root / "empty"
        empty.mkdir()
        if check(empty):
            print("  ok  a directory with no sources: rejected")
        else:
            fails.append("empty directory")
            print("  BAD a directory with no sources: accepted")
    print(f"\n{len(fails)} failed")
    return 1 if fails else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    problems = check(PROJECT)
    for problem in problems:
        print(f"  BAD {problem}")
    print(f"\n{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
