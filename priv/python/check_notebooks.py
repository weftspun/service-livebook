"""Check every loop notebook the way loop 4 was checked by hand.

Loop 4's latent arm used a name that did not exist, passed a latent where a mesh was
required, and defined a function below the cell that called it. All three are findable
without running anything, and finding them by hand once is not a method.

WHAT IS CHECKED, per notebook, in cell order.

1. NAMES RESOLVE. A symbol table is built cell by cell -- imports, assignments, function
   and class definitions, comprehension and parameter bindings -- and every load of a name
   is checked against it plus the builtins. This is what catches `region_b64`.

2. SCRIPTS EXIST. Every path handed to `pixi_run` is resolved on disk. A notebook naming a
   script that is not there fails here rather than after the environment has loaded.

3. FLAGS EXIST. Every `--flag` passed to one of those scripts is looked for in that
   script's own argparse calls. A renamed flag is silently accepted by nothing.

4. ENVIRONMENTS EXIST. Every `pixi_run("name", ...)` names an environment declared in the
   corpus repository's `pixi.toml`.

WHAT IS NOT CHECKED. Whether a loop is a good loop, whether a score means anything, and
anything that needs the model to run. This is a static read.

A missing corpus checkout is a FAIL, not a skip, because a silent skip reads exactly like a
pass. Run `--self-test` to see each check reject a known-bad notebook.

    python priv/python/check_notebooks.py [--self-test]
"""
from __future__ import annotations

import ast
import builtins
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent.parent / "notebooks"
CORPUS = Path(r"C:\weftspun-keypoint\6-datasource\anny-render-corpus")

CELL_RE = re.compile(r"```python\n(.*?)```", re.S)
# Both quote styles, because `ast.unparse` emits single quotes and the scripts being
# checked are written with double ones. Matching only one style made the flag check find no
# script at all and pass silently, which its own negative control caught.
FLAG_RE = re.compile(r"""['"](--[a-z0-9-]+)['"]""")
PIXI_RE = re.compile(r'pixi_run\(\s*"([a-z0-9_-]+)"', re.S)

# Names a Livebook Python cell may use without this file seeing them bound. Kept short and
# explicit: every entry is a name the notebook genuinely receives from elsewhere.
AMBIENT = {"__file__", "__name__"}


class Table:
    """The names bound so far, in cell order."""

    def __init__(self):
        self.names = set(dir(builtins)) | AMBIENT

    def bind(self, name):
        self.names.add(name)

    def has(self, name):
        return name in self.names


def bound_by(node) -> set[str]:
    """Every name a statement binds, including inside comprehensions and functions."""
    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            out.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(child.name)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = child.args
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                    out.add(arg.arg)
                if args.vararg:
                    out.add(args.vararg.arg)
                if args.kwarg:
                    out.add(args.kwarg.arg)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                out.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            out.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            out.update(child.names)
    return out


def loaded_in(node) -> list[tuple[str, int]]:
    """(name, line) for every load, skipping attribute bases we cannot resolve."""
    return [
        (child.id, child.lineno)
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    ]


def cells(source: str) -> list[str]:
    return CELL_RE.findall(source)


def check_names(name, source, problems):
    table = Table()
    for index, cell in enumerate(cells(source), 1):
        try:
            tree = ast.parse(cell)
        except SyntaxError as error:
            problems.append(f"{name}: cell {index} does not parse: {error}")
            continue
        # A statement's own bindings count while checking it. Without that, every function
        # parameter reads as an unbound name -- the first run of this self-test rejected
        # its own clean case on the `i` of `def propose(i)`. What survives is the check
        # that matters: a name loaded in a cell where nothing above it binds the name.
        for statement in tree.body:
            local = bound_by(statement)
            for used, line in loaded_in(statement):
                if not table.has(used) and used not in local:
                    problems.append(f"{name}: cell {index} line {line} uses {used!r}, bound nowhere above")
            for bound in local:
                table.bind(bound)


def script_calls(source):
    """(script path, flags) for every pixi_run whose first argument is a script."""
    out = []
    for cell in cells(source):
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "pixi_run"):
                continue
            if len(node.args) < 2:
                continue
            segment = ast.unparse(node.args[1])
            script = re.search(r"""['"]([^'"]*\.py)['"]""", segment)
            flags = set(FLAG_RE.findall(segment))
            if script:
                out.append((Path(script.group(1)).name, flags, segment))
    return out


def resolve(script: str, root: Path) -> Path | None:
    for candidate in (root / script, HERE / script, CORPUS / script,
                      Path(r"C:\weftspun-keypoint\1-transport\rf-detr-mcp") / script):
        if candidate.is_file():
            return candidate
    return None


def check_scripts(name, source, root, problems):
    for script, flags, _segment in script_calls(source):
        path = resolve(script, root)
        if path is None:
            problems.append(f"{name}: names {script}, which is on no path this checks")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        declared = set(FLAG_RE.findall(text))
        for flag in sorted(flags):
            if flag not in declared:
                problems.append(f"{name}: passes {flag} to {script}, which declares no such flag")


def check_environments(name, source, corpus, problems):
    pixi = corpus / "pixi.toml"
    if not pixi.is_file():
        problems.append(f"{name}: {pixi} is missing, so no environment name can be checked")
        return
    text = pixi.read_text(encoding="utf-8", errors="ignore")
    declared = set(re.findall(r"^([a-z0-9_-]+)\s*=\s*\{?\s*features", text, re.M))
    declared |= set(re.findall(r"\[feature\.([a-z0-9_-]+)", text))
    for env in sorted(set(PIXI_RE.findall(source))):
        if env not in declared:
            problems.append(f"{name}: runs in pixi environment {env!r}, declared in no pixi.toml")


def check(notebooks=NOTEBOOKS, corpus=CORPUS, root=None):
    problems = []
    notebooks = Path(notebooks)
    root = Path(root) if root else HERE
    files = sorted(notebooks.glob("*.livemd"))
    if not files:
        return [f"no notebooks under {notebooks}, which is never correct here"]
    if not Path(corpus).is_dir():
        problems.append(f"the corpus checkout {corpus} is missing; flags and environments are unchecked")
    for path in files:
        source = path.read_text(encoding="utf-8")
        name = path.name
        check_names(name, source, problems)
        check_scripts(name, source, root, problems)
        if Path(corpus).is_dir():
            check_environments(name, source, Path(corpus), problems)
    return problems


GOOD = """# A notebook

```python
from pathlib import Path
from weft_loop import pixi_run

WORK = Path("work")


def propose(i):
    pixi_run("editscore", [str(WORK / "weft_score.py"), "--source", "a.png"])
    return str(WORK / f"round_{i}.png")
```
"""


def self_test():
    import shutil
    import tempfile

    def build(tmp, body=GOOD, script="--source"):
        root = Path(tmp)
        (root / "notebooks").mkdir()
        (root / "notebooks" / "n.livemd").write_text(body, encoding="utf-8")
        (root / "weft_score.py").write_text(
            f'parser.add_argument("{script}")\n', encoding="utf-8")
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "pixi.toml").write_text("[feature.editscore]\n", encoding="utf-8")
        return root

    cases = [
        ("a clean notebook passes", {}, False),
        ("a name bound nowhere above",
         {"body": GOOD.replace('"--source", "a.png"', '"--source", region_b64')}, True),
        ("a name used in an earlier cell than it is bound",
         {"body": "# A\n\n```python\nprint(later)\n```\n\n```python\nlater = 1\n```\n"}, True),
        ("a cell that does not parse",
         {"body": "# A\n\n```python\ndef broken(\n```\n"}, True),
        ("a flag the script does not declare", {"script": "--src"}, True),
        ("an environment no pixi.toml declares",
         {"body": GOOD.replace('pixi_run("editscore"', 'pixi_run("nosuchenv"')}, True),
    ]

    ok = True
    print("self-test: each known-bad notebook must be rejected")
    for label, kw, should_fail in cases:
        tmp = tempfile.mkdtemp()
        try:
            root = build(tmp, **kw)
            found = check(root / "notebooks", root / "corpus", root)
            failed = bool(found)
            if failed != should_fail:
                ok = False
            mark = "ok " if failed == should_fail else "BAD"
            print(f"  {mark} {label}: {'rejected' if failed else 'accepted'} {found[0][:58] if found else ''}")
        finally:
            shutil.rmtree(tmp)
    return 0 if ok else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    found = check()
    for line in found:
        print(line)
    print(f"{len(found)} problems")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
