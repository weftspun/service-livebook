"""Follow every reference the loops make, and report the ones that go nowhere.

`check_notebooks.py` reads names and flags. This reads the other kind of reference: the
absolute paths a cell opens, the HTTP routes it calls, and the links the three root
documents carry. Those are the references that break silently, because a path that does not
exist looks exactly like a path that does until something opens it.

WHAT IS CHECKED.

1. INPUT PATHS. Every absolute path literal in a notebook that is read rather than written.
   A path built from the notebook's own `WORK` directory is an output and is skipped, since
   the notebook creates it. Everything else has to be there already.

2. ROUTES. Every `f"{SERVICE}/route"` call is matched against the route the corresponding
   server declares. A notebook calling `/edit` on a server that offers `/predict` fails
   here rather than at the first round.

3. DOCUMENT LINKS. Relative links and `src` targets in the root documents resolve on disk.
   Absolute URLs are reported as unchecked and counted rather than silently passed, because
   an unchecked link is not a working one.

WHAT IS NOT CHECKED. Whether a file has the right contents, and whether a route behaves.
This is reachability only.

    python priv/python/check_links.py [--self-test]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent.parent / "notebooks"
ROOT = Path(r"C:\weftspun-keypoint")
MANIFEST = ROOT / ".repo" / "manifests" / "default.xml"


def root_documents(manifest: Path = MANIFEST, root: Path = ROOT):
    """The root documents, read from the manifest rather than listed here.

    Every file at the workspace root is a `<linkfile>`, so the manifest already says which
    documents exist and a second list here is a second place the fact lives. It drifted
    exactly that way: this file named `fourloops-etnf.html` while the manifest had moved to
    `fourloops-etnf.usda`, and the checker reported a broken link that was its own.
    """
    if not manifest.is_file():
        return []
    text = manifest.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r'<linkfile[^>]*dest="([^"]+)"', text)
    return [root / n for n in names if Path(n).suffix in {".md", ".html", ".usda"}]

CELL_RE = re.compile(r"```python\n(.*?)```", re.S)
ABS_PATH_RE = re.compile(r"""[rR]?['"]([A-Za-z]:[\\/][^'"]*)['"]""")
ROUTE_RE = re.compile(r"""f['"]\{(\w+)\}(/[a-z_/]+)['"]""")
LINK_RE = re.compile(r"""(?:href|src)=['"]([^'"#][^'"]*)['"]""")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Which repository serves which constant. A notebook names a port; the routes it may call
# are the ones that repository's server declares.
SERVERS = {
    "CYCLEGAN": ROOT / "3-interactor" / "cyclegan-style-transfer" / "server.py",
    "PIXAL3D": ROOT / "3-interactor" / "pixal3d-image-to-textured-mesh" / "server.py",
    "VOXHAMMER": ROOT / "3-interactor" / "voxhammer-image-mesh-editing" / "server.py",
}


def cells(source):
    return CELL_RE.findall(source)


def work_dirs(source):
    """Path constants the notebook creates, whose children are outputs rather than inputs."""
    out = set()
    for cell in cells(source):
        for match in re.finditer(r"(\w+)\s*=\s*Path\(\s*[rR]?['\"]([^'\"]+)['\"]", cell):
            if ".loop" in match.group(2) or match.group(1) in {"WORK"}:
                out.add(match.group(2).replace("\\", "/").rstrip("/"))
    return out


def check_paths(name, source, problems, unchecked):
    outputs = work_dirs(source)
    for cell in cells(source):
        for raw in ABS_PATH_RE.findall(cell):
            path = raw.replace("\\", "/")
            if any(path.startswith(out) for out in outputs):
                continue
            if Path(path).exists():
                continue
            problems.append(f"{name}: reads {raw}, which does not exist")


def declared_routes(server: Path):
    if not server.is_file():
        return None
    text = server.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"""@app\.(?:get|post)\(['"]([^'"]+)['"]""", text))


def check_routes(name, source, problems):
    for cell in cells(source):
        for constant, route in ROUTE_RE.findall(cell):
            server = SERVERS.get(constant)
            if server is None:
                problems.append(f"{name}: calls {route} on {constant}, which names no server here")
                continue
            routes = declared_routes(server)
            if routes is None:
                problems.append(f"{name}: {server} is missing, so {route} cannot be checked")
                continue
            if route not in routes:
                problems.append(
                    f"{name}: calls {route} on {constant}, which declares {sorted(routes)}"
                )


def check_document(path: Path, problems, unchecked):
    if not path.exists():
        problems.append(f"{path.name}: missing, and it is linked from the workspace root")
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    targets = LINK_RE.findall(text) + MD_LINK_RE.findall(text)
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:")):
            unchecked.append(f"{path.name} -> {target}")
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            problems.append(f"{path.name}: links {target}, which resolves to nothing")


def check(notebooks=NOTEBOOKS, docs=None):
    problems, unchecked = [], []
    notebooks = Path(notebooks)
    files = sorted(notebooks.glob("*.livemd"))
    if not files:
        problems.append(f"no notebooks under {notebooks}, which is never correct here")
    for path in files:
        source = path.read_text(encoding="utf-8")
        check_paths(path.name, source, problems, unchecked)
        check_routes(path.name, source, problems)
    for doc in (root_documents() if docs is None else docs):
        check_document(Path(doc), problems, unchecked)
    return problems, unchecked


GOOD_NOTEBOOK = '''# A notebook

```python
from pathlib import Path

WORK = Path(r"C:\\weftspun-keypoint\\.loopX")
PIXAL3D = "http://localhost:8002"
SOURCE = r"{source}"


def go():
    requests.post(f"{{PIXAL3D}}/predict", json={{}})
    return WORK / "out.png"
```
'''


def self_test():
    import shutil
    import tempfile

    def build(tmp, notebook=None, doc="<a href='there.txt'>x</a>", there=True, source=None,
              route=None):
        root = Path(tmp)
        (root / "notebooks").mkdir()
        real = root / "input.png"
        real.write_bytes(b"x")
        template = GOOD_NOTEBOOK if route is None else GOOD_NOTEBOOK.replace("/predict", route)
        body = notebook or template.format(source=(source or str(real)).replace("\\", "\\\\"))
        (root / "notebooks" / "n.livemd").write_text(body, encoding="utf-8")
        if there:
            (root / "there.txt").write_text("x", encoding="utf-8")
        (root / "doc.html").write_text(doc, encoding="utf-8")
        return root

    cases = [
        ("a clean notebook and document pass", {}, False),
        ("an input path that does not exist",
         {"source": r"C:\nowhere\missing.png"}, True),
        # The source must be a path that exists, or this case reports the missing file and
        # the route check goes unexercised -- a control that passes for the wrong reason
        # tells you nothing about the check it was written for.
        ("a route the server does not declare", {"route": "/enhance"}, True),
        ("a document link that resolves to nothing", {"there": False}, True),
    ]

    ok = True
    print("self-test: each known-bad reference must be reported")
    for label, kw, should_fail in cases:
        tmp = tempfile.mkdtemp()
        try:
            root = build(tmp, **{k: v for k, v in kw.items()})
            problems, _ = check(root / "notebooks", [root / "doc.html"])
            failed = bool(problems)
            if failed != should_fail:
                ok = False
            mark = "ok " if failed == should_fail else "BAD"
            print(f"  {mark} {label}: {'reported' if failed else 'clean'} {problems[0][:56] if problems else ''}")
        finally:
            shutil.rmtree(tmp)
    return 0 if ok else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    problems, unchecked = check()
    for line in problems:
        print(line)
    for line in unchecked:
        print(f"unchecked (absolute URL): {line}")
    print(f"{len(problems)} broken, {len(unchecked)} unchecked")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
