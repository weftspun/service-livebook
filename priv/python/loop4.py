"""Loop 4's service contracts, in one importable place so they can be tested.

The notebook keeps the loop visible. What lives here is the fiddly half: which endpoint
takes what, and how the latent arm reports that it cannot run. That half is worth a test,
and a notebook cell is not testable.

WHAT WAS WRONG BEFORE. The notebook's latent arm had four defects, and the first three
would have raised on the first round that took it:

* `region_b64` was used and never defined, so the call raised NameError.
* VoxHammer was handed Pixal3D's latent. It takes a mesh; the prose said so and the code
  did not, so the arm skipped `/extract` entirely.
* `extract` was defined below the cell that needed it.
* Unavailability was detected by catching HTTP 500. In stub mode VoxHammer answers 200
  with `stub: true`, having marked each of its seven plan steps done without editing
  anything, so the arm would have reported a successful repair. That is the silent pass
  the notebook claims to avoid, in the one place it claimed to avoid it.

`require_voxhammer` reads `/health` instead. Stub is not a degraded mode here, it is a
different answer: the dispatcher ran and the model did not.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from weft_loop import PreconditionFailed

# VoxHammer needs a mask and there is no segmentation step in this loop yet. The whole
# object is the honest placeholder: it says "edit everything" rather than implying a
# region somebody chose. The shape matches the server's own test_input.json.
WHOLE_OBJECT = {"box": [0, 0, 0, 1, 1, 1]}


def b64(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def region(mask: dict | None = None) -> str:
    return base64.b64encode(json.dumps(mask or WHOLE_OBJECT).encode()).decode("ascii")


def voxhammer_status(url: str, timeout: int = 10) -> dict:
    """`/health` as the server reports it: ready, and whether it is stubbed."""
    response = requests.get(f"{url}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def require_voxhammer(url: str, timeout: int = 10) -> dict:
    """Raise unless VoxHammer can actually edit. Stub counts as cannot.

    The router selected this arm because the views disagreed about the shape. Falling
    through to the 2D arm would record a geometry failure as an appearance failure that was
    repaired, so this raises and the round is recorded as unavailable instead.
    """
    try:
        status = voxhammer_status(url, timeout)
    except requests.RequestException as error:
        raise PreconditionFailed(f"VoxHammer at {url} did not answer /health: {error}") from error

    if status.get("stub"):
        raise PreconditionFailed(
            "VoxHammer answered /health with stub=true. Its seven plan steps dispatch and "
            "each raises NotImplementedError outside stub mode, so a stubbed reply means "
            "the dispatcher ran and nothing was edited. The round is unavailable, not "
            "repaired."
        )
    if not status.get("ready"):
        raise PreconditionFailed(f"VoxHammer at {url} is not ready: {status}")
    return status


def extract(url: str, state_b64: str, work: Path, decimation_target: int = 20000,
            texture_size: int = 2048, timeout: int = 1800):
    """Pixal3D `/extract`: the latent becomes a glb and a USD layer.

    Deferred until a score earns it in the ordinary case, and unavoidable in the latent
    arm, because VoxHammer takes a mesh rather than the latent.
    """
    response = requests.post(
        f"{url}/extract",
        json={"state": state_b64, "decimation_target": decimation_target,
              "texture_size": texture_size},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    work = Path(work)
    glb, layer = work / "mesh.glb", work / "mesh.usda"
    glb.write_bytes(base64.b64decode(body["glb"]))
    layer.write_bytes(base64.b64decode(body["layer"]))
    return glb, layer


def repair_latent(voxhammer_url: str, pixal3d_url: str, state_b64: str, source: str,
                  work: Path, seed: int = 42, mask: dict | None = None, timeout: int = 1800):
    """The latent arm, in the order the services actually require.

    Health first, then `/extract` because the editor takes a mesh, then the edit. It stops
    at the edit today and says why: VoxHammer returns a USD layer and nothing in this loop
    turns a USD layer back into the `.npz` that `render_view.py` renders, so the edited
    shape cannot be scored even once the editor works. That second gap is stated here
    rather than discovered after the first one closes.
    """
    require_voxhammer(voxhammer_url)
    glb, _layer = extract(pixal3d_url, state_b64, work)

    response = requests.post(
        f"{voxhammer_url}/predict",
        json={"mesh": b64(glb), "reference": b64(source), "region": region(mask), "seed": seed},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()

    kept = Path(work) / "voxhammer.json"
    kept.write_text(json.dumps(body, indent=2), encoding="utf-8")

    raise PreconditionFailed(
        "VoxHammer returned a USD layer and this loop has no path from a USD layer back to "
        "the .npz that render_view.py renders, so the edited shape cannot be scored. The "
        f"reply is kept at {kept} rather than thrown away."
    )
