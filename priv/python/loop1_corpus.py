"""Loop 1's other half: write the corpus, and let the schema's own validator be the score.

THERE ARE FOUR LOOPS, NOT FIVE, AND THIS FILE WAS BRIEFLY A FIFTH. `todo.md` names loop one
as "RFDetr Fullbody Coco Keypoints-Somax-Anny to EditScore to Image (corpus data
generator)", so corpus generation is what loop one is FOR, not a new loop beside it. The
mistake is recorded here rather than quietly renamed away: a fifth loop would have produced
a second place where corpus rules live, which is the failure this schema exists to prevent.

THE OTHER THREE LOOPS SCORE WITH A MODEL. This half does not, and the difference is the
point. A
corpus is right or wrong against rules that are already written down --
`anny_render_schema.py` carries 28 relations, its foreign keys, and a `validate()` that
enforces ETNF, split hygiene and CLAUDE.md's five conditions for generated data. So the
scorer here is that function, and the loop's score is the negated problem count: zero is the
ceiling and every step away from it names itself.

WHAT WAS MISSING WAS THE WRITER, and `omnigen2_edit.py` says so in its own closing line:
the provenance goes to a sidecar JSON because the relations need a `render_id` to join to
and the renderer does not emit parquet yet. A sidecar parts company with its images the
first time somebody copies them, which is condition 1 failing quietly. This writes the rows.

WHAT IT REFUSES TO WRITE, and this is the useful half:

* A quantised run. Condition 5 says quantised weights do not produce corpus data, so an
  NF4 edit becomes device-sizing evidence and is written to a separate directory that no
  training loader reads. Today's NF4 run is the test fixture for exactly this.
* Generated pixels in the constructed pool. `edited_renders` and `render_data` stay
  separate relations, because the way condition 2 gets violated is never a decision -- it is
  a convenience write so that one loader can read one table.
* A row with a null in it. Every column is filled or the row does not exist.

NO `corpus_eligible` COLUMN. Eligibility is a pure function of `edit_runs.precision`, so
storing it would be a derivable column, which is a second place the fact lives and a second
place it can disagree. `validate()` derives it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path(r"C:\weftspun-keypoint\6-datasource\anny-render-corpus")


def _schema_module():
    """The schema is the corpus repository's, imported rather than restated."""
    if str(CORPUS) not in sys.path:
        sys.path.insert(0, str(CORPUS))
    import anny_render_schema

    return anny_render_schema


def deterministic_id(*parts, bits: int = 63) -> int:
    """A stable id from the things that define a row, never a counter.

    A counter depends on insertion order, so re-running the same render twice gives the same
    row two identities and the corpus grows a duplicate nothing can detect. Hashing what the
    row *is* makes a repeat write idempotent.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - bits)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_provenance(path) -> dict:
    """`omnigen2_edit.py`'s sidecar, which is where the conditioning currently lives."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_from_edit_run(provenance: dict, render_ids: dict[str, int], started: str | None = None):
    """Turn one sidecar into `edit_models`, `edit_prompts`, `edit_runs`, `edited_renders`.

    `render_ids` maps each output name -- "photographic", "colour-sketch" -- onto the
    `render_id` of the constructed frame it was edited from. That join is the whole reason
    the sidecar could not become rows on its own.
    """
    args = provenance.get("args", provenance)
    repo = provenance.get("model") or args.get("model") or "OmniGen2/OmniGen2"
    revision = provenance.get("code_revision") or provenance.get("revision") or "unknown"
    precision = args.get("precision") or provenance.get("precision") or "bf16"
    negative = args.get("negative_prompt") or args.get("negative") or ""

    edit_model_id = deterministic_id(repo, revision, bits=15)
    negative_id = deterministic_id("negative", negative, bits=31)

    models = [{"edit_model_id": edit_model_id, "repo_id": repo, "revision": revision}]
    prompts = [{"prompt_id": negative_id, "text": negative}]
    runs, edited = [], []

    for name, out in provenance.get("outputs", {}).items():
        if name not in render_ids:
            raise KeyError(
                f"output {name!r} has no render_id: an edited frame with no constructed "
                "frame to join to is unprovenanced, and condition 1 is not met"
            )
        prompt_id = deterministic_id("prompt", out["prompt"], bits=31)
        prompts.append({"prompt_id": prompt_id, "text": out["prompt"]})
        edit_run_id = deterministic_id(edit_model_id, prompt_id, negative_id, precision,
                                       args.get("steps", 0), out["seed"], bits=31)
        runs.append({
            "edit_run_id": edit_run_id,
            "edit_model_id": edit_model_id,
            "prompt_id": prompt_id,
            "negative_prompt_id": negative_id,
            "precision": precision,
            "steps": int(args.get("steps", 0)),
            "text_guidance_scale": float(args.get("text_guidance_scale", 0.0)),
            "image_guidance_scale": float(args.get("image_guidance_scale", 0.0)),
            "started_utc": started or utc_now(),
        })
        edited.append({
            "edit_id": deterministic_id("edit", edit_run_id, render_ids[name]),
            "render_id": render_ids[name],
            "edit_run_id": edit_run_id,
            "seed": int(out["seed"]),
            "file": out["file"],
        })

    # A prompt repeats once per image, which is why it is interned; dedupe on the way out.
    prompts = list({p["prompt_id"]: p for p in prompts}.values())
    return {"edit_models": models, "edit_prompts": prompts, "edit_runs": runs,
            "edited_renders": edited}


def is_quantised(precision: str) -> bool:
    return str(precision).lower() in _schema_module().QUANTISED_PRECISIONS


def destination(root: Path, precision: str) -> Path:
    """Where a run's rows go: the corpus, or the evidence pile beside it.

    Separate directories rather than a flag on a row. A loader that reads the corpus
    directory cannot accidentally read the other one, and no filter has to be remembered.
    """
    root = Path(root)
    return root if not is_quantised(precision) else root.parent / f"{root.name}-device-evidence"


def write_relations(root: Path, tables: dict[str, list[dict]], images: dict[int, bytes] | None = None):
    """Write each relation as a parquet file under `root`, dropping the helper columns.

    `file` is carried through `rows_from_edit_run` for the caller's convenience and is not a
    column of `EDITED_RENDERS`; the bytes go in `image`, because a path is a reference to
    something that can move and the schema stores the pixels.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema_module = _schema_module()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    written = {}

    for name, rows in tables.items():
        schema = schema_module.RELATIONS[name]
        prepared = []
        for row in rows:
            row = dict(row)
            path = row.pop("file", None)
            if name == "edited_renders":
                blob = (images or {}).get(row["edit_id"])
                if blob is None:
                    raise ValueError(
                        f"edited_render {row['edit_id']} has no image bytes; a row without its "
                        "payload is a null by another name"
                    )
                row["image"] = blob
            for field in schema:
                if field.name not in row:
                    raise ValueError(f"{name}: row is missing {field.name}, and ETNF forbids a null")
            prepared.append({field.name: row[field.name] for field in schema})
        table = pa.Table.from_pylist(prepared, schema=schema)
        out = root / f"{name}.parquet"
        pq.write_table(table, out, compression="zstd")
        written[name] = out

    return written


def score(root) -> int:
    """The loop's score: minus the number of problems, so zero is the ceiling.

    Higher is better, matching the harness. A corpus that will not open at all scores worse
    than one with a few bad rows, which is the right ordering.
    """
    problems = _schema_module().validate(str(root))
    return -len(problems)


def problems(root) -> list:
    return _schema_module().validate(str(root))
