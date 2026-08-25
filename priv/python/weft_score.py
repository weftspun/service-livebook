"""EditScore, called as a library rather than through score_edits.py's CLI.

The CLI derives its instruction from the FILENAME, matching against a fixed key list, and
a file matching no key is skipped without a word (score_edits.py L140). A silent skip
reads exactly like a pass, so for a single edit with an instruction the caller typed, this
module calls `EditScore.evaluate([source, edited], instruction)` directly and returns what
came back.

It runs inside the `editscore` pixi environment, not in the notebook's interpreter, and
the notebook reaches it through `weft_loop.pixi_run`. That is deliberate: OmniGen2 pins
torch 2.6.0+cu124 and EditScore pins cu128, so one interpreter cannot hold both.

    python weft_score.py --source a.png --edited b.png --instruction "..." --out score.json

Quantisation is permitted here and forbidden for a generator. This is a verifier, and
CLAUDE.md's condition 5 is about what writes corpus data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CORPUS = Path(r"C:\weftspun-keypoint\6-datasource\anny-render-corpus")

BASE = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER = "EditScore/EditScore-Qwen3-VL-8B-Instruct"


def cap(image, max_pixels: int):
    if not max_pixels or image.width * image.height <= max_pixels:
        return image
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))


_SCORER = {}


def scorer(precision: str = "nf4"):
    """The loaded EditScore, built once per interpreter and kept.

    A notebook scores a render each round, and reloading an 8B backbone every round would
    make the loop measure the loader. The cache is keyed by precision because the two are
    different models on the card.
    """
    if precision in _SCORER:
        return _SCORER[precision]

    # The NF4 monkeypatch is score_edits.py's, reused rather than copied. It is importable
    # because that file guards its own main() behind __main__. The corpus directory has to
    # be on the path explicitly: Python seeds sys.path[0] with this file's directory, not
    # the caller's, and score_edits.py lives beside the corpus scripts.
    if str(CORPUS) not in sys.path:
        sys.path.insert(0, str(CORPUS))
    if precision == "nf4":
        from score_edits import patch_for_4bit

        patch_for_4bit()

    from editscore import EditScore

    _SCORER[precision] = EditScore(
        backbone="qwen3vl", model_name_or_path=BASE, lora_path=ADAPTER, score_range=25
    )
    return _SCORER[precision]


def evaluate(source_path, edited_path, instruction, precision: str = "nf4",
             max_pixels: int = 262144, out=None) -> dict:
    """Score one edit and return the record, writing it beside the image when asked.

    This is the function the notebook calls. `main` below is the same thing behind an
    argument parser, so the command line and the notebook cannot drift apart.
    """
    from PIL import Image

    source = cap(Image.open(source_path).convert("RGB"), max_pixels)
    edited = cap(Image.open(edited_path).convert("RGB"), max_pixels)

    started = time.monotonic()
    raw = scorer(precision).evaluate([source, edited], instruction)
    seconds = time.monotonic() - started

    overall = raw.get("overall") if isinstance(raw, dict) else None
    refused = not isinstance(overall, (int, float))

    peak = None
    try:
        import torch

        peak = torch.cuda.max_memory_allocated() / 1024**3
    except Exception:  # noqa: BLE001
        pass

    result = {
        "source": str(source_path),
        "edited": str(edited_path),
        "instruction": instruction,
        "base": BASE,
        "adapter": ADAPTER,
        "precision": precision,
        "max_pixels": max_pixels,
        "overall": overall,
        "refused": refused,
        "seconds": seconds,
        "peak_vram_gib": peak,
        "raw": ({k: v for k, v in (raw or {}).items() if k != "raw"}
                if isinstance(raw, dict) else str(raw)[:400]),
    }
    if out is not None:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--max-pixels", type=int, default=262144)
    args = parser.parse_args()

    result = evaluate(args.source, args.edited, args.instruction,
                      precision=args.precision, max_pixels=args.max_pixels, out=args.out)
    overall, refused, peak = result["overall"], result["refused"], result["peak_vram_gib"]

    # A refusal is reported as a failure rather than as a zero, because a zero would enter
    # the history as a measurement and a refusal is the absence of one.
    print(json.dumps({"overall": overall, "refused": refused, "peak_vram_gib": peak}))
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
