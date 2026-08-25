"""A known-answer test for the fullbody 2D fit, plus the controls that make it mean something.

THE POSITIVE CONTROL IS A ROUND TRIP. Pose the rig, project its own 104 joints through a
known weak-perspective camera, throw the pose away, and fit it back from the pixels. The
answer is known by construction, so the residual is a measurement rather than a number: it
says how well the solver recovers a pose that is exactly representable.

THE NEGATIVE CONTROLS ARE WHAT STOP THAT BEING DECORATION. A fit to shuffled targets must
not reach the same residual, or the round trip is measuring the rest pose. A name the
forward does not carry, a target of the wrong shape, and a target with no confidence
anywhere must all raise rather than return something.

Runs on CPU in float64. No card, and no weights beyond what anny ships.

    pixi run -e anny python priv/python/test_loop1_fit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from loop1_fit import bone_names, build_model, fit_2d, region_of, residuals_by_region, subset

FAILS: list[str] = []
ITERS = 60


def expect_raise(label, fn, exc=(ValueError, KeyError)):
    try:
        fn()
    except exc as error:
        print(f"  ok  {label}: rejected ({str(error)[:58]})")
        return
    except Exception as error:  # noqa: BLE001
        FAILS.append(label)
        print(f"  BAD {label}: raised {type(error).__name__}, wanted {exc}")
        return
    FAILS.append(label)
    print(f"  BAD {label}: accepted")


def household(mm):
    anchors = [("penny", 1.52), ("pencil", 7.0), ("AA battery", 14.5), ("nickel", 21.2),
               ("golf ball", 42.7), ("soda can", 66.0)]
    name, size = min(anchors, key=lambda a: abs(mm - a[1]))
    n = mm / size
    return f"about one {name}" if n < 1.5 else f"about {n:.1f} {name}s"


def main() -> int:
    torch.manual_seed(0)
    import roma

    model = build_model()
    names = bone_names(model)
    n = model.bone_count
    print(f"fullbody vocabulary: {len(names)} bone labels")

    truth = 0.08 * torch.randn(n, 3, dtype=torch.float64)
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, n, 1, 1).clone()
    pose[0, :, :3, :3] = roma.rotvec_to_rotmat(truth)
    with torch.no_grad():
        out = model(pose_parameters=pose)
        joints = out["bone_poses"][0, :, :3, 3].to(torch.float64)
        verts = out["vertices"][0].to(torch.float64)
        true_stature = float(verts[:, 1].max() - verts[:, 1].min())

    scale, translation = 900.0, torch.tensor([512.0, 384.0], dtype=torch.float64)
    target = joints[:, :2] * scale + translation
    target_stature = true_stature * scale

    print("known-answer round trip: fit the projection of a pose the rig can hold")
    started = time.monotonic()
    result = fit_2d(model, target, names, iters=ITERS)
    seconds = time.monotonic() - started

    median = float(np.median(result["residual_px"].numpy()))
    worst = float(result["residual_px"].max())
    fraction = median / target_stature
    mm = fraction * 1700.0
    print(f"  median {median:.3f} px, worst {worst:.3f} px, {100 * fraction:.3f}% of stature")
    print(f"  on a 1.7 m body that is {mm:.2f} mm, {household(mm)}")
    print(f"  stature recovered {result['stature_px']:.1f} px against {target_stature:.1f} px")
    print(f"  {seconds:.1f} s on CPU, {ITERS} LBFGS iterations, {len(names)} points")

    if fraction >= 0.02:
        FAILS.append("round trip")
        print("  BAD the round trip did not beat the referee's IMPOSSIBLE threshold of 2%")
    else:
        print("  ok  the round trip is inside the referee's 2% of stature")

    print("negative controls")

    shuffled = target[torch.randperm(len(names), generator=torch.Generator().manual_seed(7))]
    scrambled = fit_2d(model, shuffled, names, iters=ITERS)
    scrambled_median = float(np.median(scrambled["residual_px"].numpy()))
    if scrambled_median <= median * 5:
        FAILS.append("shuffled targets")
        print(f"  BAD shuffled targets fit as well as the real ones ({scrambled_median:.3f} px)")
    else:
        print(f"  ok  shuffled targets do not fit: {scrambled_median:.3f} px against {median:.3f} px")

    expect_raise("a target of the wrong shape",
                 lambda: fit_2d(model, target[:, :1], names, iters=2))
    expect_raise("names that do not match the target rows",
                 lambda: fit_2d(model, target, names[:10], iters=2))
    expect_raise("a target with no confidence anywhere",
                 lambda: fit_2d(model, target, names, confidence=np.zeros(len(names)), iters=2))
    expect_raise("a name the forward does not carry",
                 lambda: subset(names, ["no_such_joint"]))

    regions = residuals_by_region(names, result["residual_px"])
    counts = {r: len(v) for r, v in sorted(regions.items())}
    print(f"regions filled by the fullbody vocabulary: {counts}")
    for wanted in ("body", "feet", "face", "left_hand", "right_hand"):
        if wanted not in regions:
            FAILS.append(f"region {wanted}")
            print(f"  BAD the fullbody vocabulary filled no {wanted}, so the referee cannot run")
    if all(w in regions for w in ("body", "feet", "face", "left_hand", "right_hand")):
        print("  ok  all five referee regions are filled, which a 17-point set never does")

    print(f"\n{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
