"""A known-answer test for the 2D fit, plus the controls that make it mean something.

THE POSITIVE CONTROL IS A ROUND TRIP. Pose the rig, project its own COCO-17 keypoints
through a known weak-perspective camera, throw the pose away, and fit it back from the 17
pixels. The answer is known by construction, so the residual is a measurement rather than a
number: it says how well the solver recovers a pose that is exactly representable.

THE NEGATIVE CONTROLS ARE WHAT STOP THAT BEING DECORATION. A fit to shuffled targets must
not reach the same residual, or the round trip is measuring nothing but the rest pose. A
target of the wrong shape, and one with no confidence anywhere, must raise rather than
return something.

Runs on CPU in float64. No card, no weights beyond the `coco.pth` that ships with anny.

    pixi run -e anny python priv/python/test_loop1_fit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from loop1_fit import COCO17, build_model, fit_2d, keypoint_names, residuals_by_region, subset_17

FAILS: list[str] = []
ITERS = 60


def expect_raise(label, fn, exc=(ValueError, KeyError)):
    try:
        fn()
    except exc as error:
        print(f"  ok  {label}: rejected ({str(error)[:60]})")
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
    model = build_model()

    from anny.keypoints import KeypointsRegressor
    import roma

    regressor = KeypointsRegressor.coco(model)
    names = keypoint_names(regressor)
    idx = subset_17(names)
    n = model.bone_count

    # A pose the rig can hold: small rotations on every bone, so the answer exists exactly.
    truth = 0.08 * torch.randn(n, 3, dtype=torch.float64)
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, n, 1, 1).clone()
    pose[0, :, :3, :3] = roma.rotvec_to_rotmat(truth)
    with torch.no_grad():
        out = model(pose_parameters=pose)
        kp3d = regressor(out)[0].to(torch.float64)[idx]
        verts = out["vertices"][0].to(torch.float64)
        true_stature = float(verts[:, 1].max() - verts[:, 1].min())

    scale, translation = 900.0, torch.tensor([512.0, 384.0], dtype=torch.float64)
    target = kp3d[:, :2] * scale + translation
    target_stature = true_stature * scale

    print("known-answer round trip: fit the projection of a pose the rig can hold")
    started = time.monotonic()
    result = fit_2d(model, regressor, target, iters=ITERS)
    seconds = time.monotonic() - started

    median = float(np.median(result["residual_px"].numpy()))
    worst = float(result["residual_px"].max())
    fraction = median / target_stature
    # 1.7 m of stature is the referee's own reference body, so the pixel residual is
    # reported as the millimetres it would be on one.
    mm = fraction * 1700.0
    print(f"  median {median:.3f} px, worst {worst:.3f} px, {100 * fraction:.3f}% of stature")
    print(f"  on a 1.7 m body that is {mm:.2f} mm, {household(mm)}")
    print(f"  stature recovered {result['stature_px']:.1f} px against {target_stature:.1f} px")
    print(f"  {seconds:.1f} s on CPU, {ITERS} LBFGS iterations")

    if fraction >= 0.02:
        FAILS.append("round trip")
        print("  BAD the round trip did not beat the referee's IMPOSSIBLE threshold of 2%")
    else:
        print("  ok  the round trip is inside the referee's 2% of stature")

    print("negative controls")

    shuffled = target[torch.randperm(17, generator=torch.Generator().manual_seed(7))]
    scrambled = fit_2d(model, regressor, shuffled, iters=ITERS)
    scrambled_median = float(np.median(scrambled["residual_px"].numpy()))
    if scrambled_median <= median * 5:
        FAILS.append("shuffled targets")
        print(f"  BAD shuffled targets fit as well as the real ones ({scrambled_median:.3f} px)")
    else:
        print(f"  ok  shuffled targets do not fit: {scrambled_median:.3f} px against {median:.3f} px")

    expect_raise("a target of the wrong shape",
                 lambda: fit_2d(model, regressor, target[:10], iters=2))
    expect_raise("a target with no confidence anywhere",
                 lambda: fit_2d(model, regressor, target, confidence=np.zeros(17), iters=2))
    expect_raise("a keypoint asset missing a COCO name",
                 lambda: subset_17([n for n in names if n != "nose"]))

    regions = residuals_by_region(result["residual_px"])
    print(f"regions filled: {sorted(regions)} -- face and hands stay absent, so the referee "
          f"answers NOT_RUN")

    print(f"\n{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
