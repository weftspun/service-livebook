"""Fit an ANNY pose to 2D keypoints, which is the front end loop 1 was missing.

WHAT WAS ACTUALLY MISSING. The notebook said no points-to-pose solver existed, and that was
wrong. `4-entities/anny-pose-retarget-work/fit_joints_to_anny.py` solves a full 104-bone
pose from 21 *three-dimensional* joint positions, by Umeyama for the global placement and
LBFGS with a strong-Wolfe line search for the per-bone rotations. What is missing is only
the front end: RF-DETR gives 2D pixels, and that file takes 3D world positions.

This is that front end, built the same way and reusing the same two ideas.

THE FORWARD IS ALREADY DIFFERENTIABLE, so no inverse model is needed and none is licensed.
`anny.keypoints.KeypointsRegressor` is a per-vertex convex blend from vertices to 23 named
keypoints, `coco.pth` ships with the package, and everything from pose parameters to those
keypoints is torch. Composing a camera onto the end gives 2D, and the loss is a
reprojection distance.

THE CAMERA IS WEAK PERSPECTIVE, and the choice is the honest one rather than the convenient
one. A full perspective camera needs a focal length, which a single uncalibrated photograph
does not carry; solving for one alongside the pose trades depth against focal length and the
pair is not separable from silhouette-free 2D points. Weak perspective has three parameters,
all of which the closed-form initial below solves exactly, so nothing here is optimised that
can be computed.

WHAT THIS DOES NOT DO. It does not recover depth. Two poses whose keypoints project to the
same pixels are indistinguishable to this loss, and the classic pair is a limb toward the
camera against the same limb away from it. `pose-consensus/python/depth_term.py` exists for
exactly that and is not wired in here. A fit from this file is a 2D-consistent pose, which
is a smaller claim than a correct one, and the referee's verdict is what says whether the
body could hold it at all.
"""
from __future__ import annotations

import numpy as np
import torch

COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def build_model():
    """The 19,158-vertex makehuman topology, because `coco.pth` is indexed against it."""
    import anny
    from anny.models.model_data import TopologyConfig

    model = anny.Anny(
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False)
    )
    assert model()["vertices"].shape[1] == 19158, "wrong topology; coco.pth expects 19,158"
    return model


def keypoint_names(regressor, fallback=COCO17):
    names = getattr(regressor, "labels", None) or getattr(regressor, "names", None)
    return list(names) if names else list(fallback)


def subset_17(names):
    """Indices of the COCO-17 inside whatever the regressor emits.

    The asset carries 23 -- COCO-17 plus six foot points -- and the detector carries 17.
    Taking the subset by name is the only version of this that cannot silently misalign,
    which is why a missing name raises rather than shortening the array.
    """
    index = {n: i for i, n in enumerate(names)}
    missing = [n for n in COCO17 if n not in index]
    if missing:
        raise KeyError(f"the keypoint asset is missing {missing}")
    return torch.tensor([index[n] for n in COCO17], dtype=torch.long)


def solve_camera(points3d, target2d, weights):
    """Closed form for weak perspective: scale and a 2D translation, exactly.

    Minimising sum w * ||s * xy + t - target||^2 over (s, tx, ty) is linear least squares,
    so it is solved rather than descended on. This is the same reasoning `fit_one` gives for
    starting from Umeyama: the global placement is not what a line search should spend
    itself on.
    """
    xy = points3d[:, :2]
    w = weights[:, None]
    xm = (w * xy).sum(0) / w.sum()
    tm = (w * target2d).sum(0) / w.sum()
    xc, tc = xy - xm, target2d - tm
    scale = (w * xc * tc).sum() / (w * xc * xc).sum().clamp_min(1e-12)
    return scale, tm - scale * xm


def fit_2d(model, regressor, target2d, confidence=None, iters=120, dtype=torch.float64,
           prior=1e-5):
    """Solve pose parameters whose projected COCO-17 keypoints match `target2d`.

    `target2d` is (17, 2) in pixels, in the COCO-17 order above. `confidence` is (17,) and
    weights each point; a keypoint the detector was unsure of should not drag a limb.

    Returns a dict carrying the pose, the fitted 2D keypoints, per-keypoint pixel residuals,
    the fitted stature in the same pixels, and the camera. Residual and stature share units,
    so the referee's fraction-of-stature thresholds apply unchanged.
    """
    target2d = torch.as_tensor(target2d, dtype=dtype)
    if target2d.shape != (17, 2):
        raise ValueError(f"target2d is {tuple(target2d.shape)}, and must be (17, 2)")
    weights = (torch.ones(17, dtype=dtype) if confidence is None
               else torch.as_tensor(confidence, dtype=dtype).clamp_min(0.0))
    if float(weights.sum()) <= 0:
        raise ValueError("every keypoint has zero confidence, so there is nothing to fit")

    n = model.bone_count
    names = keypoint_names(regressor)
    idx = subset_17(names)

    rotvec = torch.zeros(n, 3, dtype=dtype, requires_grad=True)

    def keypoints3d():
        pose = torch.eye(4, dtype=dtype)[None, None].repeat(1, n, 1, 1).clone()
        import roma

        pose[0, :, :3, :3] = roma.rotvec_to_rotmat(rotvec)
        out = model(pose_parameters=pose)
        return regressor(out)[0].to(dtype)[idx]

    with torch.no_grad():
        scale0, t0 = solve_camera(keypoints3d(), target2d, weights)
    log_scale = torch.tensor([float(torch.log(scale0.clamp_min(1e-9)))], dtype=dtype,
                             requires_grad=True)
    trans = t0.clone().detach().requires_grad_(True)

    def projected():
        return keypoints3d()[:, :2] * torch.exp(log_scale) + trans

    opt = torch.optim.LBFGS([rotvec, log_scale, trans], max_iter=iters,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-14)

    def closure():
        opt.zero_grad()
        # The rest prior is `fit_one`'s, for its reason: most of the 104 bones move no
        # keypoint at all and would otherwise be free to fold anywhere.
        loss = (weights[:, None] * (projected() - target2d) ** 2).sum() + prior * (rotvec ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        import roma

        pose = torch.eye(4, dtype=dtype)[None, None].repeat(1, n, 1, 1).clone()
        pose[0, :, :3, :3] = roma.rotvec_to_rotmat(rotvec)
        out = model(pose_parameters=pose)
        fitted = projected()
        residual = (fitted - target2d).norm(dim=1)
        verts = out["vertices"][0].to(dtype)
        stature = float((verts[:, 1].max() - verts[:, 1].min()) * torch.exp(log_scale))

    return {
        "pose": pose.detach(),
        "rotvec": rotvec.detach(),
        "keypoints2d": fitted,
        "residual_px": residual,
        "stature_px": stature,
        "camera": {"scale": float(torch.exp(log_scale)), "translation": trans.detach().tolist()},
    }


def residuals_by_region(residual_px):
    """The two regions a 17-point detector can fill, named rather than assumed.

    Face and both hands stay absent, so the referee returns NOT_RUN. That is the correct
    answer for a 17-point fit and the reason a wholebody head is wanted, not a defect in
    this file.
    """
    ankle = [COCO17.index(n) for n in COCO17 if n.endswith("ankle")]
    body = [i for i in range(17) if i not in ankle]
    return {
        "body": residual_px[body].numpy(),
        "feet": residual_px[ankle].numpy(),
    }


def detect_keypoints(image_path, threshold: float = 0.4):
    """RF-DETR's 17 keypoints for one image, or a refusal naming what is absent.

    THIS IS THE BLOCKER NOW, AND IT IS SMALLER THAN THE ONE IT REPLACED. The fit above is
    measured and works; what has no home is the detector. `rf-detr-mcp` documents
    `uv pip install rfdetr` into a local virtual environment, that environment is not in the
    checkout, and no `pixi` environment in the corpus repository declares `rfdetr` either.
    So the loop can fit any 17 points it is given and cannot yet read them off a photograph.

    Passing keypoints in from anywhere else -- a projection of an authored pose, a hand
    annotation, another detector -- needs none of this.
    """
    from weft_loop import PreconditionFailed

    try:
        from rfdetr import RFDETRKeypointPreview
    except ImportError as error:
        raise PreconditionFailed(
            "no environment here carries rfdetr: rf-detr-mcp documents a local virtual "
            "environment that is not in the checkout, and no pixi environment in the corpus "
            "repository declares it. The fit does not need this; reading keypoints off a "
            "photograph does."
        ) from error

    from PIL import Image

    detection = RFDETRKeypointPreview().predict(Image.open(image_path), threshold=threshold)
    if len(detection) == 0:
        raise PreconditionFailed(f"no person detected in {image_path} at threshold {threshold}")
    return np.asarray(detection.xy[0]), np.asarray(detection.keypoint_confidence[0])
