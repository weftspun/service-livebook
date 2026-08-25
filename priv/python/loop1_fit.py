"""Fit an ANNY pose to 2D keypoints, for the fullbody vocabulary.

WHAT WAS MISSING WAS THE FRONT END, NOT THE SOLVER.
`4-entities/anny-pose-retarget-work/fit_joints_to_anny.py` solves a full 104-bone pose from
21 *three-dimensional* joint positions, by Umeyama for the global placement and LBFGS with a
strong-Wolfe line search for the rotations. This is the same solve driven from pixels.

THE VOCABULARY IS THE CALLER'S, AND THE EARLIER VERSION HARDCODED THE WRONG ONE. It defined
COCO-17 and subset every target to it. `todo.md` names loop one's detector as
"RFDetr Fullbody Coco Keypoints", and ANNY's own `bone_labels` carries exactly 104 names, so
a fullbody target has a home and a 17-point subset was throwing 87 of them away. Names now
arrive with the targets and nothing here asserts a length.

TWO FORWARDS, CHOSEN BY WHICH VOCABULARY THE NAMES BELONG TO.

* Bone joints, `bone_poses[0, :, :3, 3]`, indexed by `bone_labels`. This is the fullbody
  path and the one `fit_joints_to_anny.py` uses.
* `anny.keypoints.KeypointsRegressor`, a per-vertex convex blend to 23 named keypoints, for
  a target expressed in those 23. `coco.pth` ships with the package.

Both are differentiable end to end, so composing a camera makes the loss a reprojection
distance and no inverse model is needed or licensed.

THE CAMERA IS WEAK PERSPECTIVE, and that is the honest choice rather than the convenient
one. A focal length is not recoverable from one uncalibrated photograph, and solving for it
alongside depth trades the two against each other. Its three parameters are linear least
squares, so `solve_camera` computes them rather than descending on them.

WHAT THIS DOES NOT DO. It does not recover depth. Two poses whose keypoints project to the
same pixels are indistinguishable to this loss, the classic pair being a limb toward the
camera against the same limb away from it. `pose-consensus/python/depth_term.py` exists for
that and is not wired in here. A fit from this file is a 2D-consistent pose, which is a
smaller claim than a correct one, and the referee says whether a body could hold it.
"""
from __future__ import annotations

import numpy as np
import torch

# The referee judges five regions. A fullbody vocabulary can fill all five; a 17-point one
# fills two. Which region a point belongs to is decided by its name, so the mapping is data
# rather than a chain of conditionals somebody has to keep in step with the rig.
HAND_MARKERS = ("finger", "thumb", "wrist", "hand", "metacarpal")
FOOT_MARKERS = ("toe", "foot", "ankle", "heel")
FACE_MARKERS = ("head", "eye", "ear", "nose", "jaw", "tongue", "lip", "brow", "orbicularis",
                "temporalis", "levator", "risorius", "mentalis", "buccinator", "zygomatic",
                "depressor", "special", "oris", "nasalis", "procerus")


def build_model(topology: str = "makehuman"):
    """The makehuman topology, because `coco.pth` is indexed against its vertex count."""
    import anny
    from anny.models.model_data import TopologyConfig

    return anny.Anny(
        topology=TopologyConfig(base_mesh=topology, remove_unattached_vertices=False)
    )


def bone_names(model) -> list[str]:
    """ANNY's own 104, asserted against the count rather than assumed."""
    labels = list(model.bone_labels)
    if len(labels) != model.bone_count:
        raise ValueError(f"{len(labels)} labels against {model.bone_count} bones")
    return labels


def regressor_names(regressor, fallback=None) -> list[str]:
    names = getattr(regressor, "labels", None) or getattr(regressor, "names", None)
    return list(names) if names else list(fallback or [])


def subset(available, wanted) -> torch.Tensor:
    """Indices of `wanted` inside `available`, raising rather than shortening.

    Taking a subset by name is the only version of this that cannot silently misalign. The
    failure it prevents is an array of one length reaching a consumer of another and being
    truncated at the tail, which is where the extremities are.
    """
    index = {n: i for i, n in enumerate(available)}
    missing = [n for n in wanted if n not in index]
    if missing:
        tail = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        raise KeyError(f"the forward has no keypoint named {missing[:6]}{tail}")
    return torch.tensor([index[n] for n in wanted], dtype=torch.long)


def region_of(name: str) -> str:
    """Which referee region a keypoint name belongs to. Body is the default, not a guess."""
    lower = name.lower()
    if any(marker in lower for marker in HAND_MARKERS):
        if lower.endswith(".l") or "left" in lower:
            return "left_hand"
        if lower.endswith(".r") or "right" in lower:
            return "right_hand"
        return "body"
    if any(marker in lower for marker in FOOT_MARKERS):
        return "feet"
    if any(marker in lower for marker in FACE_MARKERS):
        return "face"
    return "body"


def solve_camera(points3d, target2d, weights):
    """Closed form for weak perspective: one scale and a 2D translation, exactly.

    Minimising sum w * ||s * xy + t - target||^2 over (s, tx, ty) is linear least squares.
    This is `fit_one`'s reasoning for starting from Umeyama: the global placement is not what
    a line search should spend itself on.
    """
    xy = points3d[:, :2]
    w = weights[:, None]
    xm = (w * xy).sum(0) / w.sum()
    tm = (w * target2d).sum(0) / w.sum()
    xc, tc = xy - xm, target2d - tm
    scale = (w * xc * tc).sum() / (w * xc * xc).sum().clamp_min(1e-12)
    return scale, tm - scale * xm


def fit_2d(model, target2d, names, confidence=None, regressor=None, iters=120,
           dtype=torch.float64, prior=1e-5):
    """Solve pose parameters whose projected keypoints match `target2d`.

    `names` is the vocabulary of the targets, in their row order: ANNY bone labels for a
    fullbody target, or the regressor's own names when `regressor` is given. `target2d` is
    (N, 2) in pixels and `confidence` is (N,), weighting each point so that one the detector
    was unsure of does not drag a limb.

    Returns the pose, the fitted 2D keypoints, per-keypoint pixel residuals, the fitted
    stature in those same pixels, and the camera. Residual and stature share units, so the
    referee's fraction-of-stature thresholds apply unchanged.
    """
    import roma

    target2d = torch.as_tensor(target2d, dtype=dtype)
    if target2d.ndim != 2 or target2d.shape[1] != 2:
        raise ValueError(f"target2d is {tuple(target2d.shape)}, and must be (N, 2)")
    if len(names) != target2d.shape[0]:
        raise ValueError(f"{len(names)} names against {target2d.shape[0]} target rows")
    weights = (torch.ones(len(names), dtype=dtype) if confidence is None
               else torch.as_tensor(confidence, dtype=dtype).clamp_min(0.0))
    if float(weights.sum()) <= 0:
        raise ValueError("every keypoint has zero confidence, so there is nothing to fit")

    n = model.bone_count
    available = bone_names(model) if regressor is None else regressor_names(regressor)
    idx = subset(available, list(names))

    rotvec = torch.zeros(n, 3, dtype=dtype, requires_grad=True)

    def keypoints3d():
        pose = torch.eye(4, dtype=dtype)[None, None].repeat(1, n, 1, 1).clone()
        pose[0, :, :3, :3] = roma.rotvec_to_rotmat(rotvec)
        out = model(pose_parameters=pose)
        points = out["bone_poses"][0, :, :3, 3] if regressor is None else regressor(out)[0]
        return points.to(dtype)[idx]

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
        # The rest prior is `fit_one`'s, for its reason: a bone that moves no target point is
        # otherwise free to fold anywhere.
        loss = (weights[:, None] * (projected() - target2d) ** 2).sum() + prior * (rotvec ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
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
        "names": list(names),
        "keypoints2d": fitted,
        "residual_px": residual,
        "stature_px": stature,
        "camera": {"scale": float(torch.exp(log_scale)), "translation": trans.detach().tolist()},
    }


def residuals_by_region(names, residual_px) -> dict:
    """Group residuals into the referee's regions, filling only what the names support.

    A region with no keypoints is absent rather than empty, because the referee treats a
    missing region as NOT_RUN and an empty array would be a measurement of nothing. Which
    regions a vocabulary can fill is a property of the vocabulary: the fullbody 104 reach the
    face and both hands, and a 17-point set never does.
    """
    grouped: dict[str, list[float]] = {}
    for name, value in zip(names, np.asarray(residual_px)):
        grouped.setdefault(region_of(name), []).append(float(value))
    return {region: np.asarray(values) for region, values in grouped.items()}


def detect_keypoints(image_path, threshold: float = 0.4):
    """The fullbody detector's keypoints for one image, or a refusal naming what is absent.

    Nothing checked out here carries `rfdetr`: rf-detr-mcp documents a local virtual
    environment that is not in the checkout, and no pixi environment in the corpus repository
    declares it. The fit needs none of this. Points from anywhere else -- a projection of an
    authored pose, a hand annotation, another detector -- go straight into `fit_2d`.
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
