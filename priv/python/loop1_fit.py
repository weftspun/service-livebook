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

RETRACTION, MEASURED AFTER rfdetr WAS INSTALLED: THERE IS NO FULLBODY CHECKPOINT TO READ.
`RFDETRKeypointPreviewConfig` in rfdetr 1.9.4 carries `num_keypoints_per_class = [17]`, so
the published preview model produces COCO-17 and nothing wider. The paragraph above is right
that a subset throws away 87 points, and wrong that the detector supplies them. What follows
is that a detector target takes the 23-point regressor path, whose first 17 labels are
COCO-17, and that the referee then judges two regions and reports NOT_RUN for three. The
fullbody path stays for targets that are genuinely fullbody, such as the round trip's.

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

# The referee judges five regions. Which region a point belongs to is decided by its name,
# so the mapping is data rather than a chain of conditionals somebody has to keep in step
# with the rig.
#
# WRIST IS NOT A HAND, AND THAT IS A MEASUREMENT RATHER THAN A TASTE. It was in this tuple,
# and a COCO-17 target then filled `left_hand` and `right_hand` with one point each: the
# wrist. The referee would have judged a hand from a single boundary joint and called it a
# region, which is the pass a missing region exists to prevent. The fullbody vocabulary has
# `wrist.L` and `wrist.R` and 19 further points per hand, so it loses nothing by counting a
# wrist as body. `test_loop1_fit.py` drives a 17-point target and fails if a hand fills.
HAND_MARKERS = ("finger", "thumb", "hand", "metacarpal")
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


def image_axes(model, dtype=torch.float64):
    """(right, up) vertex-component indices for this rig, measured rather than assumed.

    THE FIT PROJECTED X AND Y AND THE RIG IS Z-UP, AND EVERY POSE IT PRODUCED WAS ROTATED.
    ANNY's rest mesh measures X 1.046, Y 0.434, Z 1.660, so the tall axis is Z and the axis
    the old code fed to image-vertical was the body's depth. Against an upright photograph
    the solver's only way to match was to rotate the whole figure about ninety degrees, and
    the ninety-six grid renders show exactly that. The known-answer round trip did not catch
    it because it built its targets through the same two axes, so it was wrong and
    self-consistent together.

    CLAUDE.md states the rule this breaks: conventions are data, and an up axis is parsed
    and never assumed. So it is read off the rest pose here. Up is the longest rest extent,
    right is the longest of the two that remain, and a rig that is wider than it is tall
    raises rather than quietly deciding a lying figure is upright.
    """
    with torch.no_grad():
        verts = model()["vertices"][0].to(dtype)
        extent = (verts.max(0).values - verts.min(0).values)
    order = torch.argsort(extent, descending=True)
    up, right = int(order[0]), int(order[1])
    if float(extent[up]) < 1.2 * float(extent[right]):
        raise ValueError(
            f"rest extents {[round(float(e), 3) for e in extent]} do not name an up axis: "
            "the longest is not clearly longer than the next, so this rig is not a standing body"
        )
    return right, up


def to_image(points3d, axes):
    """World points onto the image plane: right stays, up is negated.

    Image rows count downward and the rig's up axis counts upward, so the sign is not
    cosmetic. `solve_camera` fits one POSITIVE scale, which cannot absorb a flip, and a fit
    that has to absorb it will roll the body instead.
    """
    right, up = axes
    return torch.stack([points3d[:, right], -points3d[:, up]], dim=1)


def solve_camera(points3d, target2d, weights, axes):
    """Closed form for weak perspective: one scale and a 2D translation, exactly.

    Minimising sum w * ||s * xy + t - target||^2 over (s, tx, ty) is linear least squares.
    This is `fit_one`'s reasoning for starting from Umeyama: the global placement is not what
    a line search should spend itself on.
    """
    xy = to_image(points3d, axes)
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

    axes = image_axes(model, dtype)
    with torch.no_grad():
        scale0, t0 = solve_camera(keypoints3d(), target2d, weights, axes)
    log_scale = torch.tensor([float(torch.log(scale0.clamp_min(1e-9)))], dtype=dtype,
                             requires_grad=True)
    trans = t0.clone().detach().requires_grad_(True)

    def projected():
        return to_image(keypoints3d(), axes) * torch.exp(log_scale) + trans

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
        # Stature is the up axis, which is the whole point of measuring the axes. It read
        # component 1 here, the body's depth, so every "percent of stature" this file has
        # ever reported was normalised by 0.434 where it should have used 1.660.
        stature = float((verts[:, axes[1]].max() - verts[:, axes[1]].min())
                        * torch.exp(log_scale))

    return {
        "pose": pose.detach(),
        "rotvec": rotvec.detach(),
        "names": list(names),
        "keypoints2d": fitted,
        "residual_px": residual,
        "stature_px": stature,
        "axes": {"right": axes[0], "up": axes[1]},
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


def coco_regressor(model):
    """The 23-point regressor and the 17 names a COCO detector actually produces.

    THE PUBLISHED DETECTOR IS 17 POINTS, AND THE HEADER ABOVE IS RETRACTED ON THAT POINT.
    `RFDETRKeypointPreviewConfig` in rfdetr 1.9.4 carries `num_keypoints_per_class = [17]`,
    so "RFDetr Fullbody Coco Keypoints" names a model this package does not ship. The bone
    path stays for a target that is genuinely fullbody. A 17-row target belongs in this
    vocabulary instead, because `bone_labels` has no `nose` and no `left_eye` and matching
    17 detector rows against 104 bone names is what `fit_2d` refuses by design.

    The regressor's first 17 labels are COCO-17 in COCO's own row order, which is checked
    here rather than assumed: a reordered `coco.pth` would otherwise fit every point to the
    wrong joint and still report a small residual.
    """
    from anny.keypoints import KeypointsRegressor

    regressor = KeypointsRegressor.coco(model)
    names = list(regressor.labels[:17])
    if names != list(COCO17):
        raise ValueError(f"coco.pth's first 17 labels are {names}, and COCO's order is {list(COCO17)}")
    return regressor, names


# COCO's own keypoint order. Held here so `coco_regressor` can check the asset against it
# rather than trust it, and so a caller can name the vocabulary without loading the rig.
COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def write_posed_mesh(model, pose, path):
    """Write the mesh the fit implies, in the two arrays `render_view.py` reads.

    The renderer takes `verts` and `faces` from an npz, so the fit has to be turned into a
    mesh before anything can be rendered from it. Nothing did that, which is why the loop's
    `propose` read a file no cell wrote.

    The topology is the 19,158-vertex makehuman one, and it is asserted rather than assumed:
    the 13,718-vertex body submodel would write a file that renders and is a different rig.
    """
    with torch.no_grad():
        out = model(pose_parameters=pose)
    verts = out["vertices"][0].detach().cpu().numpy()
    faces = np.asarray(model.faces.detach().cpu().numpy(), dtype=np.int64)
    if verts.shape[0] != 19158:
        raise ValueError(f"{verts.shape[0]} vertices, and render_corpus.py needs 19158")
    np.savez(path, verts=verts.astype(np.float64), faces=faces)
    return path


def detect_keypoints(image_path, threshold: float = 0.4):
    """The detector's keypoints for one image, or a refusal naming what is absent.

    Returns points, confidences and the names of the rows. The names travel with the points
    because the row count decides the vocabulary, and a caller that assumes one silently
    fits the wrong joints.
    """
    from weft_loop import PreconditionFailed

    try:
        from rfdetr import RFDETRKeypointPreview
    except ImportError as error:
        raise PreconditionFailed(
            "no environment here carries rfdetr. The notebook's setup cell declares it, so "
            "run that cell; outside the notebook, install rfdetr in the environment that "
            "calls this. The fit does not need it; reading keypoints off a photograph does."
        ) from error

    from PIL import Image

    detection = RFDETRKeypointPreview().predict(Image.open(image_path), threshold=threshold)
    if len(detection) == 0:
        raise PreconditionFailed(f"no person detected in {image_path} at threshold {threshold}")
    xy = np.asarray(detection.xy[0])
    confidence = np.asarray(detection.keypoint_confidence[0])
    if xy.shape[0] != len(COCO17):
        raise PreconditionFailed(
            f"the detector returned {xy.shape[0]} points and this front end names {len(COCO17)}. "
            "Name the rows before fitting rather than trimming them."
        )
    return xy, confidence, list(COCO17)
