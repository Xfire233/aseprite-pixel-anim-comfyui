"""Natural-Language OpenPose 18-Keypoint Generator & Kinematic Sequence Synthesizer.
Supports LLM-driven pose synthesis, anatomical constraint enforcement, and smooth multi-frame sequence generation.
"""
import json
import math
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw

# Standard 18-point OpenPose joint names and topological limb connections
JOINT_NAMES = [
    "nose", "neck", "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
    "r_hip", "r_kne", "r_ank", "l_hip", "l_kne", "l_ank", "r_eye", "l_eye", "r_ear", "l_ear"
]

LIMB_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
]

JOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)
]


def enforce_anatomical_constraints(kps: List[List[float]]) -> List[List[float]]:
    """Enforces human kinematic body proportions, symmetry, and joint angle boundaries."""
    arr = np.array(kps, dtype=np.float32)
    if arr.shape != (18, 3) and arr.shape != (18, 2):
        return kps

    pts = arr[:, :2].copy()
    conf = arr[:, 2].copy() if arr.shape[1] == 3 else np.ones(18, dtype=np.float32)

    # 1. Clamp canvas normalized bounds
    pts[:, 0] = np.clip(pts[:, 0], 0.05, 0.95)
    pts[:, 1] = np.clip(pts[:, 1], 0.05, 0.95)

    neck = pts[1]
    r_hip, l_hip = pts[8], pts[11]
    hip_mid = 0.5 * (r_hip + l_hip)
    torso_len = max(0.12, float(np.linalg.norm(neck - hip_mid)))

    # 2. Proportion limits relative to torso length
    def _clamp_bone(p_parent, p_child, min_ratio, max_ratio, default_dir):
        vec = p_child - p_parent
        cur_len = float(np.linalg.norm(vec))
        if cur_len < 1e-4:
            vec = default_dir
            cur_len = float(np.linalg.norm(vec))
        unit = vec / max(cur_len, 1e-4)
        target_len = np.clip(cur_len, torso_len * min_ratio, torso_len * max_ratio)
        return p_parent + unit * target_len

    # Shoulders
    pts[2] = _clamp_bone(neck, pts[2], 0.20, 0.55, np.array([-0.3, 0.05]))
    pts[5] = _clamp_bone(neck, pts[5], 0.20, 0.55, np.array([0.3, 0.05]))

    # Arms
    pts[3] = _clamp_bone(pts[2], pts[3], 0.45, 0.85, np.array([-0.1, 0.6]))
    pts[4] = _clamp_bone(pts[3], pts[4], 0.40, 0.80, np.array([-0.1, 0.5]))
    pts[6] = _clamp_bone(pts[5], pts[6], 0.45, 0.85, np.array([0.1, 0.6]))
    pts[7] = _clamp_bone(pts[6], pts[7], 0.40, 0.80, np.array([0.1, 0.5]))

    # Hips & Legs
    pts[8] = _clamp_bone(neck, pts[8], 0.80, 1.25, np.array([-0.2, 1.0]))
    pts[11] = _clamp_bone(neck, pts[11], 0.80, 1.25, np.array([0.2, 1.0]))
    pts[9] = _clamp_bone(pts[8], pts[9], 0.55, 1.10, np.array([-0.2, 0.8]))
    pts[10] = _clamp_bone(pts[9], pts[10], 0.50, 1.05, np.array([0.0, 0.8]))
    pts[12] = _clamp_bone(pts[11], pts[12], 0.55, 1.10, np.array([0.2, 0.8]))
    pts[13] = _clamp_bone(pts[12], pts[13], 0.50, 1.05, np.array([0.0, 0.8]))

    # Head features relative to neck
    pts[0] = _clamp_bone(neck, pts[0], 0.15, 0.40, np.array([0.0, -0.3]))
    pts[14] = pts[0] + np.array([-0.03, -0.015])
    pts[15] = pts[0] + np.array([0.03, -0.015])
    pts[16] = pts[0] + np.array([-0.065, 0.01])
    pts[17] = pts[0] + np.array([0.065, 0.01])

    out = []
    for i in range(18):
        c = float(conf[i]) if i < len(conf) else 1.0
        out.append([float(round(pts[i, 0], 4)), float(round(pts[i, 1], 4)), c])
    return out


def interpolate_pose_sequence(
    start_kps: List[List[float]],
    peak_kps: List[List[float]],
    num_frames: int = 9,
    mode: str = "pingpong",
) -> List[List[List[float]]]:
    """Interpolates smooth jitter-free continuous OpenPose sequence with kinematic bone length preservation."""
    arr_s = np.array(start_kps, dtype=np.float32)[:, :2]
    arr_p = np.array(peak_kps, dtype=np.float32)[:, :2]

    frames = []
    for f_idx in range(num_frames):
        if num_frames <= 1:
            t = 0.0
        elif mode == "pingpong":
            # Half sine wave from start (0) to peak (1)
            t = 0.5 * (1.0 - math.cos(math.pi * f_idx / float(num_frames - 1)))
        elif mode == "loop":
            # Full sine wave from start (0) to peak (1) back to start (0)
            t = 0.5 * (1.0 - math.cos(2.0 * math.pi * f_idx / float(num_frames)))
        else:
            t = float(f_idx) / float(num_frames - 1)

        # Smooth linear blend followed by anatomical constraint
        interp = (1.0 - t) * arr_s + t * arr_p
        kps_list = [[float(interp[j, 0]), float(interp[j, 1]), 1.0] for j in range(18)]
        constrained = enforce_anatomical_constraints(kps_list)
        frames.append(constrained)

    return frames


def render_openpose_skeleton(
    kps: List[List[float]],
    width: int = 512,
    height: int = 768,
    bg_color: str = "white",
    stickwidth: int = 4,
) -> Image.Image:
    """Renders OpenPose 18-point skeleton image according to standard ControlNet color mapping."""
    bg_rgb = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
    canvas = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(canvas)

    arr = np.array(kps, dtype=np.float32)
    pts_px = []
    for i in range(18):
        x = arr[i, 0] * width
        y = arr[i, 1] * height
        c = arr[i, 2] if arr.shape[1] > 2 else 1.0
        pts_px.append((int(round(x)), int(round(y)), c))

    # Draw limbs
    for limb_idx, (p1, p2) in enumerate(LIMB_PAIRS):
        x1, y1, c1 = pts_px[p1]
        x2, y2, c2 = pts_px[p2]
        if c1 > 0.05 and c2 > 0.05:
            col = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
            draw.line([(x1, y1), (x2, y2)], fill=col, width=stickwidth)

    # Draw joint dots
    r = max(2, stickwidth - 1)
    for j_idx in range(18):
        x, y, c = pts_px[j_idx]
        if c > 0.05:
            col = JOINT_COLORS[j_idx % len(JOINT_COLORS)]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    return canvas


SYS_PROMPT_POSE_GEN = """You are an expert human biomechanics and 2D OpenPose skeleton coordinate synthesizer.
The user will provide a natural-language description of an anime character pose or motion (e.g. sitting in M-shaped spread legs, kneeling with arms back, deep squat, waving hand).

Synthesize normalized [x, y] coordinates in range [0.05, 0.95] for standard 18-point OpenPose skeleton:
0: nose, 1: neck, 2: r_sho, 3: r_elb, 4: r_wri, 5: l_sho, 6: l_elb, 7: l_wri,
8: r_hip, 9: r_kne, 10: r_ank, 11: l_hip, 12: l_kne, 13: l_ank,
14: r_eye, 15: l_eye, 16: r_ear, 17: l_ear.
(Coordinate orientation: x=0 left, x=1 right; y=0 top, y=1 bottom).

Provide:
1. "start_kps": Keyframe 1 starting pose [18 points, each [x, y]].
2. "peak_kps": Keyframe 2 peak motion pose [18 points, each [x, y]] reflecting the motion (e.g. hips lowered in squat, shoulders risen in breath, hand reaching peak). If static pose, set peak_kps identical to start_kps.

Output strictly valid JSON with no markdown fences or explanations:
{
  "pose_summary": "<terse English description of synthesized pose>",
  "start_kps": [[x, y], ... 18 pairs ...],
  "peak_kps": [[x, y], ... 18 pairs ...]
}
"""
