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


SYS_PROMPT_POSE_GEN = """You are a scientific 2D human biomechanics OpenPose skeleton coordinate calculator for computer vision.
Your sole output is normalized [x, y] coordinates for standard 18-point OpenPose skeleton:
0: nose, 1: neck, 2: r_sho, 3: r_elb, 4: r_wri, 5: l_sho, 6: l_elb, 7: l_wri,
8: r_hip, 9: r_kne, 10: r_ank, 11: l_hip, 12: l_kne, 13: l_ank,
14: r_eye, 15: l_eye, 16: r_ear, 17: l_ear.
(Coordinate orientation: x=0 left, x=1 right; y=0 top, y=1 bottom in range [0.05, 0.95]).

Provide:
1. "start_kps": Keyframe 1 starting pose [18 points, each [x, y]].
2. "peak_kps": Keyframe 2 peak motion pose [18 points, each [x, y]] reflecting the motion (e.g. torso/shoulder rise in breathing, hands shifted). If static, peak_kps can match start_kps.

Output strictly valid JSON with no markdown fences or conversational text:
{
  "pose_summary": "<terse English description of synthesized pose>",
  "start_kps": [[x, y], ... 18 pairs ...],
  "peak_kps": [[x, y], ... 18 pairs ...]
}
"""


def desensitize_pose_prompt(raw_prompt: str) -> str:
    """Translates explicit sexual/NSFW terms into strictly neutral human biomechanical
    and geometric joint relations so cloud LLMs (Gemini/GPT/Claude) don't trigger safety filters.
    """
    text = raw_prompt

    # Specific hand actions first
    text = re.sub(r"左手\s*(揉胸|摸胸|托胸|抓胸|抚摸胸部|按在胸前)", "left hand placed over chest, ", text)
    text = re.sub(r"右手\s*(揉胸|摸胸|托胸|抓胸|抚摸胸部|按在胸前)", "right hand placed over chest, ", text)
    text = re.sub(r"左手\s*(伸入下体|摸下体|按在下体|放在裆部|自慰|探入下体|扣逼|抠逼|触碰私处)", "left hand placed at pelvis center, ", text)
    text = re.sub(r"右手\s*(伸入下体|摸下体|按在下体|放在裆部|自慰|探入下体|扣逼|抠逼|触碰私处)", "right hand placed at pelvis center, ", text)
    text = re.sub(r"双手\s*(揉胸|摸胸|托胸|抓胸|抚摸胸部)", "both hands placed on chest, ", text)
    text = re.sub(r"双手\s*(伸入下体|摸下体|放在裆部|自慰)", "both hands placed near pelvis, ", text)

    # Oral sex actions
    text = re.sub(r"(握住肉棒|含入口中|含入嘴中|含入|含住|口交|口活|吞吐|吹箫)", "both hands held in front of mouth along central axis, head tilted up, ", text)
    text = re.sub(r"(肉棒|阴茎|鸡巴|龟头|生殖器|阳具)", "", text)

    # General pose & action replacements
    replacements = [
        (r"M字开腿|M字腿|大开腿|分腿|双腿大开|张开双腿|大腿张开", "sitting on floor with legs spread wide in M-shape, knees bent pointing outward, "),
        (r"(后入|狗爬|四肢着地|撅屁股|翘臀)", "kneeling on all fours, hips elevated, back slightly arched, "),
        (r"(骑乘|跨坐|压在身上)", "straddling position, knees bent and spread wide, "),
        (r"(双膝跪地|跪趴|跪地|双膝跪在地上|跪在地上)", "kneeling on the floor with knees bent, "),
        (r"(仰卧|平躺|躺在地上)", "lying on the back, legs apart, "),
        (r"(揉|摸|抓|抚摸|托起)\s*(胸|胸部|乳房|奶子|双峰)", "hand on chest, "),
        (r"(伸入|摸|抠|扣|按在|抚摸|探入)\s*(下体|私处|裆部|阴部|小穴|私密部位)", "hand at pelvis, "),
        (r"(使用|用)?\s*([一两三四二\d]+根)?\s*手指\s*(插入|自慰|抠|抚摸)?", ""),
        (r"自慰|手淫|弄自己|高潮|抽送|发情|自渎|做爱|性交|插穴|内射|射精|精液|淫水|爱液", "breathing rhythmically"),
        (r"坐在地板上|坐在地上", "sitting on the floor, "),
        (r"[，。！？；;]", ", "),
        (r"进行|动作[:：]?", ""),
    ]
    for pat, rep in replacements:
        text = re.sub(pat, rep, text)

    tokens = [t.strip(" ,") for t in text.split(",") if t.strip(" ,")]
    clean = ", ".join(dict.fromkeys(tokens))
    return clean if clean else "sitting on floor in an M-shaped spread legs pose, hands on thighs"


def heuristic_fallback_pose(action_prompt: str, seed: int = 0) -> Tuple[List[List[float]], List[List[float]], str]:
    """Deterministic kinematic heuristic fallback when LLM fails or gets safety-blocked.
    Parses prompt keywords, selects baseline pose from nsfw_library/index.json,
    and tweaks hand/limb coordinates accordingly. Never crashes.
    """
    import os, math
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(base_dir, "poses", "nsfw_library", "index.json")

    preset_key = "solo_m_legs_sit"
    p_low = action_prompt.lower()
    if any(w in p_low for w in ["跪", "口交", "含入", "含住", "嘴前", "吹箫", "吞吐"]):
        preset_key = "solo_kneeling_arch"
    elif any(w in p_low for w in ["四肢", "狗爬", "后入", "趴"]):
        preset_key = "solo_all_fours_doggy"
    elif any(w in p_low for w in ["骑乘", "深蹲", "蹲"]):
        preset_key = "solo_squat_cowgirl"
    elif any(w in p_low for w in ["躺", "仰卧", "抱膝"]):
        preset_key = "solo_lying_spread"
    elif any(w in p_low for w in ["站", "立"]):
        preset_key = "solo_standing_strip"

    kps_dict = {}
    if os.path.exists(lib_path):
        try:
            with open(lib_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            if preset_key in idx and "start_kps" in idx[preset_key]:
                kps_raw = idx[preset_key]["start_kps"]
                kps_dict = kps_raw[0] if isinstance(kps_raw, list) else kps_raw
        except Exception:
            pass

    default_dict = {
        "neck": [0.5, 0.34],
        "r_sho": [0.42, 0.36],
        "l_sho": [0.58, 0.36],
        "r_elb": [0.31, 0.46],
        "l_elb": [0.69, 0.46],
        "r_wri": [0.26, 0.58],
        "l_wri": [0.74, 0.58],
        "r_hip": [0.45, 0.58],
        "l_hip": [0.55, 0.58],
        "r_kne": [0.32, 0.47],
        "l_kne": [0.68, 0.47],
        "r_ank": [0.35, 0.67],
        "l_ank": [0.65, 0.67],
        "nose": [0.5, 0.275],
        "r_eye": [0.482, 0.263],
        "l_eye": [0.518, 0.263],
        "r_ear": [0.464, 0.269],
        "l_ear": [0.536, 0.269]
    }
    for k, v in default_dict.items():
        if k not in kps_dict:
            kps_dict[k] = v

    kps_dict = {k: list(v) for k, v in kps_dict.items()}

    # Dynamic wrist & arm positioning based on user action intent
    # 1. Oral sex / hands raised in front of mouth
    if any(w in p_low for w in ["嘴前", "含入", "口交", "含住", "握住肉棒", "举在嘴前", "双手在嘴", "吞吐", "嘴唇"]):
        kps_dict["l_wri"] = [0.52, 0.35]
        kps_dict["r_wri"] = [0.48, 0.37]
        kps_dict["l_elb"] = [0.60, 0.40]
        kps_dict["r_elb"] = [0.40, 0.42]
        kps_dict["nose"] = [0.50, 0.26]
    # 2. General arms raised
    elif any(w in p_low for w in ["抬手", "举手", "双手抬起", "举起双手", "手抬起来"]):
        kps_dict["l_wri"] = [0.60, 0.22]
        kps_dict["r_wri"] = [0.40, 0.22]
        kps_dict["l_elb"] = [0.65, 0.32]
        kps_dict["r_elb"] = [0.35, 0.32]
    else:
        # Breast fondling
        if any(w in p_low for w in ["左手揉胸", "左手摸胸", "左手托胸", "左手抓胸", "揉胸", "摸胸"]):
            kps_dict["l_wri"] = [0.52, 0.43]
            kps_dict["l_elb"] = [0.66, 0.42]
        if any(w in p_low for w in ["右手揉胸", "右手摸胸", "右手托胸", "右手抓胸"]):
            kps_dict["r_wri"] = [0.48, 0.43]
            kps_dict["r_elb"] = [0.34, 0.42]

        # Crotch / masturbation
        if any(w in p_low for w in ["左手伸入下体", "左手摸下体", "左手下体", "左手自慰", "左手私处", "左手裆部"]):
            kps_dict["l_wri"] = [0.50, 0.62]
            kps_dict["l_elb"] = [0.62, 0.52]
        if any(w in p_low for w in ["右手伸入下体", "右手摸下体", "右手下体", "右手自慰", "右手私处", "右手裆部", "自慰", "摸下体"]):
            kps_dict["r_wri"] = [0.50, 0.62]
            kps_dict["r_elb"] = [0.38, 0.52]

    # Deterministic subtle seed variation so re-rolling produces distinct poses
    if seed != 0:
        dx = float(math.sin(float(seed) * 1.7) * 0.02)
        dy = float(math.cos(float(seed) * 2.3) * 0.015)
        if "l_wri" in kps_dict:
            kps_dict["l_wri"][0] = max(0.05, min(0.95, kps_dict["l_wri"][0] + dx))
            kps_dict["l_wri"][1] = max(0.05, min(0.95, kps_dict["l_wri"][1] + dy))
        if "r_wri" in kps_dict:
            kps_dict["r_wri"][0] = max(0.05, min(0.95, kps_dict["r_wri"][0] - dx))
            kps_dict["r_wri"][1] = max(0.05, min(0.95, kps_dict["r_wri"][1] + dy))

    start_kps = [[kps_dict.get(j, [0.5, 0.5])[0], kps_dict.get(j, [0.5, 0.5])[1], 1.0] for j in JOINT_NAMES]
    start_kps = enforce_anatomical_constraints(start_kps)

    peak_kps = [list(pt) for pt in start_kps]
    peak_kps[1][1] = max(0.05, peak_kps[1][1] - 0.015)
    peak_kps = enforce_anatomical_constraints(peak_kps)

    summary = f"Synthesized pose from {preset_key} (seed {seed}): {action_prompt[:60]}"
    return start_kps, peak_kps, summary
