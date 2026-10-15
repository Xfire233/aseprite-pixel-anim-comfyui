#!/usr/bin/env python3
"""Builds the preset NSFW OpenPose skeleton & 9-frame motion trajectory library (single + two-person interactive).

Follows standard COCO-18 OpenPose color conventions on a solid black background, with the skeleton
occupying ~56% of the canvas height (centered with generous top/bottom margins) so Qwen-Image-Edit
and ControlNet never crop the character's head or feet.

Outputs:
  - poses/nsfw_library/index.json
  - poses/nsfw_library/<preset_id>_start.png      (512x768 Stage 1 start pose skeleton)
  - poses/nsfw_library/<preset_id>_peak.png       (512x768 Stage 1/2 peak displacement skeleton)
  - poses/nsfw_library/<preset_id>_seq/f_00..08.png (9-frame 384x576 half-cycle motion sequence)
  - poses/nsfw_library/catalog.png                (Visual contact sheet of all 10 presets)
"""
import json
import math
import os
import shutil
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY_INPUT, PROJECT_ROOT  # noqa: E402
LIB_DIR = os.path.join(PROJECT_ROOT, "poses", "nsfw_library")

# Standard COCO-18 OpenPose limb connections and colors (R, G, B)
# Keypoint names:
# nose, neck, r_sho, r_elb, r_wri, l_sho, l_elb, l_wri,
# r_hip, r_kne, r_ank, l_hip, l_kne, l_ank, r_eye, l_eye, r_ear, l_ear
LIMB_SPECS = [
    ("neck", "r_sho", (255, 0, 0)),
    ("neck", "l_sho", (255, 85, 0)),
    ("r_sho", "r_elb", (255, 170, 0)),
    ("r_elb", "r_wri", (255, 255, 0)),
    ("l_sho", "l_elb", (170, 255, 0)),
    ("l_elb", "l_wri", (85, 255, 0)),
    ("neck", "r_hip", (0, 255, 0)),
    ("r_hip", "r_kne", (0, 255, 85)),
    ("r_kne", "r_ank", (0, 255, 170)),
    ("neck", "l_hip", (0, 255, 255)),
    ("l_hip", "l_kne", (0, 170, 255)),
    ("l_kne", "l_ank", (0, 85, 255)),
    ("neck", "nose", (0, 0, 255)),
    ("nose", "r_eye", (85, 0, 255)),
    ("r_eye", "r_ear", (170, 0, 255)),
    ("nose", "l_eye", (255, 0, 255)),
    ("l_eye", "l_ear", (255, 0, 170)),
]

KP_COLORS = {
    "nose": (0, 0, 255),
    "neck": (255, 0, 0),
    "r_sho": (255, 85, 0),
    "r_elb": (255, 170, 0),
    "r_wri": (255, 255, 0),
    "l_sho": (170, 255, 0),
    "l_elb": (85, 255, 0),
    "l_wri": (0, 255, 0),
    "r_hip": (0, 255, 85),
    "r_kne": (0, 255, 170),
    "r_ank": (0, 255, 255),
    "l_hip": (0, 170, 255),
    "l_kne": (0, 85, 255),
    "l_ank": (0, 0, 255),
    "r_eye": (85, 0, 255),
    "l_eye": (255, 0, 255),
    "r_ear": (170, 0, 255),
    "l_ear": (255, 0, 85),
}


def add_head_kps(kps: dict, head_dx: float = 0.0, head_dy: float = -0.065, profile: str = "front") -> dict:
    """Given 'neck' and 'nose' (or derives 'nose' from 'neck'), populates eye and ear keypoints."""
    out = dict(kps)
    nx, ny = out["neck"]
    if "nose" not in out:
        out["nose"] = (nx + head_dx, ny + head_dy)
    nox, noy = out["nose"]
    if profile == "front":
        out.setdefault("r_eye", (nox - 0.018, noy - 0.012))
        out.setdefault("l_eye", (nox + 0.018, noy - 0.012))
        out.setdefault("r_ear", (nox - 0.036, noy - 0.006))
        out.setdefault("l_ear", (nox + 0.036, noy - 0.006))
    elif profile == "side_left":
        out.setdefault("l_eye", (nox - 0.010, noy - 0.012))
        out.setdefault("l_ear", (nox + 0.022, noy - 0.008))
    elif profile == "look_back":
        out.setdefault("r_eye", (nox - 0.015, noy - 0.012))
        out.setdefault("l_eye", (nox + 0.015, noy - 0.012))
        out.setdefault("r_ear", (nox - 0.030, noy - 0.006))
        out.setdefault("l_ear", (nox + 0.030, noy - 0.006))
    return out


def interpolate_kps(kps_a: dict, kps_b: dict, t: float) -> dict:
    out = {}
    for k in kps_a:
        if k in kps_b:
            ax, ay = kps_a[k]
            bx, by = kps_b[k]
            out[k] = (ax + (bx - ax) * t, ay + (by - ay) * t)
        else:
            out[k] = kps_a[k]
    return out


def render_skeleton(persons_kps: list[dict], width: int = 512, height: int = 768) -> Image.Image:
    im = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    line_w = max(4, int(round(min(width, height) * 0.012)))
    joint_r = max(4, int(round(min(width, height) * 0.011)))

    for kps in persons_kps:
        def _map_pt(pt):
            nx, ny = pt[0], 0.22 + (pt[1] - 0.16) * 0.78
            return int(round(nx * width)), int(round(ny * height))

        for k1, k2, col in LIMB_SPECS:
            if k1 in kps and k2 in kps:
                x1, y1 = _map_pt(kps[k1])
                x2, y2 = _map_pt(kps[k2])
                draw.line([(x1, y1), (x2, y2)], fill=col, width=line_w)
        for k, pt in kps.items():
            col = KP_COLORS.get(k, (255, 255, 255))
            cx, cy = _map_pt(pt)
            draw.ellipse([cx - joint_r, cy - joint_r, cx + joint_r, cy + joint_r], fill=col)
    return im


# Define all 10 NSFW pose & motion presets (normalized [0, 1] coordinates, centered in [0.20, 0.78] Y range)
PRESETS = [
    {
        "id": "solo_m_legs_sit",
        "label": "solo_m_legs_sit (单人·M字开腿坐地抚腿娇喘)",
        "category": "solo",
        "anima_pose_tags": "1girl, solo, sitting, sitting on floor, m legs, wide spread legs, knees up, arms at sides, hands on floor beside hips, full frontal, exposed pussy, breasts out, nipples, uncensored, shy, blush, parted lips, heavy breathing",
        "qwen_pose_prompt": "sitting on the floor in a wide M-shaped spread-legs pose with both knees bent and raised wide outward, both arms placed outward to the sides with hands resting on the floor beside her outer hips away from her body, completely bottomless and topless with exposed breasts, nipples, and unobstructed bare pussy clearly visible between her spread thighs",
        "wan_motion_prompt": "sitting on the floor in an M-shaped spread-legs pose on a plain white background, she breathes heavily and rhythmically as her chest and breasts rise and fall and her spread knees open and close",
        "start": [
            add_head_kps({
                "neck": (0.50, 0.34),
                "r_sho": (0.42, 0.36), "l_sho": (0.58, 0.36),
                "r_elb": (0.31, 0.46), "l_elb": (0.69, 0.46),
                "r_wri": (0.26, 0.58), "l_wri": (0.74, 0.58),
                "r_hip": (0.45, 0.58), "l_hip": (0.55, 0.58),
                "r_kne": (0.32, 0.47), "l_kne": (0.68, 0.47),
                "r_ank": (0.35, 0.67), "l_ank": (0.65, 0.67),
            })
        ],
        "peak": [
            add_head_kps({
                "neck": (0.50, 0.325),
                "r_sho": (0.41, 0.345), "l_sho": (0.59, 0.345),
                "r_elb": (0.29, 0.45), "l_elb": (0.71, 0.45),
                "r_wri": (0.25, 0.58), "l_wri": (0.75, 0.58),
                "r_hip": (0.45, 0.57), "l_hip": (0.55, 0.57),
                "r_kne": (0.26, 0.45), "l_kne": (0.74, 0.45),
                "r_ank": (0.33, 0.67), "l_ank": (0.67, 0.67),
            })
        ],
    },
    {
        "id": "solo_standing_strip",
        "label": "solo_standing_strip (单人·正面站立敞衣双手拉衣领展示)",
        "category": "solo",
        "anima_pose_tags": "1girl, solo, standing, full body, open clothes, undressing, pulling clothes, breasts out, nipples, bottomless, no panties, pussy, uncensored, legs slightly apart, blushing, looking at viewer",
        "qwen_pose_prompt": "standing facing forward with legs slightly apart, both hands holding open the edges of her slipped-off top/jacket to fully expose her bare breasts, nipples, flat stomach, and bare hairless pussy",
        "wan_motion_prompt": "standing facing forward on a plain white background holding her open garment with both hands, she gently sways her hips side to side and breathes softly with subtle breast bounce",
        "start": [
            add_head_kps({
                "neck": (0.50, 0.25),
                "r_sho": (0.42, 0.27), "l_sho": (0.58, 0.27),
                "r_elb": (0.35, 0.36), "l_elb": (0.65, 0.36),
                "r_wri": (0.40, 0.33), "l_wri": (0.60, 0.33),
                "r_hip": (0.45, 0.48), "l_hip": (0.55, 0.48),
                "r_kne": (0.44, 0.61), "l_kne": (0.56, 0.61),
                "r_ank": (0.43, 0.74), "l_ank": (0.57, 0.74),
            })
        ],
        "peak": [
            add_head_kps({
                "neck": (0.50, 0.245),
                "r_sho": (0.42, 0.265), "l_sho": (0.58, 0.265),
                "r_elb": (0.33, 0.35), "l_elb": (0.67, 0.35),
                "r_wri": (0.37, 0.33), "l_wri": (0.63, 0.33),
                "r_hip": (0.46, 0.48), "l_hip": (0.56, 0.48),
                "r_kne": (0.44, 0.61), "l_kne": (0.56, 0.61),
                "r_ank": (0.43, 0.74), "l_ank": (0.57, 0.74),
            })
        ],
    },
    {
        "id": "solo_kneeling_arch",
        "label": "solo_kneeling_arch (单人·双膝跪地挺胸背手展示)",
        "category": "solo",
        "anima_pose_tags": "1girl, solo, kneeling, knees apart, arms behind back, arched back, chest out, breasts out, nipples, bottomless, no panties, pussy, uncensored, looking at viewer, blush, seductive smile",
        "qwen_pose_prompt": "kneeling upright on the floor with her knees spread apart, arms held behind her back and chest pushed forward, completely topless and bottomless with exposed breasts, nipples, and bare pussy",
        "wan_motion_prompt": "kneeling on the floor with arms behind her back on a plain white background, she slowly arches her back and pushes her chest forward as her breasts gently bounce and her hips rock slightly",
        "start": [
            add_head_kps({
                "neck": (0.50, 0.29),
                "r_sho": (0.42, 0.31), "l_sho": (0.58, 0.31),
                "r_elb": (0.40, 0.41), "l_elb": (0.60, 0.41),
                "r_wri": (0.46, 0.47), "l_wri": (0.54, 0.47),
                "r_hip": (0.45, 0.52), "l_hip": (0.55, 0.52),
                "r_kne": (0.38, 0.68), "l_kne": (0.62, 0.68),
                "r_ank": (0.42, 0.72), "l_ank": (0.58, 0.72),
            })
        ],
        "peak": [
            add_head_kps({
                "neck": (0.50, 0.275),
                "r_sho": (0.41, 0.295), "l_sho": (0.59, 0.295),
                "r_elb": (0.40, 0.40), "l_elb": (0.60, 0.40),
                "r_wri": (0.46, 0.46), "l_wri": (0.54, 0.46),
                "r_hip": (0.45, 0.51), "l_hip": (0.55, 0.51),
                "r_kne": (0.37, 0.68), "l_kne": (0.63, 0.68),
                "r_ank": (0.42, 0.72), "l_ank": (0.58, 0.72),
            })
        ],
    },
    {
        "id": "solo_all_fours_doggy",
        "label": "solo_all_fours_doggy (单人·四肢着地翘臀回眸后入预备)",
        "category": "solo",
        "anima_pose_tags": "1girl, solo, all fours, doggy style position, arched back, ass, top-down bottom-up, looking back, looking at viewer, breasts out, hanging breasts, nipples, bottomless, no panties, pussy, uncensored, blush",
        "qwen_pose_prompt": "on all fours on the floor in a top-down bottom-up arched-back pose with her hips raised high and head turned looking back at the viewer, exposed hanging breasts, bare ass, and pussy",
        "wan_motion_prompt": "on all fours on a plain white background looking back at the viewer, she rhythmically sways and rocks her raised hips forward and backward while her hanging breasts gently swing",
        "start": [
            add_head_kps({
                "neck": (0.34, 0.46),
                "r_sho": (0.32, 0.48), "l_sho": (0.38, 0.47),
                "r_elb": (0.30, 0.59), "l_elb": (0.37, 0.58),
                "r_wri": (0.29, 0.69), "l_wri": (0.36, 0.68),
                "r_hip": (0.62, 0.42), "l_hip": (0.68, 0.43),
                "r_kne": (0.60, 0.60), "l_kne": (0.67, 0.61),
                "r_ank": (0.70, 0.69), "l_ank": (0.76, 0.70),
            }, head_dx=-0.02, head_dy=-0.06, profile="look_back")
        ],
        "peak": [
            add_head_kps({
                "neck": (0.32, 0.47),
                "r_sho": (0.30, 0.49), "l_sho": (0.36, 0.48),
                "r_elb": (0.29, 0.59), "l_elb": (0.36, 0.58),
                "r_wri": (0.29, 0.69), "l_wri": (0.36, 0.68),
                "r_hip": (0.65, 0.40), "l_hip": (0.71, 0.41),
                "r_kne": (0.60, 0.60), "l_kne": (0.67, 0.61),
                "r_ank": (0.70, 0.69), "l_ank": (0.76, 0.70),
            }, head_dx=-0.02, head_dy=-0.06, profile="look_back")
        ],
    },
    {
        "id": "solo_squat_cowgirl",
        "label": "solo_squat_cowgirl (单人·正面深蹲开腿骑乘上下起伏)",
        "category": "solo",
        "anima_pose_tags": "1girl, solo, squatting, wide spread legs, cowgirl position, arms spread outward, hands on outer knees, breasts out, nipples, bottomless, no panties, pussy, uncensored, full frontal, blush, open mouth, panting",
        "qwen_pose_prompt": "in a deep wide-legged squat facing forward with her knees spread wide apart, both arms angled wide outward with hands resting on the outer sides of her knees far away from her crotch, completely topless and bottomless with exposed breasts, nipples, and unobstructed bare pussy clearly visible",
        "wan_motion_prompt": "in a wide-legged deep squat facing forward on a plain white background, she rhythmically squats her hips up and down as her knees bend wide outward and her breasts bounce naturally",
        "start": [
            add_head_kps({
                "neck": (0.50, 0.28),
                "r_sho": (0.42, 0.30), "l_sho": (0.58, 0.30),
                "r_elb": (0.33, 0.40), "l_elb": (0.67, 0.40),
                "r_wri": (0.24, 0.49), "l_wri": (0.76, 0.49),
                "r_hip": (0.45, 0.52), "l_hip": (0.55, 0.52),
                "r_kne": (0.31, 0.55), "l_kne": (0.69, 0.55),
                "r_ank": (0.39, 0.72), "l_ank": (0.61, 0.72),
            })
        ],
        "peak": [
            add_head_kps({
                "neck": (0.50, 0.33),
                "r_sho": (0.42, 0.35), "l_sho": (0.58, 0.35),
                "r_elb": (0.30, 0.44), "l_elb": (0.70, 0.44),
                "r_wri": (0.19, 0.52), "l_wri": (0.81, 0.52),
                "r_hip": (0.45, 0.57), "l_hip": (0.55, 0.57),
                "r_kne": (0.24, 0.58), "l_kne": (0.76, 0.58),
                "r_ank": (0.39, 0.72), "l_ank": (0.61, 0.72),
            })
        ],
    },
    {
        "id": "solo_lying_spread",
        "label": "solo_lying_spread (单人·仰卧抬腿抱膝大开展示)",
        "category": "solo",
        "anima_pose_tags": "1girl, solo, lying, on back, legs up, spread legs, grabbing own knees, knees to chest, pussy, anus, uncensored, breasts out, nipples, bottomless, blush, heavy breathing",
        "qwen_pose_prompt": "lying on her back with both legs raised high and spread wide apart, hands holding the backs of her knees, completely topless and bottomless with exposed breasts, nipples, and exposed bare pussy",
        "wan_motion_prompt": "lying on her back with her legs raised and spread wide on a plain white background, she gently pulls her knees slightly closer and rocks her pelvis upward while breathing softly",
        "start": [
            add_head_kps({
                "neck": (0.50, 0.38),
                "r_sho": (0.41, 0.40), "l_sho": (0.59, 0.40),
                "r_elb": (0.34, 0.46), "l_elb": (0.66, 0.46),
                "r_wri": (0.32, 0.40), "l_wri": (0.68, 0.40),
                "r_hip": (0.44, 0.58), "l_hip": (0.56, 0.58),
                "r_kne": (0.30, 0.39), "l_kne": (0.70, 0.39),
                "r_ank": (0.25, 0.27), "l_ank": (0.75, 0.27),
            })
        ],
        "peak": [
            add_head_kps({
                "neck": (0.50, 0.38),
                "r_sho": (0.41, 0.40), "l_sho": (0.59, 0.40),
                "r_elb": (0.33, 0.45), "l_elb": (0.67, 0.45),
                "r_wri": (0.30, 0.37), "l_wri": (0.70, 0.37),
                "r_hip": (0.44, 0.56), "l_hip": (0.56, 0.56),
                "r_kne": (0.28, 0.36), "l_kne": (0.72, 0.36),
                "r_ank": (0.23, 0.24), "l_ank": (0.77, 0.24),
            })
        ],
    },
    # === Two-Person Interactive NSFW Presets (双人交互 NSFW 预设) ===
    {
        "id": "duo_doggy_from_behind",
        "label": "duo_doggy_from_behind (双人·后入式交合抽送律动)",
        "category": "duo",
        "anima_pose_tags": "1girl, 1boy, hetero, sex, doggy style, from behind, vaginal, all fours, arched back, grabbing hips, looking back, breasts out, hanging breasts, nipples, uncensored, blush, open mouth",
        "qwen_pose_prompt": "two characters in an explicit doggy-style sex pose from the side: the anime girl from image 1 is on all fours with her back arched and head turned looking back while a faceless male partner kneels behind her holding her hips",
        "wan_motion_prompt": "in a side-view doggy-style pose on a plain white background, the male partner behind her rhythmically thrusts his hips forward against hers as her body rocks forward and her breasts swing",
        "start": [
            # Person 1: Female on all fours (left/center)
            add_head_kps({
                "neck": (0.28, 0.45),
                "r_sho": (0.27, 0.47), "l_sho": (0.32, 0.46),
                "r_elb": (0.26, 0.58), "l_elb": (0.31, 0.57),
                "r_wri": (0.25, 0.69), "l_wri": (0.30, 0.68),
                "r_hip": (0.54, 0.43), "l_hip": (0.58, 0.44),
                "r_kne": (0.52, 0.62), "l_kne": (0.56, 0.63),
                "r_ank": (0.60, 0.70), "l_ank": (0.64, 0.71),
            }, head_dx=-0.02, head_dy=-0.06, profile="look_back"),
            # Person 2: Male kneeling behind (right)
            add_head_kps({
                "neck": (0.68, 0.27),
                "r_sho": (0.66, 0.29), "l_sho": (0.72, 0.29),
                "r_elb": (0.62, 0.38), "l_elb": (0.66, 0.39),
                "r_wri": (0.55, 0.43), "l_wri": (0.58, 0.44),
                "r_hip": (0.66, 0.47), "l_hip": (0.70, 0.47),
                "r_kne": (0.65, 0.65), "l_kne": (0.71, 0.66),
                "r_ank": (0.75, 0.70), "l_ank": (0.79, 0.71),
            }, head_dx=-0.03, head_dy=-0.06, profile="side_left"),
        ],
        "peak": [
            add_head_kps({
                "neck": (0.25, 0.46),
                "r_sho": (0.24, 0.48), "l_sho": (0.29, 0.47),
                "r_elb": (0.24, 0.58), "l_elb": (0.29, 0.57),
                "r_wri": (0.25, 0.69), "l_wri": (0.30, 0.68),
                "r_hip": (0.51, 0.43), "l_hip": (0.55, 0.44),
                "r_kne": (0.52, 0.62), "l_kne": (0.56, 0.63),
                "r_ank": (0.60, 0.70), "l_ank": (0.64, 0.71),
            }, head_dx=-0.02, head_dy=-0.06, profile="look_back"),
            add_head_kps({
                "neck": (0.64, 0.28),
                "r_sho": (0.62, 0.30), "l_sho": (0.68, 0.30),
                "r_elb": (0.58, 0.38), "l_elb": (0.62, 0.39),
                "r_wri": (0.52, 0.43), "l_wri": (0.55, 0.44),
                "r_hip": (0.59, 0.45), "l_hip": (0.63, 0.45),
                "r_kne": (0.65, 0.65), "l_kne": (0.71, 0.66),
                "r_ank": (0.75, 0.70), "l_ank": (0.79, 0.71),
            }, head_dx=-0.03, head_dy=-0.06, profile="side_left"),
        ],
    },
    {
        "id": "duo_cowgirl_ride",
        "label": "duo_cowgirl_ride (双人·女上位跨坐骑乘起伏)",
        "category": "duo",
        "anima_pose_tags": "1girl, 1boy, hetero, sex, cowgirl position, girl on top, straddling, vaginal, hands on chest, breasts out, nipples, uncensored, blush, panting",
        "qwen_pose_prompt": "two characters in a cowgirl sex position: a male partner lies on his back at the bottom while the anime girl from image 1 straddles his hips upright facing forward with exposed breasts and nipples",
        "wan_motion_prompt": "straddling her partner in a cowgirl position on a plain white background, she rhythmically rises and lowers her hips up and down while her breasts bounce naturally",
        "start": [
            # Female on top
            add_head_kps({
                "neck": (0.50, 0.27),
                "r_sho": (0.43, 0.29), "l_sho": (0.57, 0.29),
                "r_elb": (0.40, 0.39), "l_elb": (0.60, 0.39),
                "r_wri": (0.44, 0.48), "l_wri": (0.56, 0.48),
                "r_hip": (0.45, 0.49), "l_hip": (0.55, 0.49),
                "r_kne": (0.35, 0.61), "l_kne": (0.65, 0.61),
                "r_ank": (0.38, 0.70), "l_ank": (0.62, 0.70),
            }),
            # Male lying beneath
            add_head_kps({
                "neck": (0.50, 0.56),
                "r_sho": (0.40, 0.58), "l_sho": (0.60, 0.58),
                "r_elb": (0.35, 0.54), "l_elb": (0.65, 0.54),
                "r_wri": (0.43, 0.49), "l_wri": (0.57, 0.49),
                "r_hip": (0.45, 0.62), "l_hip": (0.55, 0.62),
                "r_kne": (0.42, 0.71), "l_kne": (0.58, 0.71),
                "r_ank": (0.40, 0.77), "l_ank": (0.60, 0.77),
            }, head_dy=-0.04),
        ],
        "peak": [
            add_head_kps({
                "neck": (0.50, 0.31),
                "r_sho": (0.43, 0.33), "l_sho": (0.57, 0.33),
                "r_elb": (0.40, 0.42), "l_elb": (0.60, 0.42),
                "r_wri": (0.44, 0.50), "l_wri": (0.56, 0.50),
                "r_hip": (0.45, 0.54), "l_hip": (0.55, 0.54),
                "r_kne": (0.34, 0.62), "l_kne": (0.66, 0.62),
                "r_ank": (0.38, 0.70), "l_ank": (0.62, 0.70),
            }),
            add_head_kps({
                "neck": (0.50, 0.56),
                "r_sho": (0.40, 0.58), "l_sho": (0.60, 0.58),
                "r_elb": (0.35, 0.54), "l_elb": (0.65, 0.54),
                "r_wri": (0.43, 0.53), "l_wri": (0.57, 0.53),
                "r_hip": (0.45, 0.61), "l_hip": (0.55, 0.61),
                "r_kne": (0.42, 0.71), "l_kne": (0.58, 0.71),
                "r_ank": (0.40, 0.77), "l_ank": (0.60, 0.77),
            }, head_dy=-0.04),
        ],
    },
    {
        "id": "duo_standing_from_behind",
        "label": "duo_standing_from_behind (双人·站立背后环抱揉胸顶胯)",
        "category": "duo",
        "anima_pose_tags": "1girl, 1boy, hetero, standing sex, hug from behind, breast grab, groping, breasts out, nipples, bottomless, pussy, uncensored, blush, arched back",
        "qwen_pose_prompt": "two characters standing together: the anime girl from image 1 stands in front facing forward with exposed breasts and bare lower body while a taller partner stands directly behind her reaching around to cup her breasts",
        "wan_motion_prompt": "standing on a plain white background as the partner behind her holds her chest and rhythmically presses his hips forward against hers, causing her body and breasts to gently rock upward",
        "start": [
            # Female in front
            add_head_kps({
                "neck": (0.48, 0.28),
                "r_sho": (0.41, 0.30), "l_sho": (0.55, 0.30),
                "r_elb": (0.37, 0.40), "l_elb": (0.59, 0.40),
                "r_wri": (0.39, 0.49), "l_wri": (0.57, 0.49),
                "r_hip": (0.43, 0.50), "l_hip": (0.53, 0.50),
                "r_kne": (0.42, 0.63), "l_kne": (0.54, 0.63),
                "r_ank": (0.41, 0.75), "l_ank": (0.55, 0.75),
            }),
            # Male behind
            add_head_kps({
                "neck": (0.53, 0.22),
                "r_sho": (0.44, 0.24), "l_sho": (0.62, 0.24),
                "r_elb": (0.38, 0.33), "l_elb": (0.64, 0.33),
                "r_wri": (0.44, 0.35), "l_wri": (0.52, 0.35),
                "r_hip": (0.47, 0.48), "l_hip": (0.58, 0.48),
                "r_kne": (0.46, 0.62), "l_kne": (0.59, 0.62),
                "r_ank": (0.46, 0.75), "l_ank": (0.60, 0.75),
            }),
        ],
        "peak": [
            add_head_kps({
                "neck": (0.47, 0.27),
                "r_sho": (0.40, 0.29), "l_sho": (0.54, 0.29),
                "r_elb": (0.36, 0.39), "l_elb": (0.58, 0.39),
                "r_wri": (0.38, 0.48), "l_wri": (0.56, 0.48),
                "r_hip": (0.42, 0.49), "l_hip": (0.52, 0.49),
                "r_kne": (0.41, 0.63), "l_kne": (0.53, 0.63),
                "r_ank": (0.41, 0.75), "l_ank": (0.55, 0.75),
            }),
            add_head_kps({
                "neck": (0.52, 0.22),
                "r_sho": (0.43, 0.24), "l_sho": (0.61, 0.24),
                "r_elb": (0.37, 0.33), "l_elb": (0.63, 0.33),
                "r_wri": (0.43, 0.34), "l_wri": (0.51, 0.34),
                "r_hip": (0.45, 0.48), "l_hip": (0.56, 0.48),
                "r_kne": (0.46, 0.62), "l_kne": (0.59, 0.62),
                "r_ank": (0.46, 0.75), "l_ank": (0.60, 0.75),
            }),
        ],
    },
    {
        "id": "duo_missionary_press",
        "label": "duo_missionary_press (双人·正常位仰卧抬腿交合律动)",
        "category": "duo",
        "anima_pose_tags": "1girl, 1boy, hetero, sex, missionary, lying on back, legs up, spread legs, vaginal, breasts out, nipples, uncensored, blush, panting",
        "qwen_pose_prompt": "two characters in a missionary sex pose: the anime girl from image 1 lies on her back with her legs raised and spread wide while her partner kneels between her legs leaning over her",
        "wan_motion_prompt": "in a missionary pose on a plain white background, the partner leans between her raised legs and rhythmically thrusts his hips downward and forward as her breasts and raised legs gently move in response",
        "start": [
            # Female lying on back with legs raised
            add_head_kps({
                "neck": (0.30, 0.56),
                "r_sho": (0.28, 0.58), "l_sho": (0.34, 0.58),
                "r_elb": (0.26, 0.65), "l_elb": (0.36, 0.65),
                "r_wri": (0.32, 0.68), "l_wri": (0.42, 0.68),
                "r_hip": (0.50, 0.62), "l_hip": (0.55, 0.62),
                "r_kne": (0.44, 0.46), "l_kne": (0.52, 0.45),
                "r_ank": (0.38, 0.34), "l_ank": (0.48, 0.33),
            }, head_dx=-0.05, head_dy=-0.02),
            # Male kneeling between legs
            add_head_kps({
                "neck": (0.64, 0.32),
                "r_sho": (0.62, 0.34), "l_sho": (0.68, 0.34),
                "r_elb": (0.54, 0.43), "l_elb": (0.60, 0.44),
                "r_wri": (0.46, 0.46), "l_wri": (0.52, 0.45),
                "r_hip": (0.62, 0.56), "l_hip": (0.67, 0.56),
                "r_kne": (0.60, 0.70), "l_kne": (0.68, 0.70),
                "r_ank": (0.72, 0.72), "l_ank": (0.78, 0.72),
            }, head_dx=-0.04, head_dy=-0.05, profile="side_left"),
        ],
        "peak": [
            add_head_kps({
                "neck": (0.29, 0.56),
                "r_sho": (0.27, 0.58), "l_sho": (0.33, 0.58),
                "r_elb": (0.25, 0.65), "l_elb": (0.35, 0.65),
                "r_wri": (0.31, 0.68), "l_wri": (0.41, 0.68),
                "r_hip": (0.49, 0.61), "l_hip": (0.54, 0.61),
                "r_kne": (0.41, 0.43), "l_kne": (0.49, 0.42),
                "r_ank": (0.35, 0.31), "l_ank": (0.45, 0.30),
            }, head_dx=-0.05, head_dy=-0.02),
            add_head_kps({
                "neck": (0.59, 0.35),
                "r_sho": (0.57, 0.37), "l_sho": (0.63, 0.37),
                "r_elb": (0.50, 0.44), "l_elb": (0.56, 0.45),
                "r_wri": (0.42, 0.43), "l_wri": (0.48, 0.42),
                "r_hip": (0.56, 0.58), "l_hip": (0.61, 0.58),
                "r_kne": (0.60, 0.70), "l_kne": (0.68, 0.70),
                "r_ank": (0.72, 0.72), "l_ank": (0.78, 0.72),
            }, head_dx=-0.04, head_dy=-0.05, profile="side_left"),
        ],
    },
]


def main():
    os.makedirs(LIB_DIR, exist_ok=True)
    if COMFY_INPUT and os.path.isdir(os.path.dirname(COMFY_INPUT)):
        os.makedirs(COMFY_INPUT, exist_ok=True)
    rel = lambda p: os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
    index_data = {}

    catalog_cols = 5
    catalog_rows = 2
    cell_w, cell_h = 256, 384
    catalog = Image.new("RGB", (catalog_cols * cell_w, catalog_rows * (cell_h + 32)), (20, 20, 20))
    cat_draw = ImageDraw.Draw(catalog)

    for idx, p in enumerate(PRESETS):
        pid = p["id"]
        start_im = render_skeleton(p["start"], 512, 768)
        peak_im = render_skeleton(p["peak"], 512, 768)

        start_path = os.path.join(LIB_DIR, f"{pid}_start.png")
        peak_path = os.path.join(LIB_DIR, f"{pid}_peak.png")
        start_im.save(start_path)
        peak_im.save(peak_path)

        # Also copy to ComfyUI/input (when ComfyUI is found) so LoadImage / gen.py can reference them directly
        if COMFY_INPUT and os.path.isdir(COMFY_INPUT):
            shutil.copy(start_path, os.path.join(COMFY_INPUT, f"pose_{pid}_start.png"))
            shutil.copy(peak_path, os.path.join(COMFY_INPUT, f"pose_{pid}_peak.png"))

        # Generate 9-frame half-cycle motion sequence (384x576) with smooth cosine easing
        seq_dir = os.path.join(LIB_DIR, f"{pid}_seq")
        os.makedirs(seq_dir, exist_ok=True)
        seq_files = []
        for f_idx in range(9):
            # Cosine ease 0 -> 1 over 9 frames (half-cycle for pingpong loop)
            t = 0.5 * (1.0 - math.cos(math.pi * f_idx / 8.0))
            interp_persons = [
                interpolate_kps(p["start"][person_i], p["peak"][person_i], t)
                for person_i in range(len(p["start"]))
            ]
            f_im = render_skeleton(interp_persons, 384, 576)
            fp = os.path.join(seq_dir, f"f_{f_idx:02d}.png")
            f_im.save(fp)
            seq_files.append(fp)

        index_data[pid] = {
            "id": pid,
            "label": p["label"],
            "category": p["category"],
            "anima_pose_tags": p["anima_pose_tags"],
            "qwen_pose_prompt": p["qwen_pose_prompt"],
            "wan_motion_prompt": p["wan_motion_prompt"],
            "start_png": rel(start_path),
            "peak_png": rel(peak_path),
            "comfy_input_start": f"pose_{pid}_start.png",
            "comfy_input_peak": f"pose_{pid}_peak.png",
            "seq_dir": rel(seq_dir),
            "start_kps": p["start"],
            "peak_kps": p["peak"],
        }

        col = idx % catalog_cols
        row = idx // catalog_cols
        thumb = start_im.resize((cell_w, cell_h), Image.LANCZOS)
        ox, oy = col * cell_w, row * (cell_h + 32)
        catalog.paste(thumb, (ox, oy + 32))
        cat_draw.rectangle([ox, oy + 32, ox + cell_w - 1, oy + 32 + cell_h - 1], outline=(90, 90, 90), width=1)
        cat_draw.text((ox + 8, oy + 8), f"{idx+1}. {pid}", fill=(255, 230, 100))

    index_path = os.path.join(LIB_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    cat_path = os.path.join(LIB_DIR, "catalog.png")
    catalog.save(cat_path)
    print(f"Built {len(PRESETS)} NSFW pose & motion presets -> {index_path}")
    print(f"Saved visual catalog -> {cat_path}")


if __name__ == "__main__":
    main()
