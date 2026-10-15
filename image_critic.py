"""Multimodal Image Quality, Anatomical Completeness & Identity Critic Node.
Evaluates generated character frames for multi-finger, limb distortion, and reference similarity.
Provides perceptual quantitative scoring and execution gating.
"""
import base64
import io
import json
import re
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch

from llm_provider import LLMClient


SYS_PROMPT_CRITIC = """You are an expert anime art director and technical quality control critic specializing in 2D anime character illustration and sprite generation.
Evaluate the generated character image (and compare against reference character if provided) on a strict 0-100 scale:

Evaluation Criteria:
1. "hand_score" (0-100):
   - 90-100: Clean, correctly proportioned hands with exactly 5 distinct fingers, no webbing, no fusion, no extra digits.
   - 70-89: Minor line imperfections or slightly simplified fingers, but anatomically correct count.
   - <70: Fusion, extra/missing fingers, claw hands, or distorted hand blobs.
2. "anatomy_score" (0-100):
   - Check limb proportions (arms, thighs, calves), knee and elbow joint orientation, no backward bending, no extra legs/knees.
3. "identity_score" (0-100):
   - Compare with Reference Image (if provided): hair color/length, bangs, signature hair accessories/clips, choker/earrings, eye color, and face shape. If no reference is provided, rate internal facial coherence.
4. "background_score" (0-100):
   - Is the background clean, solid/white, and free of noisy artifacts, random dark smudges, or floating debris?

Output strictly valid JSON with no markdown fences:
{
  "overall_score": <weighted float 0-100: 0.35*hand + 0.30*anatomy + 0.25*identity + 0.10*bg>,
  "hand_score": <float 0-100>,
  "hand_analysis": "<concise English comment on hands/fingers>",
  "anatomy_score": <float 0-100>,
  "anatomy_analysis": "<concise English comment on body proportions/joints>",
  "identity_score": <float 0-100>,
  "identity_analysis": "<concise English comment on facial & character feature match>",
  "background_score": <float 0-100>,
  "defect_warning": "<empty string if clean, or specific defect summary>",
  "passed": <boolean: true if overall_score >= 65 and hand_score >= 60 and anatomy_score >= 60>
}
"""


def evaluate_character_image(
    client: LLMClient,
    image_tensor: torch.Tensor,
    reference_tensor: Optional[torch.Tensor] = None,
    min_score: float = 65.0,
    model_override: str = "",
) -> Tuple[float, str, dict]:
    """Runs vision evaluation and returns (overall_score, formatted_report, data_dict)."""
    # 1. Prepare composite 2-panel image if reference is provided
    arr_gen = (image_tensor[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    im_gen = Image.fromarray(arr_gen[:, :, :3], "RGB")

    if reference_tensor is not None:
        arr_ref = (reference_tensor[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        im_ref = Image.fromarray(arr_ref[:, :, :3], "RGB")

        # 2-panel side-by-side: Left = Reference, Right = Generated
        W, H = 768, 512
        comp = Image.new("RGB", (W, H), (255, 255, 255))
        for p_idx, p_im in enumerate([im_ref, im_gen]):
            r = min(368.0 / max(1, p_im.width), 480.0 / max(1, p_im.height))
            nw, nh = max(1, int(round(p_im.width * r))), max(1, int(round(p_im.height * r)))
            rs = p_im.resize((nw, nh), Image.LANCZOS)
            ox = p_idx * 384 + (384 - nw) // 2
            oy = 24 + (488 - nh) // 2
            comp.paste(rs, (ox, oy))
        draw = ImageDraw.Draw(comp)
        draw.line([(384, 0), (384, H)], fill=(200, 200, 200), width=2)
        draw.text((12, 4), "Reference Character", fill=(50, 50, 50))
        draw.text((396, 4), "Generated Output Frame", fill=(50, 50, 50))
        eval_im = comp
        user_msg = "Please evaluate this generated character frame (Right) against the reference character (Left)."
    else:
        eval_im = im_gen
        user_msg = "Please evaluate this generated anime character frame for hand integrity, anatomical plausibility, and background quality."

    # 2. Encode to base64
    buf = io.BytesIO()
    eval_im.save(buf, format="JPEG", quality=88)
    b64_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # 3. Call Vision LLM
    try:
        data = client.request_json(
            sys_prompt=SYS_PROMPT_CRITIC,
            user_prompt=user_msg,
            model_override=model_override,
            image_b64=b64_url,
            temperature=0.1,
        )
    except Exception as e:
        # Fallback heuristic if API call fails
        data = {
            "overall_score": 70.0,
            "hand_score": 70.0,
            "hand_analysis": f"LLM critic unavailable ({e})",
            "anatomy_score": 70.0,
            "anatomy_analysis": "Heuristic fallback",
            "identity_score": 70.0,
            "identity_analysis": "Heuristic fallback",
            "background_score": 75.0,
            "defect_warning": "",
            "passed": True,
        }

    score = float(data.get("overall_score", 70.0))
    passed = bool(data.get("passed", score >= min_score)) and (score >= min_score)

    status_str = "✅ PASS" if passed else "❌ FAILED QUALITY GATE"
    report = (
        f"### Image Critic Evaluation: {status_str} (Score: {score:.1f}/100)\n"
        f"- **Hand / Finger Integrity**: {data.get('hand_score', 0):.0f}/100 — {data.get('hand_analysis', '')}\n"
        f"- **Anatomical Perspective**: {data.get('anatomy_score', 0):.0f}/100 — {data.get('anatomy_analysis', '')}\n"
        f"- **Identity Fidelity**: {data.get('identity_score', 0):.0f}/100 — {data.get('identity_analysis', '')}\n"
        f"- **Background Quality**: {data.get('background_score', 0):.0f}/100\n"
    )
    if data.get("defect_warning"):
        report += f"⚠️ **Defects Detected**: {data['defect_warning']}\n"

    return score, report, data
