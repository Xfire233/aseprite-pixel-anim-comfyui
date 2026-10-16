#!/usr/bin/env python3
"""Extract a reusable character card JSON (scripts/charcard_<name>.json) from a reference character image.

Combines:
1. Multi-scale crop preprocessing (tight full-body crop + 2x zoomed head/upper-body detail crop) so small facial
   accessories (hairclips, glasses, chokers, eye color) are never lost in full-body tachi-e sprites.
2. Local GPU/CPU ONNX Danbooru tagging (`wd-vit-tagger-v3.onnx`) for exact Danbooru vocabulary grounding.
3. Optional Multimodal LLM refinement (OpenAI-compatible `/chat/completions` & `/responses`) that fuses the
   2-panel visual composite + WD14 Danbooru tags into structured `card`, `outfit`, `outfit_nsfw`, `anima_tags`,
   `anima_nsfw_tags`, `negative_tags`, and `wan_subject` fields.

Usage:
  python scripts/extract_charcard.py --image <path> --name my_char [--local-only]
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY_ROOT, LLM_BASE_URL, PROJECT_ROOT, resolve_llm_key  # noqa: E402

WD14_DIR = os.path.join(COMFY_ROOT, "custom_nodes", "comfyui-WD14-Tagger", "models")
WD14_ONNX = os.path.join(WD14_DIR, "wd-vit-tagger-v3.onnx")
WD14_CSV = os.path.join(WD14_DIR, "wd-vit-tagger-v3.csv")

# Pose / composition / background meta-tags to exclude from permanent character appearance tags
POSE_AND_META_TAGS = {
    "standing", "sitting", "kneeling", "lying", "squatting", "leaning forward", "looking at viewer",
    "looking away", "looking back", "full body", "upper body", "cowboy shot", "portrait", "close-up",
    "white background", "simple background", "transparent background", "gradient background",
    "grey background", "tachi-e", "character sheet", "reference sheet", "multiple views",
    "arms at sides", "hand on hip", "peace sign", "v", "smile", "open mouth", "closed mouth",
    "parted lips", "blush", "embarrassed", "shy", "sweat", "feet out of frame", "cropped",
    "blurry", "solo", "1girl", "virtual youtuber",
}

LOWER_CLOTHING_TAGS = {
    "denim", "shorts", "denim shorts", "short shorts", "blue shorts", "skirt", "pleated skirt", "miniskirt",
    "pants", "trousers", "jeans", "panties", "underwear", "bikini", "black bikini", "swimsuit",
    "pantyhose", "thighhighs", "leggings", "bloomers", "sarong", "dress", "white dress", "denim skirt",
}

UPPER_UNDER_TAGS = {
    "bra", "bikini", "black bikini", "sports bra", "bandeau", "camisole", "crop top", "tube top",
}

_WD14_SESSION = None
_WD14_TAGS = None
_WD14_GEN_IDX = None
_WD14_CHAR_IDX = None


def resolve_api_key(ui_key: str = "") -> str:
    return resolve_llm_key(ui_key)


def make_character_crops(pil_im: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Returns (body_crop_rgb, head_upper_crop_rgb, composite_768x512_rgb)."""
    rgba = pil_im.convert("RGBA")
    arr_a = np.array(rgba)
    # Composite onto white if image has transparency
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb_im = Image.alpha_composite(bg, rgba).convert("RGB")
    arr = np.array(rgb_im)

    # Detect foreground via alpha or non-uniform corner background difference
    if arr_a[:, :, 3].min() < 240 and (arr_a[:, :, 3] > 24).sum() > 100:
        fg_mask = arr_a[:, :, 3] > 24
    else:
        corners = np.vstack([
            arr[:8, :8].reshape(-1, 3),
            arr[:8, -8:].reshape(-1, 3),
            arr[-8:, :8].reshape(-1, 3),
            arr[-8:, -8:].reshape(-1, 3),
        ])
        bg_med = np.median(corners, axis=0)
        dist = np.abs(arr.astype(np.float32) - bg_med).max(axis=2)
        fg_mask = dist > 18

    ys, xs = np.where(fg_mask)
    if len(xs) > 50:
        pad = 8
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(rgb_im.width - 1, int(xs.max()) + pad)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(rgb_im.height - 1, int(ys.max()) + pad)
        body_crop = rgb_im.crop((x0, y0, x1 + 1, y1 + 1))
        ch_h = y1 - y0 + 1
        head_y1 = y0 + max(32, int(ch_h * 0.44))
        # Narrow horizontal bounds to upper body foreground
        ys_u, xs_u = np.where(fg_mask[y0:head_y1, x0:x1 + 1])
        if len(xs_u) > 20:
            hx0 = max(0, x0 + int(xs_u.min()) - pad)
            hx1 = min(rgb_im.width - 1, x0 + int(xs_u.max()) + pad)
        else:
            hx0, hx1 = x0, x1
        head_crop = rgb_im.crop((hx0, y0, hx1 + 1, head_y1))
    else:
        body_crop = rgb_im
        head_crop = rgb_im.crop((0, 0, rgb_im.width, max(32, int(rgb_im.height * 0.45))))

    # Build 768x512 2-panel composite: Left = Full Body (384x512), Right = Zoomed Head/Upper Body (384x512)
    comp = Image.new("RGB", (768, 512), (255, 255, 255))
    for panel_idx, crop_im in enumerate([body_crop, head_crop]):
        r = min(368.0 / max(1, crop_im.width), 480.0 / max(1, crop_im.height))
        nw, nh = max(1, int(round(crop_im.width * r))), max(1, int(round(crop_im.height * r)))
        rs = crop_im.resize((nw, nh), Image.LANCZOS)
        ox = panel_idx * 384 + (384 - nw) // 2
        oy = 24 + (488 - nh) // 2
        comp.paste(rs, (ox, oy))
    draw = ImageDraw.Draw(comp)
    draw.line([(384, 0), (384, 512)], fill=(200, 200, 200), width=2)
    draw.text((12, 4), "Full Body Reference", fill=(40, 40, 40))
    draw.text((396, 4), "Head / Hair / Face / Accessories Close-Up", fill=(40, 40, 40))
    return body_crop, head_crop, comp


def run_local_wd14(pil_im: Image.Image, threshold: float = 0.32, char_threshold: float = 0.50) -> dict:
    """Runs local wd-vit-tagger-v3.onnx on both body_crop and head_crop and categorizes tags."""
    global _WD14_SESSION, _WD14_TAGS, _WD14_GEN_IDX, _WD14_CHAR_IDX
    body_crop, head_crop, comp = make_character_crops(pil_im)

    if not os.path.exists(WD14_ONNX) or not os.path.exists(WD14_CSV):
        return {"character_tags": [], "general_tags": [], "identity_tags": [], "clothing_tags": [], "composite": comp}

    if _WD14_SESSION is None:
        import onnxruntime as ort
        _WD14_SESSION = ort.InferenceSession(WD14_ONNX, providers=["CPUExecutionProvider"])
        _WD14_TAGS = []
        with open(WD14_CSV, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if _WD14_GEN_IDX is None and row[2] == "0":
                    _WD14_GEN_IDX = reader.line_num - 2
                elif _WD14_CHAR_IDX is None and row[2] == "4":
                    _WD14_CHAR_IDX = reader.line_num - 2
                _WD14_TAGS.append(row[1].replace("_", " "))

    inp = _WD14_SESSION.get_inputs()[0]
    h = inp.shape[1]
    out_name = _WD14_SESSION.get_outputs()[0].name

    def _infer(im_c: Image.Image):
        ratio = float(h) / max(im_c.size)
        ns = (max(1, int(im_c.size[0] * ratio)), max(1, int(im_c.size[1] * ratio)))
        im_r = im_c.resize(ns, Image.LANCZOS)
        sq = Image.new("RGB", (h, h), (255, 255, 255))
        sq.paste(im_r, ((h - ns[0]) // 2, (h - ns[1]) // 2))
        arr = np.expand_dims(np.array(sq, dtype=np.float32)[:, :, ::-1], 0)
        return _WD14_SESSION.run([out_name], {inp.name: arr})[0][0]

    p_body = _infer(body_crop)
    p_head = _infer(head_crop)
    p_max = np.maximum(p_body, p_head)
    pairs = list(zip(_WD14_TAGS, p_max))

    chars = [k for k, v in sorted(pairs[_WD14_CHAR_IDX:], key=lambda x: -x[1]) if v >= char_threshold]
    gens = [k for k, v in sorted(pairs[_WD14_GEN_IDX:_WD14_CHAR_IDX], key=lambda x: -x[1]) if v >= threshold]

    # Filter out generic meta/pose tags and conflicting duplicate hair colors
    filtered = [t for t in gens if t not in POSE_AND_META_TAGS]
    identity_tags = []
    clothing_tags = []
    for t in filtered:
        if t in LOWER_CLOTHING_TAGS or t in UPPER_UNDER_TAGS or any(
            w in t for w in ["shirt", "jacket", "coat", "dress", "kimono", "obi", "sash", "shorts", "skirt", "pants", "bikini", "footwear", "shoes", "sandals", "socks", "thighhighs", "sleeves", "belt", "denim"]
        ):
            clothing_tags.append(t)
        else:
            identity_tags.append(t)

    return {
        "character_tags": chars[:2],
        "general_tags": gens,
        "identity_tags": identity_tags,
        "clothing_tags": clothing_tags,
        "composite": comp,
    }


def build_charcard_from_wd14(
    card_name: str,
    wd14_info: dict,
    nsfw_preset: str = "topless_and_bottomless",
) -> dict:
    """Deterministically builds a full charcard dict from WD14 tags when offline or local-only."""
    id_tags = list(wd14_info.get("identity_tags") or [])
    cloth_tags = list(wd14_info.get("clothing_tags") or [])

    base_id = ["1girl", "solo"] + id_tags
    clothed_tags = base_id + cloth_tags

    # Keep outerwear/accessories (open jacket/kimono/choker/sleeves/footwear) while stripping under/lower layers for NSFW
    kept_outer = [
        t for t in cloth_tags
        if t not in LOWER_CLOTHING_TAGS and t not in UPPER_UNDER_TAGS and "shirt" not in t
    ]
    if any("jacket" in t or "coat" in t or "kimono" in t for t in kept_outer) and "open clothes" not in kept_outer:
        kept_outer.append("open clothes")
        kept_outer.append("off shoulder")

    if "bottomless_only" in nsfw_preset:
        kept_upper = [t for t in cloth_tags if t not in LOWER_CLOTHING_TAGS]
        nsfw_tags = base_id + kept_upper + ["nsfw", "bottomless", "no panties", "nude lower body", "pussy", "uncensored", "bare legs"]
        neg_extra = ["panties", "underwear", "shorts", "denim shorts", "skirt", "pants"]
        outfit_nsfw_desc = f"wearing {', '.join(kept_upper) if kept_upper else 'top'}, completely bottomless with no panties and exposed bare pussy and bare legs"
    elif "keep_original" in nsfw_preset:
        nsfw_tags = clothed_tags
        neg_extra = []
        outfit_nsfw_desc = f"wearing {', '.join(cloth_tags)}"
    else:
        outer_str = f"open {', '.join(kept_outer)} slipped off shoulders, " if kept_outer else ""
        nsfw_tags = base_id + kept_outer + [
            "nsfw", "breasts out", "nipples", "bottomless", "no panties", "nude lower body", "pussy", "uncensored", "bare legs"
        ]
        neg_extra = ["bra", "bikini", "panties", "underwear", "shorts", "denim", "denim shorts", "denim skirt", "skirt", "pants"]
        outfit_nsfw_desc = f"{outer_str}completely topless and bottomless with exposed breasts, pink nipples, no panties, and exposed bare pussy"

    is_petite = any(t in wd14_info.get("general_tags", []) for t in ["flat chest", "small breasts", "petite"]) or any(w in card_name.lower() for w in ["hiro", "small", "flat"])
    if is_petite and "keep_original" not in nsfw_preset:
        if "small breasts" not in nsfw_tags:
            nsfw_tags.extend(["small breasts", "flat chest", "slender"])
        neg_extra.extend(["large breasts", "huge breasts"])
    id_str = ", ".join(id_tags[:12]) if id_tags else "anime girl"
    cloth_str = ", ".join(cloth_tags[:10]) if cloth_tags else "casual anime outfit"

    return {
        "name": card_name,
        "card": f"a cute 2D anime girl with {id_str}",
        "outfit": f"{cloth_str}",
        "outfit_nsfw": outfit_nsfw_desc,
        "anima_tags": ", ".join(dict.fromkeys(clothed_tags)),
        "anima_nsfw_tags": ", ".join(dict.fromkeys(nsfw_tags)),
        "negative_tags": ", ".join(dict.fromkeys(["censored", "mosaic censoring", "bar censor"] + neg_extra)),
        "wan_subject": f"the cute anime girl with {id_str}",
        "wd14_raw": {
            "character_tags": wd14_info.get("character_tags", []),
            "identity_tags": id_tags,
            "clothing_tags": cloth_tags,
        },
    }


def extract_charcard(
    pil_im: Image.Image,
    card_name: str = "custom_char",
    nsfw_preset: str = "auto",
    use_llm: bool = True,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    user_prompt: str = "",
    save_to_disk: bool = True,
) -> tuple[dict, str]:
    """Extracts a complete charcard dict guided by the user's prompt intent."""
    wants_strip = True
    if user_prompt and user_prompt.strip():
        up_lower = user_prompt.lower()
        explicit_strip = any(k in user_prompt for k in ("脱", "裸", "不穿", "露出", "全裸", "下身暴露", "胸部暴露", "私处")) or any(k in up_lower for k in ("nsfw", "bottomless", "topless", "nude", "pussy", "nipple"))
        explicit_pose_nsfw = any(k in user_prompt for k in ("m字", "骑乘", "开腿", "深蹲", "娇喘", "自慰", "抽插"))
        wants_strip = explicit_strip or explicit_pose_nsfw

    wd14_preset = "topless_and_bottomless" if wants_strip else "keep_original_outfit"
    wd14_info = run_local_wd14(pil_im)
    fallback_card = build_charcard_from_wd14(card_name, wd14_info, nsfw_preset=wd14_preset)

    base_url = (base_url or LLM_BASE_URL).strip()
    if use_llm and not (base_url and model):
        print("[extract_charcard] No LLM endpoint/model configured; using local WD14 only.")
        use_llm = False
    if use_llm:
        resolved_key = resolve_api_key(api_key)
        comp_im = wd14_info["composite"]
        buf = io.BytesIO()
        comp_im.save(buf, format="JPEG", quality=90)
        b64_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        wd14_summary = (
            f"本地 WD14 预检测到的 Danbooru 标签供参考：\n"
            f"- 角色特征标签: {', '.join(wd14_info['identity_tags'])}\n"
            f"- 原始服装标签: {', '.join(wd14_info['clothing_tags'])}\n"
        )
        user_intent_hint = f"\n【用户当前动作与服装具体意图】\n\"{user_prompt}\"\n" if user_prompt else ""
        sys_prompt = (
            "你是一个专业的二次元角色设定卡（Charcard）提取器，服务于 Anima (Danbooru Tag 模型)、Qwen-Image-Edit 与 Wan2.2 视频模型。\n"
            "用户会提供一张包含【左：全身图 | 右：头部与上半身特写】的角色参考图，以及本地 WD14 预检测标签与用户的具体需求。\n"
            f"{user_intent_hint}"
            "请仔细观察图片（尤其是发型、发色、刘海、发饰形状与颜色、项圈/饰品、瞳色、面部气质、胸围身材体型：如 flat chest / small breasts / petite 或 large breasts，以及服装）：\n"
            "1. 知名角色识别：如果能认出是知名动漫角色（如偶像大师、原神、碧蓝航线、蔚蓝档案、初音未来等），必须在 card 与 anima_tags 中指明角色名字（例如 Shinozaki Hiro, Gakuen Idolmaster）！\n"
            "2. 身材体型：如果是平胸/娇小身材，必须在 card 中写明 slender flat-chested petite build，在 anima_nsfw_tags 中加入 small breasts, flat chest, slender，并在 negative_tags 中封禁 large breasts, huge breasts！\n"
            "3. 服装与配件处理策略（正向词必须精简有效，严禁提取战术配件与武器垃圾）：\n"
            "   - 提取角色设定卡的核心是提炼其【不可替代的生物与面部特征】（发型发色、刘海、兽耳、角、尾巴、瞳色、核心发饰/耳坠、胸围体型）；\n"
            "   - 严禁提取原立绘里繁复的武装战术背带(harness)、腰包(pouch)、手套(gloves)、过膝袜(thighhighs)、战术长靴(combat boots)、护膝(knee pads)、电锯/法杖/重剑/枪械等配件词！这些琐碎配件在生成动作与脱衣时会极度稀释扩散模型的注意力预算，导致四肢与装备相互缠结撕裂；\n"
            "   - 若用户要求脱衣/下装裸露/不穿裤子：outfit_nsfw 必须完全脱去下装与内裤（完全裸腹、私处与双腿暴露），保留滑落的外套/项圈，严禁包含任何下装词与材质词（如 denim, shorts, jeans, pants, skirt, panties）！在 anima_nsfw_tags 中完全剔除所有下装与战术杂项词，只保留极简纯净的 `bottomless, no panties, bare legs, bare pussy`，并在 negative_tags 中封禁 weapon, sword, staff, chainsaw, bag, pouch, combat boots, thighhighs, panties, underwear, shorts, denim, denim shorts, denim skirt, skirt, pants, jeans！\n"
            "   - 若用户要求正常着装（未要求脱衣）：outfit_nsfw 保持原始完整着装，anima_nsfw_tags 保持正常着装标签，不添加暴露词条，negative_tags 不封禁下装。\n"
            "输出严格符合以下结构的 JSON 对象（所有字段均为英文，不要输出任何额外解释）：\n"
            "{\n"
            '  "card": "<英文详细描述角色的固定外貌特征与胸围身材（不含可脱下装）>",\n'
            '  "outfit": "<英文详细描述角色的原始完整着装>",\n'
            '  "outfit_nsfw": "<英文描述角色的服装状态（脱衣或保持原装）>",\n'
            '  "anima_tags": "1girl, solo, <Danbooru角色外貌标签 + 原始服装标签>",\n'
            '  "anima_nsfw_tags": "1girl, solo, <Danbooru角色外貌与服装状态标签>",\n'
            '  "negative_tags": "censored, mosaic censoring, bar censor, <被脱掉的服装词与冲突体型词>",\n'
            '  "wan_subject": "the cute anime girl with <简明英文概括发色、发型、标志性发饰与瞳色>"\n'
            "}"
        )
        user_msg = f"{wd14_summary}\n请结合参考图左半全身与右半头部特写，提取该角色的完整 JSON 设定卡。"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-AsepriteCharcardExtractor/1.0",
        }
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"

        base = base_url.strip().rstrip("/")
        prefer_resp = "argon" in model.lower() or model.lower().endswith("-high")
        endpoints = ["responses", "chat"] if prefer_resp else ["chat", "responses"]

        for ep in endpoints:
            try:
                if ep == "chat":
                    url = f"{base}/chat/completions"
                    payload = {
                        "model": model.strip(),
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": b64_url}},
                                    {"type": "text", "text": user_msg},
                                ],
                            },
                        ],
                        "temperature": 0.2,
                    }
                else:
                    url = f"{base}/responses"
                    payload = {
                        "model": model.strip(),
                        "instructions": sys_prompt,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_image", "image_url": b64_url},
                                    {"type": "input_text", "text": user_msg},
                                ],
                            }
                        ],
                    }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                raw = urllib.request.urlopen(req, timeout=45).read().decode("utf-8")
                rj = json.loads(raw)
                text = ""
                if isinstance(rj.get("choices"), list) and rj["choices"]:
                    msg = rj["choices"][0].get("message") or {}
                    c = msg.get("content")
                    if isinstance(c, str):
                        text = c.strip()
                elif isinstance(rj.get("output_text"), str):
                    text = rj["output_text"].strip()
                else:
                    for item in rj.get("output") or []:
                        if isinstance(item, dict):
                            for c in item.get("content") or []:
                                if isinstance(c, dict) and isinstance(c.get("text"), str):
                                    text = c["text"].strip()
                                    break
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    parsed = json.loads(m.group(0))
                    if parsed.get("anima_tags") and ("hair" in parsed.get("anima_tags", "").lower()):
                        for k in ["card", "outfit", "outfit_nsfw", "anima_tags", "anima_nsfw_tags", "negative_tags", "wan_subject"]:
                            if parsed.get(k):
                                fallback_card[k] = parsed[k]
                        break
            except Exception as e:
                print(f"[extract_charcard] LLM ({ep}) fallback to local WD14: {e}")

    if wants_strip:
        raw_nsfw = fallback_card.get("anima_nsfw_tags", "")
        if raw_nsfw:
            cleaned_tokens = [
                t.strip() for t in raw_nsfw.split(",")
                if t.strip() and t.strip().lower() not in LOWER_CLOTHING_TAGS
                and not any(w in t.strip().lower() for w in ["denim", "shorts", "pants", "jeans", "skirt", "underwear", "panties", "bikini"])
            ]
            fallback_card["anima_nsfw_tags"] = ", ".join(cleaned_tokens)
        neg = fallback_card.get("negative_tags", "")
        neg_additions = ["panties", "underwear", "shorts", "denim", "denim shorts", "denim skirt", "skirt", "pants", "jeans", "bra", "bikini"]
        neg_tokens = [t.strip() for t in neg.split(",") if t.strip()]
        for na in neg_additions:
            if na not in neg_tokens:
                neg_tokens.append(na)
        fallback_card["negative_tags"] = ", ".join(neg_tokens)

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", card_name.strip()) or "custom_char"
    if not safe_name.startswith("charcard_"):
        fname = f"charcard_{safe_name}.json"
    else:
        fname = f"{safe_name}.json"
    out_path = os.path.join(PROJECT_ROOT, "scripts", fname)
    if save_to_disk:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fallback_card, f, ensure_ascii=False, indent=2)
    return fallback_card, fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to reference character image")
    ap.add_argument("--name", default="custom_char", help="Short character name (saves scripts/charcard_<name>.json)")
    ap.add_argument("--nsfw-preset", default="topless_and_bottomless", choices=["topless_and_bottomless", "bottomless_only", "keep_original_outfit"])
    ap.add_argument("--local-only", action="store_true", help="Use only local WD14 ONNX tagger without external LLM API")
    ap.add_argument("--base-url", default=LLM_BASE_URL, help="OpenAI-compatible endpoint (or env LLM_BASE_URL); required unless --local-only")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""), help="model id on that endpoint (or env LLM_MODEL)")
    args = ap.parse_args()

    im = Image.open(args.image)
    card, fname = extract_charcard(
        im,
        card_name=args.name,
        nsfw_preset=args.nsfw_preset,
        use_llm=not args.local_only,
        base_url=args.base_url,
        model=args.model,
        save_to_disk=True,
    )
    print(f"Saved scripts/{fname}:")
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
