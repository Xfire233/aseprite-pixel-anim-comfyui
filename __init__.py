"""ComfyUI custom node wrapping scripts/pixelize.py + assemble.lua + validate_loop.py.

Takes IMAGE frames directly from VAEDecode/VAEDecodeTiled, runs the deterministic
pixelization + shared-palette + temporal alpha consensus + Aseprite CLI (.aseprite / .gif / _sheet.png)
pipeline, validates loop coherence, and displays the animated GIF + zoom sheet right inside the node.
"""
import base64
import gc
import io
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from aiohttp import web
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import folder_paths
import math
import node_helpers
import server
# This repository IS the custom node: clone it into ComfyUI/custom_nodes/ and scripts/, poses/, work/ sit next to this file.
PROJECT_ROOT = os.environ.get("ASEPRITE_WF_ROOT", os.path.dirname(os.path.abspath(__file__)))
PIXELIZE_PY = os.path.join(PROJECT_ROOT, "scripts", "pixelize.py")
VALIDATE_PY = os.path.join(PROJECT_ROOT, "scripts", "validate_loop.py")
SECRET_KEY_FILE = os.path.join(PROJECT_ROOT, "work", ".llm_api_key.secret")
MODELS_CACHE_FILE = os.path.join(PROJECT_ROOT, "work", ".llm_models_cache.json")
POSE_LIB_INDEX = os.path.join(PROJECT_ROOT, "poses", "nsfw_library", "index.json")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# No endpoint is baked in: fill base_url in the node (or env LLM_BASE_URL), then "fetch models" to populate this list.
DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
DEFAULT_MODELS = [os.environ.get("LLM_MODEL", "")] if os.environ.get("LLM_MODEL") else ["(先填 base_url 再点获取模型)"]

DEFAULT_SYSTEM_PROMPT = """你是一个专精于 Anima (Danbooru Tag 图像模型) 与 Wan2.2 I2V (图生视频动作模型) 的提示词编译器。
角色固定外观与服装标签已预置为宏 `{{CHARCARD_ANIMA_TAGS}}`，固定负面冲突词已预置为宏 `{{CHARCARD_NEG_TAGS}}`，角色英文主语已预置为宏 `{{CHARCARD_WAN_SUBJECT}}`。

【体型与服装一致性红线（严禁 OOC，精炼去重）】：
1. 角色体型一致性：严格维护角色原设身材！若角色为平胸/娇小身材（如 Shinozaki Hiro），正面提示词严禁出现任何大胸描述，负面词必须包含 `large breasts, huge breasts, hanging breasts, cleavage, wide hips`，绝不可被 NSFW LoRA 误导变异为巨乳/肥臀！
2. 正向关键词「精简而有效」法则（注意力预算最大化，防肢体崩坏）：
   - 强锁定（硬核识别特征）：只保留最核心的 5-8 个不可替代特征（发色、发长、刘海、标志性兽耳/角/尾巴、瞳色、核心发饰），用权重括号前置（如 (cat ears:1.2), (horns:1.2)）；
   - 彻底剥离战术配件与装备垃圾：在进行脱衣/下身暴露/做动作时，严禁把原图立绘里繁复的战术背带(harness)、腰包(pouch)、手套(gloves)、过膝袜(thighhighs)、战术长靴(combat boots)、护膝(knee pads)、电锯/法杖/武器等配件词塞入正向词！这些琐碎配件会极度稀释扩散模型的注意力预算，导致四肢与装备相互缠结撕裂、M腿等大动作彻底崩溃；
   - 下半身极简纯净：涉及下身脱除或裸露时，下半身只保留极简的 `bottomless, no panties, bare legs, bare pussy`，让模型将全部注意力集中于解剖学透视与动作骨骼！
3. 精准脱衣：若用户仅要求“下身不穿裤子/下装暴露”，只脱下半身（bottomless, no panties, bare legs），上衣保持原样；严禁在用户未明确要求脱上衣时擅自添加 topless 或 exposed breasts！
4. 双人姿势（Duo Pose）稳定性与画风统一准则：
   - 角色层级明确：主角色（Picture 1 的二次元角色）占据视觉前景，副角色统一描述为 `faceless 2D anime male partner`，严禁写成真实真人或半写实面孔！
   - 画风 100% 统一：双人场景正面词强包含 `2d, cel shading, clean lineart, anime style, 1girl, 1boy, hetero, sex`，负面词必须封杀 `photorealistic, realistic, 3d, realistic skin, real photo, western comic`；
   - 大冲程起伏动作法则（骑乘/抽送）：骑乘位视频动作必须编译为大幅度垂直起伏冲程（large amplitude riding motion, intense deep rhythmic up-and-down pelvic bouncing, smoothly lifting her hips up and driving them down with full range of motion, pronounced vertical stroke, vigorous hip bounce and chest bounce, full body continuous smooth riding），负面词封杀 `static, still, motionless, stationary, subtle, barely moving`！
5. 严禁冗余堆叠：去除所有重复出现的 tag，正面词总词条数量严格控制在 30 个以内，保持提示词绝对精炼、高效、聚焦。
请根据用户输入的【中文姿势与动作描述】，严格以 JSON 格式输出以下 4 个字段（保留宏名原样输出）：
{
  "anima_positive": "masterpiece, best quality, score_9, score_8, newest, {{CHARCARD_ANIMA_TAGS}}, <精确姿势与视角标签，精炼无重复>, full body, white background, simple background",
  "anima_negative": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, censored, mosaic censoring, bar censor, multiple girls, 2girls, cropped, out of frame, bad anatomy, extra limbs, {{CHARCARD_NEG_TAGS}}",
  "wan_positive": "anime style, {{CHARCARD_WAN_SUBJECT}} <当前姿势与表情> on a plain white background, <柔和微动轨迹，不堆砌tag>, static camera, fixed composition, full body, clean 2D anime illustration, smooth subtle motion",
  "wan_negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，camera movement, zoom, morphing, warping, changing clothes"
}"""

_PROMPT_CACHE = {}

def _load_pose_library() -> dict:
    if os.path.exists(POSE_LIB_INDEX):
        try:
            with open(POSE_LIB_INDEX, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_pose_preset_choices() -> list:
    choices = ["none (不使用预设骨骼)"]
    lib = _load_pose_library()
    for pid, meta in lib.items():
        lbl = str(meta.get("label") or pid)
        choices.append(lbl if lbl.startswith(pid) else f"{pid} ({lbl})")
    return choices


_CHARCARD_NONE = "none (不使用预设角色卡，自动识别参考图1外观)"


def _get_charcard_choices() -> list:
    """'none' (read appearance from the reference image) plus every scripts/charcard*.json on disk."""
    choices = [_CHARCARD_NONE]
    for p in sorted(glob.glob(os.path.join(PROJECT_ROOT, "scripts", "charcard*.json"))):
        fn = os.path.basename(p)
        lbl = "预设角色卡"
        try:
            d = json.load(open(p, encoding="utf-8"))
            if d.get("name"):
                lbl = str(d["name"])
        except Exception:
            pass
        choices.append(f"{fn} ({lbl})")
    return choices
def _resolve_api_key(raw_key: str) -> str:
    """UI key (or name of an env var typed into the UI) > local git-ignored secret file > env LLM_API_KEY."""
    val = (raw_key or "").strip()
    if val and not re.match(r"^[•*\s]+$", val) and not val.startswith("(空") and val != "__SAVED__":
        if val in os.environ:
            return os.environ[val]
        # Save newly entered key locally so it never ends up inside workflow JSON
        os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
        with open(SECRET_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(val)
        return val
    if os.path.exists(SECRET_KEY_FILE):
        saved = open(SECRET_KEY_FILE, encoding="utf-8").read().strip()
        if saved:
            return saved
    return os.environ.get("LLM_API_KEY", "")


def _get_cached_models():
    models = list(DEFAULT_MODELS)
    if os.path.exists(MODELS_CACHE_FILE):
        try:
            cached = json.load(open(MODELS_CACHE_FILE, encoding="utf-8"))
            if isinstance(cached, list) and cached:
                for m in cached:
                    if isinstance(m, str) and m not in models:
                        models.append(m)
        except Exception:
            pass
    return models

if hasattr(server, "PromptServer") and getattr(server.PromptServer, "instance", None) is not None:
    @server.PromptServer.instance.routes.post("/aseprite_llm/fetch_models")
    async def fetch_llm_models(request):
        try:
            body = await request.json()
            base_url = (body.get("base_url") or "").strip().rstrip("/")
            api_key = _resolve_api_key(body.get("api_key") or "")
            if not base_url:
                return web.json_response({"ok": False, "error": "base_url 为空"})
            url = f"{base_url}/models"
            headers = {"User-Agent": "ComfyUI-AsepritePromptDirector/1.0"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(url, headers=headers)
            raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
            data = json.loads(raw)
            items = data.get("data", []) if isinstance(data, dict) else []
            models = [it.get("id") for it in items if isinstance(it, dict) and it.get("id")]
            if not models:
                return web.json_response({"ok": False, "error": f"接口未返回有效模型列表: {raw[:200]}"})
            os.makedirs(os.path.dirname(MODELS_CACHE_FILE), exist_ok=True)
            with open(MODELS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2, ensure_ascii=False)
            return web.json_response({"ok": True, "models": models})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)})

    @server.PromptServer.instance.routes.post("/aseprite_llm/save_key")
    async def save_llm_key(request):
        try:
            body = await request.json()
            raw_key = body.get("api_key") or ""
            _resolve_api_key(raw_key)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)})

_BIREFNET_MODEL = None
_BIREFNET_TFM = None


def _get_birefnet_mask(pil_rgb: Image.Image) -> np.ndarray:
    """Runs RMBG-2.0 (BiRefNet) to get an [H, W] uint8 (0..255) foreground alpha mask,
    keeping the model on CPU when idle so it uses 0 MB of GPU VRAM during diffusion.
    """
    global _BIREFNET_MODEL, _BIREFNET_TFM
    rmbg_dir = os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0")
    if not os.path.exists(rmbg_dir):
        return None
    try:
        if _BIREFNET_MODEL is None:
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms
            _BIREFNET_MODEL = AutoModelForImageSegmentation.from_pretrained(rmbg_dir, trust_remote_code=True).eval()
            _BIREFNET_TFM = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        _BIREFNET_MODEL.to("cuda")
        inp = _BIREFNET_TFM(pil_rgb.convert("RGB")).unsqueeze(0).to("cuda")
        with torch.no_grad():
            pred = _BIREFNET_MODEL(inp)[-1].sigmoid().cpu()[0, 0].numpy()
        _BIREFNET_MODEL.to("cpu")
        torch.cuda.empty_cache()
        mask_pil = Image.fromarray((pred * 255.0).round().astype(np.uint8)).resize(pil_rgb.size, Image.BILINEAR)
        return np.array(mask_pil)
    except Exception as e:
        print(f"[BiRefNet Mask Warning] {e}")
        try:
            if _BIREFNET_MODEL is not None:
                _BIREFNET_MODEL.to("cpu")
        except Exception:
            pass
        return None


def _clean_bg_to_pure_white(arr_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Replaces background (mask < 165) with pure white (255,255,255) and decontaminates dark room fringes."""
    out = arr_rgb.copy().astype(np.float32)
    fg_core = mask_u8 >= 235
    ref_fg = out.copy()
    valid = fg_core.copy()
    if valid.sum() > 50:
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        for _ in range(8):
            unfilled = ~valid
            if not unfilled.any():
                break
            acc = np.zeros_like(out)
            cnt = np.zeros(out.shape[:2], dtype=np.float32)
            for dy, dx in neighbors:
                v_s = np.roll(np.roll(valid, dy, axis=0), dx, axis=1)
                c_s = np.roll(np.roll(ref_fg, dy, axis=0), dx, axis=1)
                m_t = unfilled & v_s
                acc[m_t] += c_s[m_t]
                cnt[m_t] += 1.0
            new_m = unfilled & (cnt > 0)
            if not new_m.any():
                break
            ref_fg[new_m] = acc[new_m] / cnt[new_m, None]
            valid |= new_m
    alpha = np.clip((mask_u8.astype(np.float32) - 140.0) / 100.0, 0.0, 1.0)[:, :, None]
    comp = ref_fg * alpha + 255.0 * (1.0 - alpha)
    comp[mask_u8 >= 235] = out[mask_u8 >= 235]
    return comp.round().clip(0, 255).astype(np.uint8)

class AsepritePixelizeAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "name": ("STRING", {"default": "wan_sprite"}),
                "canvas": ("STRING", {"default": "128x224"}),
                "subsample": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1,
                                      "tooltip": "Number of keyframes to subsample evenly from input video frames (0 = use all frames)"}),
                "pingpong": ("BOOLEAN", {"default": True,
                                         "tooltip": "Mirror keyframes (1..K..2) into a closed ping-pong loop"}),
                "consensus": (["alpha", "none", "full"], {"default": "alpha"}),
                "colors": ("INT", {"default": 32, "min": 4, "max": 256, "step": 1}),
                "frame_duration_ms": ("INT", {"default": 180, "min": 20, "max": 2000, "step": 10}),
                "gif_scale": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "tag": ("STRING", {"default": "idle_loop"}),
            },
            "optional": {
                "enable_pixelize": ("BOOLEAN", {"default": True, "tooltip": "开启则进行像素化下采样并导出像素动画；关闭则跳过像素化与组装，直接直通视频帧（适合仅需生成 MP4 视频的用户）"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("pixel_frames", "zoom_sheet", "report")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "pixel_art/aseprite"

    def execute(
        self,
        images,
        name="wan_sprite",
        canvas="128x224",
        subsample=4,
        pingpong=True,
        consensus="alpha",
        colors=32,
        frame_duration_ms=180,
        gif_scale=4,
        tag="idle_loop",
        prompt=None,
        extra_pnginfo=None,
        **kwargs,
    ):
        enable_pixelize = kwargs.get("enable_pixelize", True)
        if enable_pixelize is False:
            report = "[AsepritePixelizeAssemble] 像素化与动画组装已由用户关闭。视频帧已直通输出，MP4 视频由前面的 SaveVideo 节点正常落盘。"
            print(report)
            _purge_vram_and_vbars()
            return {"ui": {"images": []}, "result": (images, images[:1], report)}
        if not isinstance(name, str) or not name.strip():
            name = "wan_sprite"
        if not isinstance(canvas, str) or "x" not in canvas:
            canvas = "128x224"
        try:
            subsample = int(subsample)
            if not (0 <= subsample <= 64):
                subsample = 4
        except Exception:
            subsample = 4
        pingpong = bool(pingpong) if isinstance(pingpong, bool) else True
        if not isinstance(consensus, str) or consensus not in ("alpha", "none", "full"):
            consensus = "alpha"
        try:
            colors = int(colors)
            if not (4 <= colors <= 256) or colors == 180:
                colors = 32
        except Exception:
            colors = 32
        try:
            frame_duration_ms = int(frame_duration_ms)
            if frame_duration_ms < 20:
                frame_duration_ms = 180
        except Exception:
            frame_duration_ms = 180
        try:
            gif_scale = int(gif_scale)
            if not (1 <= gif_scale <= 10):
                gif_scale = 4
        except Exception:
            gif_scale = 4
        if not isinstance(tag, str) or not tag.strip():
            tag = "idle_loop"

        safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name.strip()) or "wan_sprite"
        raw_dir = os.path.join(PROJECT_ROOT, "work", f"{safe_name}_raw")
        work_dir = os.path.join(PROJECT_ROOT, "work", f"{safe_name}_pixel")
        out_dir = os.path.join(folder_paths.get_output_directory(), "aseprite")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        # Clean old raw/work frames so stale frame counts don't mix
        for old_p in glob.glob(os.path.join(raw_dir, "f_*.png")) + glob.glob(os.path.join(work_dir, "frame*.png")):
            try:
                os.remove(old_p)
            except OSError:
                pass

        # 1. Save incoming ComfyUI IMAGE batch [N, H, W, C] to raw_dir
        imgs_np = (images.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        # Always segment frames with BiRefNet (RMBG-2.0) when available so floor shadows and border specks never leak
        raw_paths = []
        for idx, arr in enumerate(imgs_np, 1):
            p = os.path.join(raw_dir, f"f_{idx:05d}_.png")
            pil_im = Image.fromarray(arr[:, :, :3], "RGB")
            m_u8 = _get_birefnet_mask(pil_im)
            if m_u8 is not None and (m_u8 >= 165).sum() > 200:
                Image.fromarray(np.dstack([arr[:, :, :3], m_u8]), "RGBA").save(p)
            else:
                pil_im.save(p)
            raw_paths.append(p)
        ase_path = os.path.join(out_dir, f"{safe_name}.aseprite")
        gif_path = os.path.join(out_dir, f"{safe_name}.gif")
        sheet_path = os.path.join(out_dir, f"{safe_name}_sheet.png")
        zoom_path = os.path.join(PROJECT_ROOT, "work", f"{safe_name}_zoom.png")

        use_pingpong = bool(pingpong)
        k = subsample if (0 < subsample < len(raw_paths)) else len(raw_paths)
        n_out = (2 * k - 2) if (use_pingpong and k >= 3) else k
        durations_str = ",".join([str(int(frame_duration_ms))] * max(1, n_out))
        # 2. Run scripts/pixelize.py
        cmd_pix = [
            sys.executable,
            PIXELIZE_PY,
            "--frames", *raw_paths,
            "--canvas", canvas,
            "--out", work_dir,
            "--assemble", ase_path,
            "--tag", tag,
            "--durations", durations_str,
            "--zoom", zoom_path,
            "--gif-scale", str(int(gif_scale)),
            "--consensus", ("alpha" if consensus not in ("alpha", "none", "full") else consensus),
            "--colors", str(int(colors)),
            "--no-register",
        ]
        if subsample > 0:
            cmd_pix += ["--subsample", str(int(subsample))]
        if use_pingpong:
            cmd_pix += ["--pingpong"]

        r_pix = subprocess.run(cmd_pix, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
        if r_pix.returncode != 0:
            raise RuntimeError(f"pixelize.py failed (exit {r_pix.returncode}):\n{r_pix.stdout}\n{r_pix.stderr}")

        # 3. Run scripts/validate_loop.py (non-fatal report so user can see PASS/FAIL in UI)
        pix_frame_paths = sorted(
            glob.glob(os.path.join(work_dir, "frame*.png")),
            key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or 0),
        )
        cmd_val = [sys.executable, VALIDATE_PY, "--frames", *pix_frame_paths]
        if consensus == "none":
            cmd_val += ["--w-spread", "8"]
        r_val = subprocess.run(cmd_val, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)

        # 4. GIF / sheet already live in ComfyUI output/aseprite; copy the zoom sheet there for inline Web UI preview
        comfy_out = out_dir
        ui_images = []

        from PIL.PngImagePlugin import PngInfo

        metadata = None
        if prompt is not None or extra_pnginfo is not None:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        # Save standalone workflow JSON alongside for 100% fail-safe parameter recovery
        if extra_pnginfo and "workflow" in extra_pnginfo:
            try:
                wf_path = os.path.join(out_dir, f"{safe_name}_workflow.json")
                with open(wf_path, "w", encoding="utf-8") as f:
                    json.dump(extra_pnginfo["workflow"], f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[AsepritePixelizeAssemble] workflow.json save warning: {e}")
        if prompt is not None:
            try:
                p_path = os.path.join(out_dir, f"{safe_name}_prompt.json")
                with open(p_path, "w", encoding="utf-8") as f:
                    json.dump(prompt, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[AsepritePixelizeAssemble] prompt.json save warning: {e}")

        for src, suffix in [(gif_path, ".gif"), (zoom_path, "_zoom.png"), (sheet_path, "_sheet.png")]:
            if os.path.exists(src):
                fname = f"{safe_name}{suffix}"
                dst = os.path.join(comfy_out, fname)
                if suffix.endswith(".png") and metadata is not None:
                    try:
                        im = Image.open(src)
                        im.save(dst, pnginfo=metadata)
                    except Exception as e:
                        print(f"[AsepritePixelizeAssemble] metadata save error on {fname}: {e}")
                        if os.path.abspath(src) != os.path.abspath(dst):
                            shutil.copyfile(src, dst)
                else:
                    if os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copyfile(src, dst)
                ui_images.append({"filename": fname, "subfolder": "aseprite", "type": "output"})
        # 5. Build output IMAGE tensors (upscaled pixel frames on dark gray + zoom sheet)
        upscaled_tensors = []
        for p in pix_frame_paths:
            rgba = Image.open(p).convert("RGBA")
            w_c, h_c = rgba.size
            bg = Image.new("RGBA", (w_c, h_c), (40, 40, 40, 255))
            bg.alpha_composite(rgba)
            up = bg.convert("RGB").resize((w_c * max(1, int(gif_scale)), h_c * max(1, int(gif_scale))), Image.NEAREST)
            upscaled_tensors.append(torch.from_numpy(np.array(up).astype(np.float32) / 255.0))
        pixel_frames_tensor = torch.stack(upscaled_tensors, dim=0)

        if os.path.exists(zoom_path):
            z_rgb = Image.open(zoom_path).convert("RGB")
            zoom_tensor = torch.from_numpy(np.array(z_rgb).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            zoom_tensor = pixel_frames_tensor[:1]

        report = (
            f"=== Aseprite Outputs ===\n"
            f".aseprite : {ase_path}\n"
            f".gif ({gif_scale}x) : {gif_path}\n"
            f"sheet (1x): {sheet_path}\n"
            f"zoom (2x) : {zoom_path}\n\n"
            f"=== pixelize.py ===\n{r_pix.stdout.strip()}\n\n"
            f"=== validate_loop.py ===\n{r_val.stdout.strip()}"
        )
        print(report)
        _purge_vram_and_vbars()
        print("[AsepritePixelizeAssemble] Freed Stage 2 Wan2.2 14B vbars from VRAM after completion so re-rolls never stall.")
        return {"ui": {"images": ui_images}, "result": (pixel_frames_tensor, zoom_tensor, report)}

class AsepriteLLMProvider:
    """Centralized LLM Connection Provider node distributing LLM_CLIENT to all downstream nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_LLM_BASE_URL, "tooltip": "OpenAI 兼容接口地址，例如 https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": "", "tooltip": "API 密钥，点击下方按钮或留空可读取本地保存的配置"}),
                "model": (_get_cached_models(), {"default": _get_cached_models()[0]}),
                "timeout": ("INT", {"default": 45, "min": 5, "max": 300, "step": 5}),
            }
        }

    RETURN_TYPES = ("LLM_CLIENT",)
    RETURN_NAMES = ("llm_client",)
    FUNCTION = "provide"
    CATEGORY = "pixel_art/aseprite"

    def provide(self, base_url="", api_key="", model="", timeout=45):
        from llm_provider import LLMClient
        client = LLMClient(
            base_url=base_url or DEFAULT_LLM_BASE_URL,
            api_key=api_key or "",
            default_model=model or _get_cached_models()[0],
            timeout=int(timeout),
        )
        return (client,)


class AsepriteLLMPoseGenerator:
    """Generates standard OpenPose 18-point skeleton images & continuous multi-frame sequences
    from natural language action descriptions via LLM + kinematic constraints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action_prompt": ("STRING", {"multiline": True, "default": "sitting on the floor in an M-shaped spread legs pose, hands on outer thighs, breathing gently"}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 24, "step": 1, "tooltip": "1=生成单帧骨骼；>1=生成连续动作时序序列（默认 9 帧）"}),
                "motion_mode": (["pingpong", "loop", "linear"], {"default": "pingpong"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 16}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1536, "step": 16}),
                "bg_color": (["white", "black"], {"default": "white"}),
                "enforce_constraints": ("BOOLEAN", {"default": True, "tooltip": "开启前置解剖学刚性拓扑约束，钳位防止肢体反折或脱臼"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "随机种子；改变或随机化即可重新生成全新的骨骼姿态坐标"}),
            },
            "optional": {
                "llm_client": ("LLM_CLIENT",),
                "model_override": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return kwargs.get("seed", float("nan"))

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_images", "pose_keypoints", "pose_summary")
    FUNCTION = "generate"
    CATEGORY = "pixel_art/aseprite"

    def generate(self, action_prompt, num_frames=1, motion_mode="pingpong", width=512, height=768, bg_color="white", enforce_constraints=True, seed=0, llm_client=None, model_override=""):
        from pose_generator import (
            SYS_PROMPT_POSE_GEN,
            desensitize_pose_prompt,
            heuristic_fallback_pose,
            enforce_anatomical_constraints,
            interpolate_pose_sequence,
            render_openpose_skeleton,
        )
        from llm_provider import LLMClient

        client = llm_client
        if client is None:
            client = LLMClient(base_url=DEFAULT_LLM_BASE_URL, api_key="", default_model=_get_cached_models()[0])

        clean_prompt = desensitize_pose_prompt(action_prompt)
        print(f"[AsepriteLLMPoseGenerator] (seed {seed}) Raw action: '{action_prompt[:60]}' -> Kinematic: '{clean_prompt[:60]}'")

        try:
            temp = 0.15 + (int(seed) % 1000) * 0.0003
            data = client.request_json(
                sys_prompt=SYS_PROMPT_POSE_GEN,
                user_prompt=clean_prompt,
                model_override=model_override,
                temperature=temp,
            )
            start_kps = data.get("start_kps", [])
            peak_kps = data.get("peak_kps", start_kps)
            summary = data.get("pose_summary", clean_prompt)
            if not start_kps or len(start_kps) < 14:
                raise ValueError(f"LLM returned incomplete keypoints ({len(start_kps)} points)")
        except Exception as e:
            print(f"[AsepriteLLMPoseGenerator] LLM synthesis blocked/failed ({e}); engaged kinematic heuristic fallback.")
            start_kps, peak_kps, summary = heuristic_fallback_pose(action_prompt, seed=int(seed))
            start_kps = enforce_anatomical_constraints(start_kps)
            peak_kps = enforce_anatomical_constraints(peak_kps)

        seq_kps = interpolate_pose_sequence(start_kps, peak_kps, num_frames=max(1, int(num_frames)), mode=motion_mode)

        tensors = []
        for frame_kps in seq_kps:
            pil_im = render_openpose_skeleton(frame_kps, width=int(width), height=int(height), bg_color=bg_color)
            arr = np.array(pil_im).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))

        batch_tensor = torch.stack(tensors, dim=0)

        pose_keypoints = []
        for frame_kps in seq_kps:
            flat = []
            for pt in frame_kps:
                flat.extend([float(pt[0]) * width, float(pt[1]) * height, float(pt[2])])
            pose_keypoints.append({
                "people": [{"pose_keypoints_2d": flat}],
                "canvas_width": int(width),
                "canvas_height": int(height),
            })

        return (batch_tensor, pose_keypoints, summary)


class AsepriteImageCritic:
    """Perceptual Multimodal Visual Quality & Identity QA Evaluator with optional execution gate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_gate": ("BOOLEAN", {"default": False, "tooltip": "若开启且总分低于 min_score，则熔断拦截后续视频渲染"}),
                "min_score": ("FLOAT", {"default": 65.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "llm_client": ("LLM_CLIENT",),
                "model_override": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("passed_image", "overall_score", "report", "critic_json")
    FUNCTION = "evaluate"
    CATEGORY = "pixel_art/aseprite"

    def evaluate(self, image, enable_gate=False, min_score=65.0, reference_image=None, llm_client=None, model_override=""):
        from image_critic import evaluate_character_image
        from llm_provider import LLMClient
        from comfy_execution.graph import ExecutionBlocker

        client = llm_client
        if client is None:
            client = LLMClient(base_url=DEFAULT_LLM_BASE_URL, api_key="", default_model=_get_cached_models()[0])

        score, report, data_dict = evaluate_character_image(
            client=client,
            image_tensor=image,
            reference_tensor=reference_image,
            min_score=float(min_score),
            model_override=model_override,
        )

        passed = bool(data_dict.get("passed", score >= min_score)) and (score >= min_score)
        print(f"[AsepriteImageCritic] {report.strip()}")

        if enable_gate and not passed:
            msg = f"Image Critic Quality Gate Blocked: Score {score:.1f} < {min_score:.1f}. Defects: {data_dict.get('defect_warning') or 'Low anatomy/identity score'}"
            return (ExecutionBlocker(msg), score, report, json.dumps(data_dict, indent=2, ensure_ascii=False))

        return (image, score, report, json.dumps(data_dict, indent=2, ensure_ascii=False))

class AsepritePoseSelector:
    """Switches between Builtin Pose Preset (Node 42) and LLM Generated Pose (Node 65)
    to dynamically route the active pose image and description into Stage 1A and Prompt Director.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["builtin_preset (使用阶段0骨骼预设库)", "llm_generated (使用阶段0自然语言生成的动作)"], {"default": "builtin_preset (使用阶段0骨骼预设库)"}),
            },
            "optional": {
                "preset_pose_image": ("IMAGE",),
                "preset_pose_name": ("STRING", {"default": ""}),
                "llm_pose_image": ("IMAGE",),
                "llm_pose_summary": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("active_pose_image", "active_pose_name")
    FUNCTION = "select_pose"
    CATEGORY = "pixel_art/aseprite"

    def select_pose(self, source, preset_pose_image=None, preset_pose_name="", llm_pose_image=None, llm_pose_summary=""):
        if "llm_generated" in source:
            img = llm_pose_image if llm_pose_image is not None else preset_pose_image
            name = llm_pose_summary if llm_pose_summary else preset_pose_name
            return (img, name)
        else:
            img = preset_pose_image if preset_pose_image is not None else llm_pose_image
            name = preset_pose_name if preset_pose_name else llm_pose_summary
            return (img, name)


class AsepriteStageBridge:
    """Bridges Stage 1 (Anima reference image generation) to Stage 2 (Wan2.2 I2V).

    Unloads Stage 1 models (Anima UNet / Qwen CLIP / Qwen VAE) from 16GB VRAM once
    the reference image is decoded, preventing VRAM contention when Wan2.2 14B loads.
    Also supports optional override_image input if the user wants to feed an existing image,
    and skip_stage2_video to easily test Stage 1 single frames rapidly without waiting for video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anima_image": ("IMAGE",),
                "skip_stage2_video": ("BOOLEAN", {"default": False,
                                                   "tooltip": "若为 True，则跳过后续 Stage 2 Wan 视频生成与像素化，专心快速调试 Stage 1 单帧效果"}),
                "use_override_image": ("BOOLEAN", {"default": False,
                                                   "tooltip": "If True and override_image is connected, use override_image instead of anima_image"}),
                "unload_stage1_vram": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Free Stage 1 (Anima) models from GPU VRAM before Wan2.2 14B starts"}),
                "save_as_input_name": ("STRING", {"default": "anima_latest_ref.png"}),
            },
            "optional": {
                "override_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("start_image",)
    FUNCTION = "bridge"
    CATEGORY = "pixel_art/aseprite"

    def bridge(self, anima_image, skip_stage2_video=False, use_override_image=False, unload_stage1_vram=True, save_as_input_name="anima_latest_ref.png", override_image=None):
        chosen = override_image if (use_override_image and override_image is not None) else anima_image
        if save_as_input_name and save_as_input_name.strip():
            fname = os.path.basename(save_as_input_name.strip())
            if not fname.lower().endswith(".png"):
                fname += ".png"
            in_path = os.path.join(folder_paths.get_input_directory(), fname)
            arr = (chosen[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
            Image.fromarray(arr[:, :, :3], "RGB").save(in_path)

        if skip_stage2_video:
            from comfy_execution.graph import ExecutionBlocker
            print("[AsepriteStageBridge] Stage 2 video generation skipped by user (skip_stage2_video=True). Stage 1 frame preserved.")
            return (ExecutionBlocker("Stage 2 视频生成已跳过（仅运行 Stage 1 单帧调试）"),)

        if unload_stage1_vram:
            _purge_vram_and_vbars()
            print("[AsepriteStageBridge] Unloaded Stage 1 models + vbars from VRAM before Stage 2 (Wan2.2 14B).")

        return (chosen,)


class AsepriteLLMPromptDirector:
    """Calls an OpenAI-compatible LLM endpoint inside ComfyUI to compile a short Chinese
    pose + motion description (plus character card) into standardized Stage 1 (Anima Danbooru tags)
    and Stage 2 (Wan2.2 natural-language motion) positive & negative prompts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_LLM_BASE_URL, "tooltip": "OpenAI 兼容接口地址，例如 https://api.openai.com/v1；留空则只用本地模板编译"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (_get_cached_models(), {"default": _get_cached_models()[0]}),
                "charcard": (
                    _get_charcard_choices(),
                    {"default": _CHARCARD_NONE},
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "参考图1的角色，换成M字开腿坐在地上的姿势，下身不穿裤子完全裸露，双手扶在大腿上，害羞脸红；动作：缓慢呼吸，胸口与肩膀轻微起伏",
                    },
                ),
            },
            "optional": {
                "llm_client": ("LLM_CLIENT", {"tooltip": "接入统一 AsepriteLLMProvider 客户端，无需在本地填写 URL 和 Key"}),
                "reference_image": ("IMAGE",),
                "charcard_file": ("STRING", {"forceInput": True, "tooltip": "接「角色卡提取」节点的 charcard_filename；接上后覆盖上方 charcard 下拉框"}),
                "custom_pose_name": ("STRING", {"forceInput": True, "tooltip": "接 AsepritePoseSelector 或 AsepriteLLMPoseGenerator 的动态姿势描述，当次运行直接生效"}),
                "pose_preset": (_get_pose_preset_choices(), {"default": "none (不使用预设骨骼)"}),
                "prompt_mode": (
                    [
                        "auto_compile (LLM编译并自动回填下方文本框)",
                        "manual_override (直接使用下方4个文本框，不调LLM)",
                        "compile_if_empty (文本框为空时调LLM，非空时用文本框)",
                    ],
                    {"default": "auto_compile (LLM编译并自动回填下方文本框)"},
                ),
                "stage1_positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Stage 1 正面提示词（可见/可直接手改；点上方按钮可一键预填）",
                    },
                ),
                "stage1_negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Stage 1 负面提示词（可见/可直接手改）",
                    },
                ),
                "stage2_positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Stage 2 视频动作正面提示词（可见/可直接手改）",
                    },
                ),
                "stage2_negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Stage 2 视频动作负面提示词（可见/可直接手改）",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, prompt=None, **kwargs):
        pose_node_state = ""
        if isinstance(prompt, dict):
            for nid, ndata in prompt.items():
                if isinstance(ndata, dict) and ndata.get("class_type") == "AsepriteNSFWPoseLibrary":
                    pose_node_state += json.dumps(ndata.get("inputs", {}), sort_keys=True, ensure_ascii=False)
        sig = json.dumps({k: str(v) for k, v in kwargs.items() if k != "reference_image"}, sort_keys=True, ensure_ascii=False) + "|" + pose_node_state
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("anima_positive", "anima_negative", "wan_positive", "wan_negative")
    FUNCTION = "compile_prompts"
    OUTPUT_NODE = True
    CATEGORY = "pixel_art/aseprite"
    def _get_charcard_macros(self, charcard_choice: str) -> dict:
        none_macros = {"anima_tags": "", "neg_tags": "", "wan_subject": "the character"}
        if not charcard_choice or charcard_choice.startswith("none"):
            return none_macros
        disk_path = os.path.join(PROJECT_ROOT, "scripts", charcard_choice.split(" ")[0].strip())
        if os.path.exists(disk_path):
            try:
                data = json.load(open(disk_path, encoding="utf-8"))
                return {
                    "anima_tags": data.get("anima_nsfw_tags") or data.get("anima_tags") or "",
                    "neg_tags": data.get("negative_tags") or "panties, underwear, shorts, bra",
                    "wan_subject": data.get("wan_subject") or data.get("card") or "the character",
                }
            except Exception:
                pass
        return none_macros

    def _extract_text_from_response(self, resp_json: dict) -> str:
        if not isinstance(resp_json, dict):
            return ""
        # 1. OpenAI /chat/completions format
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")]
                if parts:
                    return "\n".join(parts).strip()
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(reasoning, str) and "{" in reasoning and "}" in reasoning:
                return reasoning.strip()
        # 2. OpenAI /responses format
        if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
            return resp_json["output_text"].strip()
        for item in resp_json.get("output") or []:
            if isinstance(item, dict):
                for c in item.get("content") or []:
                    if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                        return c["text"].strip()
        return ""

    def _tensor_to_jpeg_data_url(self, img_tensor, max_side: int = 512) -> str:
        arr = (img_tensor[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        pil_im = Image.fromarray(arr).convert("RGB")
        w, h = pil_im.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            pil_im = pil_im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
        buf = io.BytesIO()
        pil_im.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _request_llm_text(self, base_url: str, resolved_key: str, model: str, sys_text: str, user_msg: str, image_data_url: str = "") -> str:
        base = base_url.strip().rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-AsepritePromptDirector/1.0",
        }
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"

        errors = []
        prefer_responses = "argon" in model.lower() or model.lower().endswith("-high")
        endpoints = ["responses", "chat"] if prefer_responses else ["chat", "responses"]

        # Try with image first (if provided), then automatically retry text-only if multimodal safety filter returns empty
        img_candidates = [image_data_url, ""] if image_data_url else [""]
        for img_url in img_candidates:
            for ep in endpoints:
                try:
                    if ep == "chat":
                        url = f"{base}/chat/completions"
                        payload = {
                            "model": model.strip(),
                            "messages": [
                                {"role": "system", "content": sys_text},
                                {
                                    "role": "user",
                                    "content": (
                                        [
                                            {"type": "image_url", "image_url": {"url": img_url}},
                                            {"type": "text", "text": user_msg},
                                        ]
                                        if img_url
                                        else user_msg
                                    ),
                                },
                            ],
                            "temperature": 0.2,
                        }
                        if model.lower().endswith("-high") or "argon" in model.lower():
                            payload["reasoning_effort"] = "low"
                    else:
                        url = f"{base}/responses"
                        payload = {
                            "model": model.strip(),
                            "instructions": sys_text,
                            "input": (
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "input_image", "image_url": img_url},
                                            {"type": "input_text", "text": user_msg},
                                        ],
                                    }
                                ]
                                if img_url
                                else user_msg
                            ),
                        }
                        if model.lower().endswith("-high") or "argon" in model.lower():
                            payload["reasoning"] = {"effort": "low"}
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                    raw_resp = urllib.request.urlopen(req, timeout=35).read().decode("utf-8")
                    resp_json = json.loads(raw_resp)
                    text = self._extract_text_from_response(resp_json)
                    if text and re.search(r"\{[\s\S]*\}", text):
                        if img_url and ("hair" not in text.lower() and "eyes" not in text.lower()):
                            errors.append(f"{ep} did not extract character appearance tags from image")
                            continue
                        return text
                    elif text:
                        errors.append(f"{ep} returned non-JSON: {text[:120]}")
                    else:
                        errors.append(f"{ep}{'(img)' if img_url else ''} returned empty content")
                except Exception as e:
                    errors.append(f"{ep}{'(img)' if img_url else ''} error: {e}")
        raise RuntimeError(f"模型 {model} 未返回有效 JSON ({'; '.join(errors)})")

    @staticmethod
    def _sanitize_cn_for_cloud_llm(text: str) -> str:
        """Replaces explicit Chinese NSFW phrases with structural pose phrases before calling cloud LLM
        so Gemini safety filters never return empty choices=[], while preserving pose/hand/motion semantics.
        """
        s = str(text or "")
        replacements = [
            ("蹲姿骑乘的姿势并起伏", "正面深蹲姿势"),
            ("蹲姿骑乘", "正面深蹲姿势"),
            ("女上位骑乘", "正面深蹲姿势"),
            ("骑乘", "深蹲"),
            ("下身不穿裤子完全裸露", "露腿短装"),
            ("不穿裤子完全裸露", "露腿短装"),
            ("下身完全裸露", "露腿短装"),
            ("完全裸露", "轻便衣装"),
            ("全裸", "轻便衣装"),
            ("裸体", "轻便衣装"),
            ("赤裸", "轻便衣装"),
            ("不穿内裤", "轻便短装"),
            ("不穿裤子", "露腿短装"),
            ("露出胸部与乳头", "敞开外套"),
            ("害羞脸红", "神情柔和微红"),
            ("露点", ""),
            ("乳头", ""),
            ("私处", ""),
            ("小穴", ""),
            ("阴部", ""),
            ("肉棒", ""),
            ("做爱", "互动"),
            ("交合", "贴近互动"),
            ("抽插", "上下律动"),
            ("抽送", "前后律动"),
            ("后入式", "背后环抱姿态"),
            ("后入", "背后姿态"),
            ("娇喘", "轻微呼吸"),
            ("双手撑在大腿上", "双臂向两侧张开、双手搭在外侧膝盖边缘（远离腰腹中线）"),
            ("双手放在大腿上", "双臂向两侧张开、双手搭在外侧膝盖边缘（远离腰腹中线）"),
            ("手撑在大腿上", "双臂向两侧张开、双手搭在外侧膝盖边缘"),
            ("手扶大腿", "双手搭在外侧膝盖边缘"),
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        return s

    @staticmethod
    def _parse_cn_pose_and_motion(user_prompt: str, pose_meta: dict) -> tuple:
        """Extracts pose description, Danbooru tags, and Wan2.2 motion from Chinese user_prompt + pose_meta."""
        up = str(user_prompt or "")
        pose_a = pose_meta.get("anima_pose_tags") or "standing, looking at viewer, gentle smile"
        pose_q = pose_meta.get("qwen_pose_prompt") or "standing facing forward with a gentle expression, entire full body from head to feet inside the frame with white margins"
        pose_w = pose_meta.get("wan_motion_prompt") or "gently breathing with subtle chest and shoulder rise and fall"

        # Only override pose_q / pose_a from Chinese text when no preset skeleton was selected (preserves duo_* and all preset skeletons!)
        # 1. Parse custom hand actions globally from user prompt
        custom_hands_q = []
        custom_hands_a = []
        # Hand to mouth / oral / blowjob
        if any(k in up for k in ("嘴前", "含入", "口交", "含住", "握住肉棒", "举在嘴前", "双手在嘴", "吞吐", "嘴唇", "吹箫", "深喉")):
            custom_hands_q.append("both hands raised in front of her mouth holding the penis shaft")
            custom_hands_a.append("hands raised, holding penis, fellatio, oral, blowjob, penis in mouth")
        elif any(k in up for k in ("抬手", "举手", "双手抬起", "举起双手", "手抬起来", "挥手", "招手", "双手举高", "高举双手")):
            custom_hands_q.append("both arms raised straight up high into the air with open palms waving")
            custom_hands_a.append("arms up, hands raised, waving")
        elif any(k in up for k in ("捂嘴", "捂着嘴", "遮嘴", "衣袖捂嘴")):
            custom_hands_q.append("one hand raised covering own mouth shyly")
            custom_hands_a.append("covering own mouth, hand over mouth, shy")

        # Hand to breast
        if any(k in up for k in ("揉胸", "摸胸", "托胸", "抓胸", "抚摸胸部", "放在胸前", "放在胸口", "双乳")):
            if "左手" in up and "右手" not in up:
                custom_hands_q.append("left hand placed over chest fondling own breast")
                custom_hands_a.append("left hand on own breast, fondling own breast")
            elif "右手" in up and "左手" not in up:
                custom_hands_q.append("right hand placed over chest fondling own breast")
                custom_hands_a.append("right hand on own breast, fondling own breast")
            else:
                custom_hands_q.append("one hand placed over chest fondling own breast")
                custom_hands_a.append("hand on own breast, fondling own breast")

        # Hand to crotch / masturbation
        if any(k in up for k in ("伸入下体", "摸下体", "放在裆部", "自慰", "抠逼", "扣逼", "手指插入", "触碰私处", "手在裆", "触摸私处")):
            if "右手" in up and "左手" not in up:
                custom_hands_q.append("right hand at pelvis center touching herself")
                custom_hands_a.append("fingering, masturbation, hand between legs, touching own pussy")
            elif "左手" in up and "右手" not in up:
                custom_hands_q.append("left hand at pelvis center touching herself")
                custom_hands_a.append("fingering, masturbation, hand between legs, touching own pussy")
            else:
                custom_hands_q.append("hand placed between legs at pelvis center touching herself")
                custom_hands_a.append("fingering, masturbation, hand between legs, touching own pussy")

        # Other hand actions
        if any(k in up for k in ("叉腰", "手叉腰", "双手叉腰")):
            custom_hands_q.append("both hands placed firmly on her hips")
            custom_hands_a.append("hands on hips")
        elif any(k in up for k in ("抱胸", "双手抱胸", "抱住自己")):
            custom_hands_q.append("arms crossed over her chest")
            custom_hands_a.append("arms crossed")
        elif any(k in up for k in ("抱头", "双手抱头", "枕在脑后")):
            custom_hands_q.append("hands clasped behind her head")
            custom_hands_a.append("hands behind head")
        elif any(k in up for k in ("抚摸大腿", "摸大腿", "扶大腿", "双手放在大腿", "手在大腿")):
            custom_hands_q.append("both hands resting gently on her thighs")
            custom_hands_a.append("hands on own thighs")
        elif any(k in up for k in ("撑地", "双手撑地", "扶地", "手在地上")):
            custom_hands_q.append("both hands resting on the floor supporting her body")
            custom_hands_a.append("hands on floor, supporting self")
        elif any(k in up for k in ("背手", "手在背后", "双手放在身后")):
            custom_hands_q.append("arms behind her back")
            custom_hands_a.append("arms behind back")
        elif any(k in up for k in ("自然下垂", "垂在身体两侧", "两手垂放", "手放在两侧")):
            custom_hands_q.append("both arms resting naturally at her sides")
            custom_hands_a.append("arms at sides")

        # 2. If a preset skeleton exists, integrate custom hands smoothly
        if pose_meta:
            if custom_hands_q:
                for h_pat in [
                    r",? both arms placed outward to the sides with hands resting on the floor beside her outer hips away from her body",
                    r",? with hands resting on the floor beside her outer hips away from her body",
                    r",? with both hands resting on the floor outside her hips for support",
                    r",? with hands resting on the floor",
                    r",? arms at sides",
                ]:
                    pose_q = re.sub(h_pat, "", pose_q, flags=re.IGNORECASE)
                pose_q = f"{pose_q}, with {', and '.join(custom_hands_q)}"
                for a_pat in [r"\barms at sides\b,?", r"\bhands on floor\b,?", r"\bhands on outer knees\b,?"]:
                    pose_a = re.sub(a_pat, "", pose_a, flags=re.IGNORECASE).strip(" ,")
                pose_a = f"{pose_a}, {', '.join(custom_hands_a)}"
        else:
            # No preset skeleton: build complete pose from prompt keywords
            if any(k in up for k in ("蹲", "骑乘")):
                if custom_hands_q:
                    hand_q = f"with {', and '.join(custom_hands_q)}"
                    hand_a = f"{', '.join(custom_hands_a)}"
                else:
                    hand_q = "with both arms angled wide outward and hands resting on the outer sides of her knees far away from her crotch"
                    hand_a = "arms spread outward, hands on outer knees"
                pose_q = f"in a balanced front-view wide-legged squat with knees spread apart {hand_q}, natural 2D anime body proportions without extreme foreshortening, entire full body from head to feet inside the frame with white margins"
                pose_a = f"1girl, solo, squatting, wide spread legs, {hand_a}, full frontal, balanced proportions"
            elif any(k in up for k in ("M字", "m字", "坐在地上", "坐姿", "坐")):
                if custom_hands_q:
                    hand_q = f"with {', and '.join(custom_hands_q)}"
                    hand_a = f"{', '.join(custom_hands_a)}"
                else:
                    hand_q = "with both hands resting on the floor beside her outer hips away from her crotch"
                    hand_a = "arms at sides, hands on floor"
                pose_q = f"sitting on the floor in an M-shaped spread-legs pose (m legs) facing forward {hand_q}, entire full body inside the frame with white margins"
                pose_a = f"1girl, solo, sitting, sitting on floor, m legs, spread legs, knees up, {hand_a}, full frontal"
            elif any(k in up for k in ("跪", "挺胸")):
                if custom_hands_q:
                    hand_q = f"with {', and '.join(custom_hands_q)}"
                    hand_a = f"{', '.join(custom_hands_a)}"
                    pose_q = f"kneeling on both knees facing forward with her head slightly tilted up {hand_q}, entire full body inside the frame"
                    pose_a = f"1girl, solo, kneeling, looking up, {hand_a}, full frontal"
                else:
                    pose_q = "kneeling on both knees facing forward, entire full body inside the frame"
                    pose_a = "1girl, solo, kneeling, full frontal"
            elif any(k in up for k in ("趴", "四肢着地", "后入")):
                pose_q = "on all fours on the floor with her hips raised high and head turned to look back at the viewer"
                pose_a = "1girl, solo, all fours, doggy style, top-down bottom-up, looking back, arched back"
            elif any(k in up for k in ("仰卧", "躺")):
                if custom_hands_q:
                    hand_q = f"with {', and '.join(custom_hands_q)}"
                    hand_a = f"{', '.join(custom_hands_a)}"
                else:
                    hand_q = "hands holding the backs of her outer knees"
                    hand_a = "grabbing own knees"
                pose_q = f"lying on her back with both legs raised and spread wide apart, {hand_q}"
                pose_a = f"1girl, solo, on back, lying, spread legs, legs up, {hand_a}"
            else:
                # Standing or default pose
                if custom_hands_q:
                    hand_q = f"with {', and '.join(custom_hands_q)}"
                    hand_a = f"{', '.join(custom_hands_a)}"
                    pose_q = f"standing facing forward {hand_q}, entire full body inside the frame"
                    pose_a = f"1girl, solo, standing, {hand_a}, full frontal"
                else:
                    pose_q = "standing facing forward with a gentle expression, entire full body from head to feet inside the frame with white margins"
                    pose_a = "1girl, solo, standing, full frontal"
        has_oral_motion = any(k in up for k in ("口交", "含入", "含住", "嘴前", "吞吐", "吹箫", "前后吞吐", "头部前后"))
        has_active_motion = any(k in up for k in ("并起伏", "上下起伏", "骑乘", "弹跳", "律动", "抽插", "抽送", "摇晃", "摆动", "挺腰"))
        has_riding_motion = any(k in up for k in ("骑乘", "女上位", "上下起伏", "起伏上下", "骑乘位", "挺腰起伏")) or any(k in pose_meta.get("id", "") for k in ("cowgirl", "riding", "squat"))
        is_default_breath_suffix = up.endswith("动作：缓慢呼吸，胸口与肩膀轻微起伏")
        if has_riding_motion and (is_default_breath_suffix or any(k in up.split("动作")[-1] for k in ("起伏", "骑乘", "弹跳", "律动", "抽插", "摇晃")) or pose_meta.get("id", "").startswith("duo_cowgirl")):
            pose_w = "large amplitude riding motion, intense deep rhythmic up-and-down pelvic bouncing, smoothly lifting her hips up and driving them down with full range of motion, pronounced vertical stroke, vigorous hip bounce and chest bounce, expressive dynamic sexual motion, full body continuous smooth riding"
        elif has_oral_motion:
            pose_w = "large amplitude rhythmic oral sex motion, vigorous pronounced head and torso bobbing deeply back and forth along the penis shaft, head thrusting forward and backward with full range of motion, mouth sliding all the way up and down the shaft, dynamic expressive movement"
        elif has_active_motion and (is_default_breath_suffix or any(k in up.split("动作")[-1] for k in ("起伏", "骑乘", "弹跳", "律动", "抽插", "摇晃"))):
            if pose_meta.get("wan_motion_prompt") and any(k in pose_meta.get("id", "") for k in ("squat", "cowgirl", "doggy", "arch", "mating", "standing_behind")):
                pose_w = f"large amplitude motion, {pose_meta['wan_motion_prompt']}"
            else:
                pose_w = "large amplitude riding motion, smoothly bouncing her hips, pelvis, and torso up and down in a pronounced rhythmic riding motion with natural knee flexion and chest bounce"
        elif "缓慢呼吸" in up and not has_active_motion:
            pose_w = "breathing slowly with her chest and shoulders gently rising and falling"
        extra_tags = []
        extra_qwen = []
        if any(k in up for k in ("不穿裤", "脱下短裤", "下身裸露", "完全裸露", "全裸", "裸体")):
            if "bottomless" not in pose_a:
                extra_tags.append("bottomless, no panties, exposed pussy, uncensored")
            if "bottomless" not in pose_q:
                extra_qwen.append("lower body completely bare and bottomless without shorts or panties")
        if any(k in up for k in ("敞开", "解开", "露出胸", "上身裸", "乳头")):
            if "breasts out" not in pose_a:
                extra_tags.append("open clothes, breasts out, nipples, uncensored")
        if any(k in up for k in ("脸红", "害羞", "娇羞")):
            if "blush" not in pose_a:
                extra_tags.append("shy, blush, embarrassed")
        if extra_tags:
            pose_a = f"{pose_a}, {', '.join(extra_tags)}"
        if extra_qwen:
            pose_q = f"{pose_q}, {', '.join(extra_qwen)}"
        return pose_a, pose_q, pose_w

    def build_local_template(self, charcard: str, pose_preset: str = "none (不使用预设骨骼)", user_prompt: str = "", wd14_override: dict = None, custom_pose_name: str = "") -> tuple:
        macros = dict(wd14_override) if wd14_override else dict(self._get_charcard_macros(charcard))
        pose_id = (pose_preset or "none").split(" ")[0].strip()
        pose_meta = _load_pose_library().get(pose_id, {}) if pose_id != "none" else {}
        pose_a, pose_q, pose_w = self._parse_cn_pose_and_motion(user_prompt, pose_meta)

        is_duo = pose_id.startswith("duo_") or any(k in str(user_prompt) for k in ("双人", "两人", "男女", "两个女孩", "1boy", "2girls"))
        if is_duo:
            macros["anima_tags"] = re.sub(r"\bsolo\b\s*,?", "", macros.get("anima_tags", "")).strip(" ,")
        neg_multi = "" if is_duo else "multiple girls, 2girls, "

        # Enforce character body build & breast size consistency to prevent NSFW LoRA OOC distortion
        is_petite = any(k in str(macros).lower() for k in ("small breasts", "flat chest", "slender", "petite", "hiro"))
        if is_petite:
            pose_q = re.sub(r"\b(hanging breasts|large breasts|huge breasts|cleavage|bare ass)\b\s*,?", "", pose_q, flags=re.IGNORECASE)
            pose_a = re.sub(r"\b(hanging breasts|large breasts|huge breasts|cleavage|ass)\b\s*,?", "", pose_a, flags=re.IGNORECASE)
            if "small breasts" not in macros.get("anima_tags", "").lower():
                macros["anima_tags"] = f"small breasts, flat chest, slender, narrow hips, pink nipples, {macros['anima_tags']}"
            if "large breasts" not in macros.get("neg_tags", "").lower():
                macros["neg_tags"] = f"large breasts, huge breasts, hanging breasts, cleavage, wide hips, voluptuous, curvy, thick thighs, fat, plump, big ass, {macros['neg_tags']}"
            if "small" not in pose_q.lower() and "flat" not in pose_q.lower():
                pose_q = f"{pose_q}, slender flat-chested petite build with small flat breasts and narrow hips without large breast mounds"
        # Deduplicate tags to avoid token budget bloat
        raw_tokens = [t.strip() for t in f"{macros['anima_tags']}, {pose_a}, full body, white background, simple background".split(",") if t.strip()]
        dedup_tags = ", ".join(dict.fromkeys(raw_tokens))

        has_pose_guide = (pose_id != "none" and bool(pose_meta)) or bool(custom_pose_name and custom_pose_name.strip())
        if has_pose_guide:
            if is_duo:
                anima_pos = (
                    f"Picture 1 is the main female character ({macros['wan_subject']}). "
                    f"Picture 2 is a dual-character skeleton pose guide showing two interacting characters. "
                    f"Redraw the 2D anime girl from Picture 1 together with a faceless 2D anime male partner in the exact interaction of Picture 2: {pose_q}. "
                    f"Keep the female character's exact 2D anime face, hair, eyes, hair ornaments, and identity from Picture 1. "
                    f"Both characters are drawn in 100% unified 2D cel-shaded anime illustration style with clean lineart. "
                    f"Plain white background. "
                    f"masterpiece, best quality, score_9, score_8, newest, 2d, cel shading, clean lineart, {dedup_tags}"
                ).strip(" ,")
            else:
                anima_pos = (
                    f"Picture 1 is the character reference ({macros['wan_subject']}). "
                    f"Picture 2 is a skeleton pose guide. "
                    f"Redraw the character from Picture 1 in the exact pose of Picture 2: {pose_q}. "
                    f"Keep her exact 2D anime face, hair, eyes, hair ornaments, and identity from Picture 1. "
                    f"Plain white background, clean anime illustration style. "
                    f"masterpiece, best quality, score_9, score_8, newest, 2d, cel shading, clean lineart, {dedup_tags}"
                ).strip(" ,")
        else:
            anima_pos = (
                f"2D flat cel-shaded anime illustration, clean anime lineart. "
                f"Keep the exact 2D anime character identity, face, hair, and accessories from image 1 ({macros['wan_subject']}). "
                f"Redraw her: {pose_q}. "
                f"masterpiece, best quality, score_9, score_8, newest, 2d, cel shading, clean lineart, {dedup_tags}"
            ).strip(" ,")

        up_str = str(user_prompt or "")
        has_crotch_touch = any(k in up_str for k in ("自慰", "摸下体", "伸入下体", "抠逼", "扣逼", "手在裆", "触摸私处", "触碰私处", "手摸私处")) or ("fingering" in anima_pos) or ("masturbation" in anima_pos)
        has_hand_action = has_crotch_touch or any(k in up_str for k in ("手", "摸", "抚", "握", "揉", "抓", "自慰", "扣", "抠", "抱", "叉腰", "抬手", "举手"))
        neg_hand_filter = "" if has_hand_action else "hands between legs, hands covering crotch, hands on inner thighs, "
        anima_neg = f"worst quality, low quality, score_1, score_2, score_3, realistic, 3d, blurry, jpeg artifacts, sepia, {neg_multi}cropped, out of frame, bad anatomy, extra limbs, {neg_hand_filter}{macros['neg_tags']}".strip(" ,")
        wan_pos = f"anime style, {macros['wan_subject']} {pose_q.split(',')[0]} on a plain white background, {pose_w}, static camera, fixed composition, full body, clean 2D anime illustration".strip(" ,")
        wan_neg = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，camera movement, zoom, morphing, warping, changing clothes"
        return (anima_pos, anima_neg, wan_pos, wan_neg)

    def _compile_core(self, base_url, api_key, model, charcard, user_prompt, system_prompt, reference_image=None, pose_preset="none (不使用预设骨骼)", custom_pose_name=""):
        ref_hash = ""
        image_data_url = ""
        wd14_info = None
        wd14_override = None
        macros = dict(self._get_charcard_macros(charcard))
        if reference_image is not None:
            try:
                from scripts.extract_charcard import run_local_wd14
                arr_ref = (reference_image[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
                wd14_info = run_local_wd14(Image.fromarray(arr_ref).convert("RGB"))
                # Auto-detect if user swapped LoadImage to a character that doesn't match the selected charcard preset's hair color!
                id_tags = wd14_info.get("identity_tags", [])
                ref_hairs = [t for t in id_tags if t.endswith("_hair") and t not in ("long_hair", "very_long_hair", "short_hair", "medium_hair", "straight_hair", "wavy_hair", "twintails", "ponytail")]
                card_tags_lower = macros.get("anima_tags", "").lower()
                hair_mismatch = bool(ref_hairs) and not any(rh.replace("_", " ") in card_tags_lower for rh in ref_hairs)
                if not charcard or charcard.startswith("none") or hair_mismatch:
                    image_data_url = self._tensor_to_jpeg_data_url(reference_image, max_side=512)
                    ref_hash = hashlib.md5(image_data_url.encode("ascii")).hexdigest()[:12]
                    clean_id = [t.replace("_", " ") for t in id_tags if not any(w in t for w in ("shorts", "panties", "underwear", "skirt", "pants", "dress", "bikini", "bra"))]
                    wd14_override = {
                        "anima_tags": "1girl, solo, " + ", ".join(clean_id[:18]) + ", breasts out, nipples, bottomless, no panties, pussy, uncensored",
                        "neg_tags": "panties, underwear, shorts, bra, censored",
                        "wan_subject": "a 2D anime girl (" + ", ".join(clean_id[:8]) + ")",
                    }
                    macros = dict(wd14_override)
                    if hair_mismatch:
                        print(f"[AsepriteLLMPromptDirector] Detected new reference image hair ({ref_hairs}); auto-adapting character tags from reference_image.")
            except Exception as e:
                print(f"[AsepriteLLMPromptDirector] Local WD14 hint warning: {e}")
        pose_id = (pose_preset or "none").split(" ")[0].strip()
        pose_meta = _load_pose_library().get(pose_id, {}) if pose_id != "none" else {}
        is_duo = pose_id.startswith("duo_") or any(k in str(user_prompt) for k in ("双人", "两人", "男女", "两个女孩", "1boy", "2girls"))
        if is_duo:
            macros["anima_tags"] = re.sub(r"\bsolo\b\s*,?", "", macros.get("anima_tags", "")).strip(" ,")
        card_path = os.path.join(PROJECT_ROOT, "scripts", (charcard or "").split(" ")[0].strip())
        card_sig = hashlib.md5(open(card_path, "rb").read()).hexdigest()[:12] if charcard and not charcard.startswith("none") and os.path.isfile(card_path) else ""
        cache_key = hashlib.sha256(
            f"{base_url}|{model}|{charcard}|{card_sig}|{user_prompt}|{ref_hash}|{pose_id}|v8".encode("utf-8")
        ).hexdigest()
        if cache_key in _PROMPT_CACHE:
            print("[AsepriteLLMPromptDirector] Reusing cached compiled prompts.")
            return _PROMPT_CACHE[cache_key]

        resolved_key = _resolve_api_key(api_key)
        pose_a_local, pose_q_local, pose_w_local = self._parse_cn_pose_and_motion(user_prompt, pose_meta)

        # If user selected an active motion or wrote "并起伏/骑乘" while leaving the default "动作：缓慢呼吸..." suffix, replace suffix so LLM compiles active motion
        up_for_llm = user_prompt
        if any(k in str(user_prompt) for k in ("并起伏", "上下起伏", "骑乘", "抽插", "弹跳", "律动")) and str(user_prompt).endswith("动作：缓慢呼吸，胸口与肩膀轻微起伏"):
            up_for_llm = str(user_prompt).replace("动作：缓慢呼吸，胸口与肩膀轻微起伏", "动作：身体与臀部明显上下起伏律动，膝盖随动作弯曲伸展，胸部自然晃动")
        clean_up = self._sanitize_cn_for_cloud_llm(up_for_llm)
        prompt_with_pose = clean_up
        if pose_meta and pose_meta.get("label"):
            clean_lbl = self._sanitize_cn_for_cloud_llm(pose_meta.get("label"))
            prompt_with_pose = f"{clean_up}\n(参考骨骼姿态: {clean_lbl})"

        if image_data_url:
            sys_text = (
                "你是一个专精于二次元角色立绘 (Danbooru Tag) 与图生视频动画 (Wan2.2 I2V) 的多模态视觉提示词编译器。\n"
                "用户上传了一张【角色参考图】并给出了【中文姿势与动作要求】。\n"
                "请观察参考图中的角色，提取她最核心的标志性外观特征（1girl, solo, 发色, 发长, 发型, 标志性兽耳/角/尾巴, 核心发饰/发卡, 瞳色, 身材特征，以及极简的保留服饰颜色款式）。\n"
                "【正向关键词精简而有效原则（防肢体崩坏）】：在进行脱衣、暴露或姿势变换时，严禁提取原立绘里繁复的战术背带、腰包、过膝袜、战斗靴、手套、护膝或武器！正面关键词必须精简聚焦在面容、发型与动作（控制在 30 词以内），避免注意力分散导致肢体变形！\n"
                "然后结合用户的中文姿势、表情与动作描述，严格输出以下 JSON 格式（不要输出任何额外解释）：\n"
                "{\n"
                '  "char_appearance_tags": "1girl, solo, <从参考图提取的完整Danbooru角色外观、发型发色、瞳色、发饰与服饰英文标签>",\n'
                '  "char_wan_subject": "<用英文自然语言详细描述参考图中角色的外貌特征>",\n'
                '  "anima_positive": "masterpiece, best quality, score_9, score_8, newest, <char_appearance_tags全部标签>, <根据用户描述转换的姿势/肢体位置/表情/视角Danbooru英文标签>, full body, white background, simple background, character in center with margins around body",\n'
                '  "anima_negative": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, multiple girls, 2girls, cropped, out of frame, bad anatomy, extra limbs",\n'
                '  "wan_positive": "anime style, <char_wan_subject> <用英文自然语言描述当前所处姿势与表情> on a plain white background, <用英文自然语言描述单程半周期柔和运动轨迹，不要堆砌tag>, static camera, fixed composition, full body, clean 2D anime illustration, smooth subtle motion",\n'
                '  "wan_negative": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，camera movement, zoom, morphing, warping, changing clothes"\n'
                "}"
            )
            wd14_anchor = ""
            if wd14_info and wd14_info.get("identity_tags"):
                wd14_anchor = f"\n【本地 WD14 视觉标签锚点（高置信度特征，请保留入 char_appearance_tags）】：{', '.join(wd14_info['identity_tags'])}"
            user_msg = (
                f"请观察附图中角色的外貌特征，并按以下要求编译 JSON 提示词：{wd14_anchor}\n"
                f"【用户姿势与动作要求】：\n{prompt_with_pose}"
            )
        else:
            sys_text = DEFAULT_SYSTEM_PROMPT
            user_msg = (
                f"【用户姿势与动作要求】：\n{prompt_with_pose}\n\n"
                f"请严格输出包含 anima_positive, anima_negative, wan_positive, wan_negative 四个键的 JSON 对象，"
                f"并在对应位置保留 {{CHARCARD_ANIMA_TAGS}}、{{CHARCARD_NEG_TAGS}}、{{CHARCARD_WAN_SUBJECT}} 宏占位符。"
            )

        try:
            content = self._request_llm_text(base_url, resolved_key, model, sys_text, user_msg, image_data_url=image_data_url)
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise RuntimeError(f"LLM 未返回 JSON 格式: {content}")
            parsed = json.loads(m.group(0))
        except Exception as e:
            print(f"[AsepriteLLMPromptDirector] Cloud LLM refused or returned empty ({e}); falling back to local charcard+pose compiler.")
            res = self.build_local_template(charcard, pose_preset, user_prompt=user_prompt, wd14_override=wd14_override)
            _PROMPT_CACHE[cache_key] = res
            return res

        extracted_tags = str(parsed.get("char_appearance_tags") or "").strip(" ,")
        extracted_subj = str(parsed.get("char_wan_subject") or "").strip()
        if not macros["anima_tags"] and extracted_tags:
            macros["anima_tags"] = extracted_tags
        if macros["wan_subject"] == "the character" and extracted_subj:
            macros["wan_subject"] = extracted_subj

        def _expand(s: str) -> str:
            out = (
                str(s or "")
                .replace("{{CHARCARD_ANIMA_TAGS}}", macros["anima_tags"])
                .replace("{{CHARCARD_NEG_TAGS}}", macros["neg_tags"])
                .replace("{{CHARCARD_WAN_SUBJECT}}", macros["wan_subject"])
                .replace("<char_appearance_tags>", macros["anima_tags"])
                .replace("<char_appearance_tags全部标签>", macros["anima_tags"])
                .replace("<char_wan_subject>", macros["wan_subject"])
                .replace("<角色固定外观与服装Danbooru标签>", macros["anima_tags"])
                .replace("<与当前服装状态冲突的标签>", macros["neg_tags"])
            )
            out = re.sub(r"\(\s*\)", "", out)
            out = re.sub(r",\s*,", ",", out)
            out = re.sub(r"\s{2,}", " ", out).strip(" ,")
            return out

        anima_pos = _expand(parsed.get("anima_positive", ""))
        anima_neg = _expand(parsed.get("anima_negative", ""))
        has_pose_guide = (pose_id != "none" and bool(pose_meta)) or bool(custom_pose_name and custom_pose_name.strip())
        is_single_image = not has_pose_guide
        if "picture 1" not in anima_pos.lower() and "image 1" not in anima_pos.lower():
            if is_single_image:
                anima_pos = (
                    f"2D flat cel-shaded anime illustration, clean anime lineart. "
                    f"Keep the exact 2D anime character identity, face, hair, and accessories from image 1 ({macros['wan_subject']}). "
                    f"Redraw her in the pose: {pose_q_local}. {anima_pos}"
                )
            elif is_duo:
                anima_pos = (
                    f"Picture 1 is the main female character ({macros['wan_subject']}). "
                    f"Picture 2 is a dual-character skeleton pose guide showing two interacting characters. "
                    f"Redraw the 2D anime girl from Picture 1 together with a faceless 2D anime male partner in the exact interaction of Picture 2: {pose_q_local}. "
                    f"Keep the female character's exact 2D anime face, hair, eyes, hair ornaments, and identity from Picture 1. "
                    f"Both characters are drawn in 100% unified 2D cel-shaded anime illustration style with clean lineart. "
                    f"Plain white background. {anima_pos}"
                )
            else:
                anima_pos = (
                    f"Picture 1 is the character reference ({macros['wan_subject']}). "
                    f"Picture 2 is a skeleton pose guide. "
                    f"Keep her exact 2D anime face, hair, eyes, hair ornaments, and identity from Picture 1, "
                    f"and redraw her in the pose of Picture 2 ({pose_q_local}). {anima_pos}"
                )
        elif is_single_image:
            anima_pos = re.sub(r"in the pose of Picture 2\s*(\([^\)]*\))?", "in the requested pose", anima_pos, flags=re.IGNORECASE)

        # Enforce body build / breast size consistency & prevent NSFW LoRA OOC explosion
        is_petite = any(k in str(macros).lower() for k in ("small breasts", "flat chest", "slender", "petite", "hiro"))
        if is_petite:
            anima_pos = re.sub(r"\b(large breasts|huge breasts|hanging breasts|cleavage|wide hips|curvy)\b\s*,?", "", anima_pos, flags=re.IGNORECASE)
            if "flat chest" not in anima_pos.lower():
                anima_pos = f"{anima_pos}, small breasts, flat chest, slender, narrow hips, pink nipples"
            if "large breasts" not in anima_neg.lower():
                anima_neg = f"large breasts, huge breasts, hanging breasts, cleavage, wide hips, voluptuous, curvy, thick thighs, fat, plump, big ass, {anima_neg}"
        # Clean deduplication
        raw_tokens = [t.strip() for t in anima_pos.split(",") if t.strip()]
        anima_pos = ", ".join(dict.fromkeys(raw_tokens))

        up_str = str(user_prompt or "")
        has_crotch_touch = any(k in up_str for k in ("自慰", "摸下体", "伸入下体", "抠逼", "扣逼", "手在裆", "触摸私处", "触碰私处", "手摸私处")) or ("fingering" in anima_pos) or ("masturbation" in anima_pos)
        has_hand_action = has_crotch_touch or any(k in up_str for k in ("手", "摸", "抚", "握", "揉", "抓", "自慰", "扣", "抠", "抱", "叉腰", "抬手", "举手"))

        if not has_crotch_touch and not has_hand_action:
            for bad_h, good_h in [
                ("with both hands resting on her thighs", "with both arms angled wide outward and hands resting on the outer sides of her knees far away from her crotch"),
                ("hands resting on her thighs", "hands resting on outer knee edges away from pelvis"),
                ("hands on own thighs", "arms spread outward, hands on outer knees"),
                ("hands on thighs", "arms spread outward, hands on outer knees"),
                ("grabbing own thighs", "hands on outer knees"),
            ]:
                anima_pos = anima_pos.replace(bad_h, good_h)

        if is_duo:
            anima_neg = re.sub(r"\b(multiple girls|2girls)\b\s*,?", "", anima_neg).strip(" ,")
        if has_hand_action or has_crotch_touch:
            anima_neg = re.sub(r"\b(hands between legs|hands covering crotch|hands on inner thighs)\b\s*,?", "", anima_neg, flags=re.IGNORECASE).strip(" ,")
        elif "hands covering crotch" not in anima_neg:
            anima_neg = f"{anima_neg}, hands between legs, hands covering crotch, hands on inner thighs"
        wan_pos = _expand(parsed.get("wan_positive", ""))
        if not wan_pos:
            wan_pos = f"anime style, {macros['wan_subject']} on a plain white background, {pose_w_local}, static camera, fixed composition, full body, clean 2D anime illustration"
        wan_neg = _expand(parsed.get("wan_negative", ""))
        has_dynamic = any(k in str(user_prompt) for k in ("口交", "含入", "含住", "嘴前", "吞吐", "吹箫", "前后吞吐", "起伏", "骑乘", "弹跳", "律动", "抽插", "抽送"))
        if has_dynamic and "motionless" not in wan_neg.lower():
            wan_neg = f"static, still, motionless, stationary, subtle, barely moving, small movement, slow motion, {wan_neg}"
        res = (anima_pos, anima_neg, wan_pos, wan_neg)
        _PROMPT_CACHE[cache_key] = res
        return res

    def compile_prompts(
        self,
        base_url,
        api_key,
        model,
        charcard,
        user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reference_image=None,
        pose_preset="none (不使用预设骨骼)",
        prompt_mode="auto_compile (LLM编译并自动回填下方文本框)",
        stage1_positive="",
        stage1_negative="",
        stage2_positive="",
        stage2_negative="",
        charcard_file="",
        prompt=None,
        **kwargs,
    ):
        llm_client = kwargs.get("llm_client")
        if llm_client is not None:
            base_url = getattr(llm_client, "base_url", base_url)
            api_key = getattr(llm_client, "api_key", api_key)
            if hasattr(llm_client, "get_effective_model"):
                model = llm_client.get_effective_model(model)
        # A connected AsepriteCharcardExtractor overrides when it emits a non-empty card filename
        if isinstance(charcard_file, str) and charcard_file.strip():
            charcard = charcard_file.strip()
        # A connected PoseSelector overrides when it emits a dynamic pose description
        custom_pose_name = kwargs.get("custom_pose_name")
        if isinstance(custom_pose_name, str) and custom_pose_name.strip():
            user_prompt = f"{user_prompt}，动态动作姿势：{custom_pose_name.strip()}"
        # 1. Self-heal if an un-refreshed browser tab sent shifted widgets (system_prompt -> pose_preset -> prompt_mode)
        if isinstance(system_prompt, str) and any(system_prompt.startswith(p) for p in ("solo_", "duo_", "none")):
            if isinstance(pose_preset, str) and any(pose_preset.startswith(m) for m in ("compile_if_empty", "auto_compile", "manual_override")):
                prompt_mode = pose_preset
            pose_preset = system_prompt
            system_prompt = DEFAULT_SYSTEM_PROMPT

        # 2. Always sync pose_preset with AsepriteNSFWPoseLibrary (Node #42) in the active prompt graph if present!
        if isinstance(prompt, dict):
            for nid, ndata in prompt.items():
                if isinstance(ndata, dict) and ndata.get("class_type") == "AsepriteNSFWPoseLibrary":
                    n_in = ndata.get("inputs", {})
                    if n_in.get("enable_pose_control", True) and n_in.get("pose_preset"):
                        pose_preset = n_in["pose_preset"]
                    elif not n_in.get("enable_pose_control", True):
                        pose_preset = "none (不使用预设骨骼)"
                    break

        mode_key = (prompt_mode or "auto_compile").split(" ")[0].strip()
        has_manual = any(s and s.strip() for s in (stage1_positive, stage1_negative, stage2_positive, stage2_negative))
        cur_sig = f"{charcard}|{pose_preset}|{user_prompt}"
        prev_sig = getattr(self.__class__, "_LAST_COMPILED_SIG", None)
        prev_prompts = getattr(self.__class__, "_LAST_COMPILED_PROMPTS", None)

        # If user changed charcard, pose_preset (on Node 42 or 33), or user_prompt, or if boxes match previous auto-output, re-compile!
        is_auto_filled_stale = (prev_prompts is not None and (stage1_positive.strip(), stage1_negative.strip(), stage2_positive.strip(), stage2_negative.strip()) == prev_prompts)
        inputs_changed = (prev_sig is not None and cur_sig != prev_sig)

        if mode_key == "manual_override" or (mode_key == "compile_if_empty" and has_manual and not inputs_changed and not is_auto_filled_stale):
            tpl = self.build_local_template(charcard, pose_preset, user_prompt=user_prompt, custom_pose_name=custom_pose_name)
            res = (
                stage1_positive.strip() if (stage1_positive and stage1_positive.strip()) else tpl[0],
                stage1_negative.strip() if (stage1_negative and stage1_negative.strip()) else tpl[1],
                stage2_positive.strip() if (stage2_positive and stage2_positive.strip()) else tpl[2],
                stage2_negative.strip() if (stage2_negative and stage2_negative.strip()) else tpl[3],
            )
            print(
                f"=== [AsepriteLLMPromptDirector (MANUAL OVERRIDE)] ===\n"
                f"[anima_positive]: {res[0]}\n"
                f"[anima_negative]: {res[1]}\n"
                f"[wan_positive]  : {res[2]}\n"
                f"[wan_negative]  : {res[3]}"
            )
            return {"ui": {"compiled_prompts": list(res)}, "result": res}

        res = self._compile_core(
            base_url, api_key, model, charcard, user_prompt, system_prompt,
            reference_image=reference_image, pose_preset=pose_preset, custom_pose_name=custom_pose_name,
        )
        self.__class__._LAST_COMPILED_SIG = cur_sig
        self.__class__._LAST_COMPILED_PROMPTS = tuple(s.strip() for s in res)
        print(
            f"=== [AsepriteLLMPromptDirector ({model} | pose={pose_preset.split(' ')[0]})] ===\n"
            f"[anima_positive]: {res[0]}\n"
            f"[anima_negative]: {res[1]}\n"
            f"[wan_positive]  : {res[2]}\n"
            f"[wan_negative]  : {res[3]}"
        )
        return {"ui": {"compiled_prompts": list(res)}, "result": res}

if hasattr(server, "PromptServer") and getattr(server.PromptServer, "instance", None) is not None:
    @server.PromptServer.instance.routes.post("/aseprite_llm/compile_preview")
    async def compile_llm_preview(request):
        director = AsepriteLLMPromptDirector()
        try:
            body = await request.json()
            mode = body.get("mode", "llm")
            charcard = body.get("charcard") or _CHARCARD_NONE
            pose_preset = body.get("pose_preset") or "none (不使用预设骨骼)"
            if mode == "template":
                res = director.build_local_template(charcard, pose_preset, user_prompt=body.get("user_prompt") or "")
                return web.json_response({"ok": True, "prompts": list(res), "source": "template"})
            res = director._compile_core(
                base_url=body.get("base_url") or DEFAULT_LLM_BASE_URL,
                api_key=body.get("api_key") or "",
                model=body.get("model") or "",
                charcard=charcard,
                user_prompt=body.get("user_prompt") or "",
                system_prompt=body.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
                reference_image=None,
                pose_preset=pose_preset,
            )
            return web.json_response({"ok": True, "prompts": list(res), "source": "llm"})
        except Exception as e:
            try:
                tpl = director.build_local_template(
                    body.get("charcard") or _CHARCARD_NONE,
                    body.get("pose_preset") or "none",
                    user_prompt=body.get("user_prompt") or "",
                )
            except Exception:
                tpl = ("", "", "", "")
            return web.json_response({"ok": False, "error": str(e), "prompts": list(tpl)})

def _purge_vram_and_vbars(*extra_patchers):
    """Unloads all models and purges comfy-aimdo ModelVBARs without aborting the active malloc_graph."""
    patchers = list(extra_patchers)
    for lm in list(getattr(comfy.model_management, "current_loaded_models", [])):
        if getattr(lm, "model", None) is not None:
            patchers.append(lm.model)

    comfy.model_management.unload_all_models()

    page_bytes = 32 * 1024 * 1024
    total_vbar_freed = 0
    seen_vbars = set()
    for p in patchers:
        if p is None:
            continue
        try:
            vbar = p._vbar_get() if hasattr(p, "_vbar_get") else None
            if vbar is None or id(vbar) in seen_vbars:
                continue
            seen_vbars.add(id(vbar))
            for _ in range(128):
                res = vbar.get_residency()
                pinned = [idx for idx, flag in enumerate(res) if (flag & 2)]
                if not pinned:
                    break
                for idx in pinned:
                    vbar.unpin(vbar.base_addr + idx * page_bytes, page_bytes)
            total_vbar_freed += vbar.free_memory(32 * 1024**3)
        except Exception as e:
            print(f"[_purge_vram_and_vbars] vbar free warning: {e}")

    gc.collect()
    comfy.model_management.soft_empty_cache()


_ORIG_COMFY_SAMPLE = getattr(comfy.sample, "_aseprite_orig_sample", comfy.sample.sample)
comfy.sample._aseprite_orig_sample = _ORIG_COMFY_SAMPLE

def _aseprite_guarded_sample(model, *args, **kwargs):
    """Prevents 16GB VRAM WDDM thrashing across QwenImage, Anima, WAN21, and WanVAE."""
    cur_cls = ""
    try:
        cur_cls = model.model.__class__.__name__ if hasattr(model, "model") else ""
        if any(k in cur_cls for k in ("Anima", "QwenImage", "WAN21", "Wan")):
            other_loaded = [
                lm.model
                for lm in getattr(comfy.model_management, "current_loaded_models", [])
                if getattr(lm, "model", None) is not None and lm.model is not model
            ]
            if other_loaded:
                other_names = [m.model.__class__.__name__ for m in other_loaded if hasattr(m, "model")]
                print(f"[VRAM Guard] Evicting leftover {other_names} vbars before sampling {cur_cls}...")
                _purge_vram_and_vbars(*other_loaded)
    except Exception as e:
        print(f"[VRAM Guard] pre-sample warning: {e}")

    res = _ORIG_COMFY_SAMPLE(model, *args, **kwargs)
    try:
        if any(k in cur_cls for k in ("WAN21", "Wan")):
            _purge_vram_and_vbars(model)
            print(f"[VRAM Guard] Freed {cur_cls} vbars immediately before VAEDecodeTiled.")
    except Exception as e:
        print(f"[VRAM Guard] post-sample warning: {e}")
    return res


comfy.sample.sample = _aseprite_guarded_sample
class QwenEditFastConditioning:
    """Fast I2I conditioning node for Qwen-Image-Edit-2511 on 16GB VRAM.

    Fixes three major bottlenecks in standard TextEncodeQwenImageEditPlus:
    1. Runs Qwen2.5-VL-7B (8.74 GiB) only ONCE (creates negative via zero-out for CFG=1.0 4-step Lightning).
    2. Scales reference_latents to ref_max_side (default 512 instead of hardcoded 1024x1024), cutting
       reference latent token count by ~4x across all 60 transformer blocks.
    3. Sets reference_latents_method='index_timestep_zero' and immediately unloads Qwen2.5-VL-7B from
       16GB VRAM before KSampler loads the 20B Qwen-Image-Edit UNet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 16}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1536, "step": 16}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL")
    RETURN_NAMES = ("positive", "negative", "latent", "model")
    FUNCTION = "encode_fast"
    CATEGORY = "pixel_art/aseprite"

    def encode_fast(
        self,
        clip,
        vae,
        image1,
        prompt,
        width=512,
        height=768,
        ref_max_side=576,
        unload_vl_after_encode=True,
        image2=None,
        model=None,
        **kwargs,
    ):
        if unload_vl_after_encode:
            # Purge any previous QwenImage (11GB) or Wan2.2 (15GB) dynamic vbars before loading Qwen2.5-VL-7B (7.9GB)
            _purge_vram_and_vbars()
        ref_latents = []
        images_vl = []
        llama_template = (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the "
            "user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        image_prompt = ""

        for i, img in enumerate([image1, image2]):
            if img is None:
                continue
            samples = img[:1].movedim(-1, 1)
            h_in, w_in = samples.shape[2], samples.shape[3]
            is_pose_skeleton = (i == 1) and (float(samples.mean().item()) < 0.18)

            # If image2 is a black-background OpenPose skeleton, convert black bg to white for Qwen2.5-VL
            # and DO NOT add it to VAE reference_latents (prevents black latent pollution of background/colors!)
            samples_vl = samples
            if is_pose_skeleton:
                s_hwc = samples.movedim(1, -1).clone()
                bg_m = s_hwc[:, :, :, :3].max(dim=-1, keepdim=True)[0] < 0.08
                s_hwc[:, :, :, :3] = torch.where(bg_m, torch.ones_like(s_hwc[:, :, :, :3]), s_hwc[:, :, :, :3])
                samples_vl = s_hwc.movedim(-1, 1)

            # 1. Vision tower input (~504x504 area, aligned to 28px ViT patch grid for sharper identity retention)
            scale_vl = math.sqrt((504 * 504) / (w_in * h_in))
            w_vl = max(28, round(w_in * scale_vl / 28.0) * 28)
            h_vl = max(28, round(h_in * scale_vl / 28.0) * 28)
            s_vl = comfy.utils.common_upscale(samples_vl, w_vl, h_vl, "area", "disabled")
            images_vl.append(s_vl.movedim(1, -1)[:, :, :, :3])

            # 2. VAE reference latent (uses white-bg skeleton samples_vl so pose keypoints align spatially without dark bg pollution)
            scale_ref = min(1.0, float(ref_max_side) / float(max(w_in, h_in)))
            w_ref = max(64, round(w_in * scale_ref / 16.0) * 16)
            h_ref = max(64, round(h_in * scale_ref / 16.0) * 16)
            s_ref = comfy.utils.common_upscale(samples_vl, w_ref, h_ref, "area", "disabled")
            ref_latents.append(vae.encode(s_ref.movedim(1, -1)[:, :, :, :3]))
            image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        positive = clip.encode_from_tokens_scheduled(tokens)

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive,
                {
                    "reference_latents": ref_latents,
                    "reference_latents_method": "index_timestep_zero",
                },
                append=True,
            )

        # Zero out negative conditioning for CFG=1.0 4-step Lightning (avoids 2nd VL + VAE pass)
        negative = []
        for t in positive:
            d = t[1].copy()
            pooled = d.get("pooled_output", None)
            if pooled is not None:
                d["pooled_output"] = torch.zeros_like(pooled)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            negative.append([torch.zeros_like(t[0]), d])

        if unload_vl_after_encode:
            _purge_vram_and_vbars(getattr(clip, "patcher", None), getattr(vae, "patcher", None))
            print("[QwenEditFastConditioning] Unloaded Qwen2.5-VL-7B + vbars from VRAM before Qwen-Edit KSampler.")

        if model is not None and hasattr(model, "model"):
            # QwenImage.memory_required overestimates CFG=2 BF16 activation memory (~8GB),
            # and comfy-aimdo detours cuMemGetInfo to cap single-shot free VRAM at ~8.8GB when loaded_memory=0.
            # Capping memory_required at 1.2GB (and preloading non-dynamic GGUF models with force_full_load=True)
            # ensures all weights stay 100% resident in 16GB VRAM.
            model.model.memory_required = lambda *a, **kw: 1200 * 1024 * 1024
            is_dyn = hasattr(model, "is_dynamic") and model.is_dynamic()
            if unload_vl_after_encode and not is_dyn:
                comfy.model_management.load_models_gpu([model], force_full_load=True)
        latent = torch.zeros([1, 16, 1, int(height) // 8, int(width) // 8], device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent}, model)

class AsepriteCharcardExtractor:
    """Extracts and saves a reusable charcard_<name>.json from any reference character image
    using local WD14 ONNX (full-body + zoomed head crop) + optional Multimodal Vision LLM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "enable_extraction": ("BOOLEAN", {"default": False, "tooltip": "开启后从参考图自动提取新角色卡；关闭时直通，让提示词导演使用其下拉菜单选中的角色卡"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "输入用户的动作/姿势/脱衣指令，让 LLM 根据你的具体意图精准提取角色卡与设定服装状态"}),
                "char_name": ("STRING", {"default": "custom_char"}),
                "use_llm": ("BOOLEAN", {"default": True}),
                "base_url": ("STRING", {"default": DEFAULT_LLM_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "model": (_get_cached_models(), {"default": _get_cached_models()[0]}),
            },
            "optional": {
                "llm_client": ("LLM_CLIENT", {"tooltip": "接入统一 AsepriteLLMProvider，无需在本地填写 URL 和 Key"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("charcard_filename", "anima_tags", "anima_nsfw_tags", "wan_subject", "charcard_json")
    FUNCTION = "extract"
    CATEGORY = "pixel_art/aseprite"

    def extract(self, reference_image, enable_extraction=False, user_prompt="", char_name="custom_char", use_llm=True, base_url="", api_key="", model="", llm_client=None, **kwargs):
        if not enable_extraction:
            return ("", "", "", "", "")
        if llm_client is not None:
            base_url = getattr(llm_client, "base_url", base_url)
            api_key = getattr(llm_client, "api_key", api_key)
            if hasattr(llm_client, "get_effective_model"):
                model = llm_client.get_effective_model(model)
        from scripts.extract_charcard import extract_charcard
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", (char_name or "custom_char").strip()).strip("_") or "custom_char"
        arr = (reference_image[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        card_data, out_fname = extract_charcard(
            Image.fromarray(arr[:, :, :3], "RGB"),
            card_name=safe_name,
            use_llm=use_llm,
            base_url=base_url,
            api_key=_resolve_api_key(api_key),
            model=model,
            user_prompt=user_prompt,
            save_to_disk=True,
        )
        return (
            out_fname,
            card_data.get("anima_tags", ""),
            card_data.get("anima_nsfw_tags", ""),
            card_data.get("wan_subject", ""),
            json.dumps(card_data, indent=2, ensure_ascii=False),
        )


class AsepriteNSFWPoseLibrary:
    """Provides 10 built-in solo & two-character interactive NSFW OpenPose skeleton guides
    and 9-frame half-cycle motion sequences for Stage 1 & Stage 2 pose/motion control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        choices = _get_pose_preset_choices()
        default_val = choices[1] if len(choices) > 1 else choices[0]
        return {
            "required": {
                "pose_preset": (choices, {"default": default_val}),
                "enable_pose_control": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "get_pose"
    CATEGORY = "pixel_art/aseprite"

    def _load_img_tensor(self, path: str):
        im = Image.open(path).convert("RGB")
        arr = np.array(im).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def get_pose(self, pose_preset="solo_m_legs_sit (单人·M字开腿坐地抚腿娇喘)", enable_pose_control=True, **kwargs):
        global _ACTIVE_POSE_ID
        pose_id = (pose_preset or "none").split(" ")[0].strip()
        _ACTIVE_POSE_ID = pose_id if enable_pose_control else "none"
        lib = _load_pose_library()
        meta = lib.get(pose_id, {})
        if not enable_pose_control or pose_id == "none" or not meta:
            return (None,)
        start_p = os.path.join(PROJECT_ROOT, meta["start_png"])
        return (self._load_img_tensor(start_p),)




class AsepritePoseRegionMask:
    """Builds a feathered inpaint MASK from detected DWPose/OpenPose body keypoints (POSE_KEYPOINT).

    Regions are placed from the keypoints actually detected on the image, for every detected person, so the mask
    follows any reference character, pose orientation, or person count. Pair with SetLatentNoiseMask + KSampler.
    Body keypoint order (OpenPose-18): 1 neck, 2 r_sho, 5 l_sho, 8 r_hip, 9 r_kne, 11 l_hip, 12 l_kne.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "pelvis": ("BOOLEAN", {"default": True}),
                "chest": ("BOOLEAN", {"default": False}),
                "size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                                   "tooltip": "Scales every region ellipse"}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 64}),
                "hands": ("BOOLEAN", {"default": False, "tooltip": "在手腕处添加重绘羽化遮罩，允许 Anima 修复错位或畸变的手部动作"}),
            },
            "optional": {
                "action_prompt": ("STRING", {"forceInput": True, "tooltip": "接导演或用户的提示词，自动根据动作激活胸部与手部重绘遮罩"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "build"
    CATEGORY = "pixel_art/aseprite"

    def build(self, pose_kps, pelvis=True, chest=False, size=1.0, feather=8, hands=False, action_prompt=""):
        frame = pose_kps[0]
        W, H = int(frame["canvas_width"]), int(frame["canvas_height"])

        # Smart Natural-Language & Explicit Syntax Mask Region Parser
        feet = False
        face = False
        if action_prompt:
            p = str(action_prompt).lower()
            # 1. Full body redraw shortcut
            if any(k in p for k in ("全身重绘", "全部重绘", "全图重绘", "full_body", "fullbody")):
                return (torch.ones((1, H, W), dtype=torch.float32),)

            # 2. Explicit syntax: 重绘范围: 胸部, 下体, 手
            m = re.search(r"重绘范围[:：]\s*([^\n;，,]+(?:[，,][^\n;，,]+)*)", p)
            if m:
                for term in re.split(r"[,，\s]+", m.group(1)):
                    t = term.strip()
                    if any(k in t for k in ("胸", "乳")): chest = True
                    if any(k in t for k in ("下体", "骨盆", "裆", "私处", "穴", "裤")): pelvis = True
                    if any(k in t for k in ("手", "腕")): hands = True
                    if any(k in t for k in ("脚", "足", "赤足")): feet = True
                    if any(k in t for k in ("脸", "面", "头", "表情")): face = True

            # 3. Dynamic Natural Language Keywords
            if any(k in p for k in ("揉胸", "摸胸", "托胸", "抓胸", "露胸", "上身裸", "乳房", "乳头", "胸部", "敞开", "贫乳", "平胸", "赤裸", "全裸", "裸体")):
                chest = True
            if any(k in p for k in ("下体", "私处", "裆部", "自慰", "脱裤", "不穿裤", "底裤", "内裤", "胖次", "短裤", "开腿", "分腿", "骨盆", "阴部", "小穴")):
                pelvis = True
            if any(k in p for k in ("手", "揉胸", "摸胸", "自慰", "摸下体", "抠逼", "扣逼", "伸入下体", "手在裆", "触碰私处", "抓胸", "双手")):
                hands = True
            if any(k in p for k in ("赤足", "光脚", "脱鞋", "光脚丫", "脚丫", "足部", "足交")):
                feet = True
            if any(k in p for k in ("表情", "脸红", "吐舌", "闭眼", "害羞", "张嘴", "含住", "吐舌头")):
                face = True

        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for person in frame.get("people", []):
            k = np.array(person.get("pose_keypoints_2d") or [], dtype=np.float32).reshape(-1, 3)
            if k.shape[0] < 13:
                continue
            if k[:, :2].max() <= 1.0:
                k[:, 0] *= W
                k[:, 1] *= H

            def pt(i):
                return k[i, :2] if k[i, 2] > 0 else None

            neck, r_sho, l_sho = pt(1), pt(2), pt(5)
            r_hip, r_kne, l_hip, l_kne = pt(8), pt(9), pt(11), pt(12)
            if r_hip is None or l_hip is None:
                continue
            hip_mid = 0.5 * (r_hip + l_hip)
            hip_w = max(float(np.linalg.norm(r_hip - l_hip)), 1.0)
            torso = float(np.linalg.norm(hip_mid - neck)) if neck is not None else 2.5 * hip_w

            # Pelvis / Crotch region
            if pelvis:
                down = (hip_mid - neck) / max(torso, 1.0) if neck is not None else np.array([0.0, 1.0], dtype=np.float32)
                c = hip_mid + down * 0.22 * torso
                rx, ry = 0.95 * hip_w * size, 0.38 * torso * size
                draw.ellipse([c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry], fill=255)

            # Chest / Breasts region
            if chest and neck is not None:
                c = 0.62 * neck + 0.38 * hip_mid
                sho_w = float(np.linalg.norm(r_sho - l_sho)) if (r_sho is not None and l_sho is not None) else 1.6 * hip_w
                rx, ry = 0.55 * sho_w * size, 0.24 * torso * size
                draw.ellipse([c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry], fill=255)

            # Hands / Wrists region
            if hands:
                r_wri, l_wri = pt(4), pt(7)
                hr = 0.20 * torso * size
                if r_wri is not None:
                    draw.ellipse([r_wri[0] - hr, r_wri[1] - hr, r_wri[0] + hr, r_wri[1] + hr], fill=255)
                if l_wri is not None:
                    draw.ellipse([l_wri[0] - hr, l_wri[1] - hr, l_wri[0] + hr, l_wri[1] + hr], fill=255)

            # Feet / Ankles region
            if feet:
                r_ank, l_ank = pt(10), pt(13)
                fr = 0.18 * torso * size
                if r_ank is not None:
                    draw.ellipse([r_ank[0] - fr, r_ank[1] - fr, r_ank[0] + fr, r_ank[1] + fr], fill=255)
                if l_ank is not None:
                    draw.ellipse([l_ank[0] - fr, l_ank[1] - fr, l_ank[0] + fr, l_ank[1] + fr], fill=255)

            # Face region
            if face and neck is not None:
                nose = pt(0)
                fc = nose if nose is not None else 0.5 * (neck + pt(14) if pt(14) is not None else neck)
                far = 0.22 * torso * size
                draw.ellipse([fc[0] - far, fc[1] - far, fc[0] + far, fc[1] + far], fill=255)

        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        return (torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0),)
try:
    import comfy_extras.nodes_string as _cns
    if hasattr(_cns, "RegexReplace"):
        _orig_regex_replace_exec = _cns.RegexReplace.execute

        @classmethod
        def _safe_regex_replace_exec(cls, string, regex_pattern, replace, case_insensitive=True, multiline=False, dotall=False, count=0):
            pat = str(regex_pattern or "").strip()
            rep = str(replace or "")
            if not pat or rep.lower() == "true":
                if "masterpiece" in str(string):
                    m = re.search(r"masterpiece[\s\S]*", str(string))
                    return _cns.io.NodeOutput(m.group(0) if m else str(string))
                return _cns.io.NodeOutput(str(string))
            return _orig_regex_replace_exec(string, regex_pattern, replace, case_insensitive=case_insensitive, multiline=multiline, dotall=dotall, count=count)

        _cns.RegexReplace.execute = _safe_regex_replace_exec
except Exception as _e:
    pass


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AsepriteLLMProvider": AsepriteLLMProvider,
    "AsepriteLLMPoseGenerator": AsepriteLLMPoseGenerator,
    "AsepritePoseSelector": AsepritePoseSelector,
    "AsepriteImageCritic": AsepriteImageCritic,
    "AsepritePixelizeAssemble": AsepritePixelizeAssemble,
    "AsepriteStageBridge": AsepriteStageBridge,
    "AsepriteLLMPromptDirector": AsepriteLLMPromptDirector,
    "QwenEditFastConditioning": QwenEditFastConditioning,
    "AsepriteCharcardExtractor": AsepriteCharcardExtractor,
    "AsepriteNSFWPoseLibrary": AsepriteNSFWPoseLibrary,
    "AsepritePoseRegionMask": AsepritePoseRegionMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsepriteLLMProvider": "LLM Provider (Centralized API Client)",
    "AsepriteLLMPoseGenerator": "LLM Pose Generator (Natural Language → OpenPose)",
    "AsepritePoseSelector": "Pose Source Selector (Preset vs LLM Generated)",
    "AsepriteImageCritic": "Image Critic & QA Gate (Multi-finger & Anatomy Evaluator)",
    "AsepritePixelizeAssemble": "Aseprite Pixelize & Assemble (.aseprite + GIF)",
    "AsepriteStageBridge": "Stage 1->2 Bridge (Free Anima VRAM)",
    "AsepriteLLMPromptDirector": "LLM Prompt Director (Anima + Wan2.2)",
    "QwenEditFastConditioning": "Qwen-Edit Fast I2I Encode (1x VL + Free VRAM)",
    "AsepriteCharcardExtractor": "Aseprite Role Card Extractor (WD14 + Vision LLM)",
    "AsepriteNSFWPoseLibrary": "NSFW Pose & Motion Library (OpenPose Control)",
    "AsepritePoseRegionMask": "Pose Region Mask (DWPose keypoints → inpaint mask)",
}
