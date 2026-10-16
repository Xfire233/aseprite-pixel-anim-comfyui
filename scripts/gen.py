#!/usr/bin/env python
"""Generate one pose-edited frame via ComfyUI (定版配方: Qwen-Image-Edit-2511 W4A8-ConvRot + 4-step Lightning LoRA Bypass + RisaAttention + QwenEditFastConditioning + VAEDecodeTiled).

Usage:
  uv run python scripts/gen.py --phase "standing with one hand on hip, waving" --prefix phase0/wave_ref
  uv run python scripts/gen.py --ref my_char.png --card none --phase "sitting on floor" --prefix phase0/custom_ref
  uv run python scripts/gen.py --pose idle2_s2.png --phase "shoulders risen slightly" --prefix phase0/idle_f2

Rules baked in (see SPEC 坑速查):
- Uses UNETLoader(qwen_image_edit_2511_w4a8.safetensors) + LoraLoaderBypassModelOnly + RISAAttention(int8_attention)
- Uses QwenEditFastConditioning (single VL pass at 504x504, ref_max_side=512, index_timestep_zero, unloads VL)
- Uses VAEDecodeTiled(tile_size=256) to prevent WanVAE WDDM shared-memory spillover on 16GB VRAM
- 240s hard cap with server interrupt; preflight disk/server/model checks.
Requires ComfyUI already running on 127.0.0.1:8188.
"""
import argparse, json, os, shutil, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY_API as API, COMFY_ROOT as COMFY, PROJECT_ROOT as ROOT  # noqa: E402

MODEL = "qwen_image_edit_2511_w4a8.safetensors"
LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def post(path, obj=None):
    req = urllib.request.Request(API + path, data=json.dumps(obj).encode() if obj else b"",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r) if obj else r.read()


def get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.load(r)


def preflight():
    if not COMFY:
        sys.exit("PREFLIGHT FAIL: ComfyUI folder not found. Clone this repo into ComfyUI/custom_nodes/ or set ASEPRITE_WF_COMFY_ROOT.")
    free = shutil.disk_usage(COMFY).free / 1e9
    if free < 20:
        sys.exit(f"PREFLIGHT FAIL: free space {free:.0f}GB < 20GB on the ComfyUI drive")
    try:
        get("/system_stats")
    except Exception:
        sys.exit(f"PREFLIGHT FAIL: ComfyUI not reachable at {API} — start it first")
    if not os.path.exists(os.path.join(COMFY, "models/diffusion_models", MODEL)):
        sys.exit(f"PREFLIGHT FAIL: model missing: {MODEL} — run: python scripts/install.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=None, help="optional pose image filename already in ComfyUI input/")
    ap.add_argument("--pose-file", default=None, help="optional local pose image path (copied into ComfyUI input/)")
    ap.add_argument("--pose-preset", default=None, help="optional NSFW pose preset ID from poses/nsfw_library/index.json (e.g. solo_m_legs_sit, duo_cowgirl_ride)")
    ap.add_argument("--ref", default=None, help="reference image: any file path (auto-copied into ComfyUI input/) or a filename already there")
    ap.add_argument("--phase", required=True, help="pose/phase description inserted into prompt")
    ap.add_argument("--prefix", required=True, help="output filename prefix, e.g. phase0/idle_f2")
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--canvas", default="512x768", help="generation resolution WxH (default: 512x768, or 384x576 for fastest)")
    ap.add_argument("--ref-max-side", type=int, default=512, help="max side for reference_latents VAE encode (default: 512)")
    ap.add_argument("--unchained-lora", type=float, default=0.50, help="strength for Unchained-XXX.safetensors LoRA")
    ap.add_argument("--nsfw-lora", type=float, default=0.45, help="strength for qwen_nsfw_adv.safetensors LoKr")
    ap.add_argument("--anima-refine", action="store_true", help="run Stage 1B: DWPose pelvis mask + Anima Base masked inpaint (rest of image untouched)")
    ap.add_argument("--anima-denoise", type=float, default=0.70, help="denoise strength inside the pelvis mask")
    ap.add_argument("--mask-size", type=float, default=1.4, help="AsepritePoseRegionMask size (pelvis region scale)")
    ap.add_argument("--inpaint-seed", type=int, default=None, help="Stage 1B KSampler seed (re-roll if clothing remains); defaults to --seed")
    ap.add_argument("--attention", default="int8_attention",
                    choices=["int8_attention", "sparse_int8_attention", "pytorch_attention"])
    ap.add_argument("--theta", type=float, default=0.99)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--card", default="none", help="character card JSON path (see scripts/charcard.example.json) or 'none' to read appearance from --ref")
    args = ap.parse_args()

    preflight()
    use_card = args.card and args.card.strip().lower() != "none"
    card = json.load(open(args.card, encoding="utf-8")) if use_card else None
    def stage(path_or_name):
        """Accept a file path anywhere on disk (copied into ComfyUI/input) or a name already in ComfyUI/input."""
        if path_or_name and os.path.isfile(path_or_name):
            dst = os.path.join(COMFY, "input", os.path.basename(path_or_name))
            if os.path.abspath(path_or_name) != os.path.abspath(dst):
                shutil.copy(path_or_name, dst)
            return os.path.basename(path_or_name)
        return path_or_name

    ref = stage(args.ref or ((card or {}).get("comfy_ref")))
    if not ref:
        sys.exit("need --ref <reference image path or ComfyUI input filename>")
    pose = None
    pose_meta = None
    if args.pose_preset:
        idx_p = os.path.join(ROOT, "poses", "nsfw_library", "index.json")
        if os.path.exists(idx_p):
            lib = json.load(open(idx_p, encoding="utf-8"))
            pose_meta = lib.get(args.pose_preset)
            if pose_meta:
                src = os.path.join(ROOT, pose_meta["start_png"])
                pose = pose_meta["comfy_input_start"]
                shutil.copy(src, os.path.join(COMFY, "input", pose))
    if args.pose_file:
        pose = stage(args.pose_file)
    elif args.pose:
        pose = args.pose
    w, h = map(int, args.canvas.split("x"))
    char_desc = f"Keep her exact appearance from image 1: {card['card']}. She wears {card['outfit']}." if card else (
        "Keep the exact same character identity, face, hairstyle, hair color, eye color, accessories, and outfit from image 1."
    )
    if pose:
        prompt = (
            f"Image 1 is the character. Image 2 is a skeleton pose guide. "
            f"Redraw the character from image 1 in the exact pose of image 2 ({args.phase}, "
            f"the character is small in the frame occupying only the center 55 percent of the image height "
            f"with lots of empty white space above the head and below the feet), "
            f"entire full body including head and feet fully inside the frame. "
            f"{char_desc} Plain white background, clean anime illustration style.")
    else:
        prompt = (
            f"{char_desc} "
            f"Change the character's pose to: {args.phase}, entire full body including head and feet fully inside the frame "
            f"with empty white space above the head and below the feet. "
            f"Plain white background, clean anime illustration style.")
    cond_inputs = {
        "clip": ["3", 0],
        "vae": ["4", 0],
        "image1": ["5", 0],
        "model": ["2", 0],
        "prompt": prompt,
        "width": w,
        "height": h,
        "ref_max_side": args.ref_max_side,
        "unload_vl_after_encode": True,
    }
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "1u": {"class_type": "LoraLoaderBypassModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": "Unchained-XXX.safetensors", "strength_model": args.unchained_lora}},
        "1n": {"class_type": "LoraLoaderBypassModelOnly",
               "inputs": {"model": ["1u", 0], "lora_name": "qwen_nsfw_adv.safetensors", "strength_model": args.nsfw_lora}},
        "1l": {"class_type": "LoraLoaderBypassModelOnly",
               "inputs": {"model": ["1n", 0], "lora_name": LORA, "strength_model": 1.0}},
        "1r": {"class_type": "RISAAttention",
               "inputs": {"model": ["1l", 0], "attention": args.attention, "theta": args.theta}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1r", 0], "shift": 3.1}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "7": {"class_type": "QwenEditFastConditioning", "inputs": cond_inputs},
        "10": {"class_type": "KSampler", "inputs": {"model": ["7", 3], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["7", 2], "seed": args.seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["10", 0], "vae": ["4", 0], "tile_size": 256, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
    }
    if pose:
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": pose}}
        cond_inputs["image2"] = ["6", 0]
    if args.anima_refine:
        is_petite = True if not card else any(k in str(card).lower() for k in ("small breasts", "flat chest", "slender", "petite", "hiro"))
        anima_pos = (card.get("anima_nsfw_tags") or card.get("anima_tags") or "") if card else ""
        anima_pos = f"masterpiece, best quality, score_9, score_8, newest, 2d, cel shading, clean lineart, 1girl, nsfw, bottomless, no panties, pussy, uncensored, {anima_pos}".strip(" ,")
        if pose_meta and pose_meta.get("anima_pose_tags"):
            anima_pos = f"{anima_pos}, {pose_meta['anima_pose_tags']}".strip(" ,")
        if is_petite:
            import re
            anima_pos = re.sub(r"\b(hanging breasts|large breasts|huge breasts|cleavage)\b\s*,?", "", anima_pos, flags=re.IGNORECASE)
            anima_pos = f"{anima_pos}, small breasts, flat chest, slender, narrow hips, pink nipples".strip(" ,")
        anima_neg = (card.get("negative_tags") or "panties, underwear, shorts, bra") if card else "panties, underwear, shorts, bra"
        if is_petite:
            anima_neg = f"{anima_neg}, large breasts, huge breasts, hanging breasts, cleavage, wide hips, voluptuous, curvy, thick thighs"
        has_chest_refine = any(k in args.phase.lower() for k in ("breast", "nipple", "chest", "topless", "nude", "胸", "乳", "赤裸", "全裸", "贫乳", "平胸"))
        wf["11m"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}}
        wf["11c"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}}
        wf["11d"] = {"class_type": "DWPreprocessor", "inputs": {"image": ["11", 0], "detect_hand": "disable", "detect_body": "enable", "detect_face": "disable", "resolution": 512, "bbox_detector": "None", "pose_estimator": "dw-ll_ucoco_384.onnx", "scale_stick_for_xinsr_cn": "disable"}}
        wf["11k"] = {"class_type": "AsepritePoseRegionMask", "inputs": {"pose_kps": ["11d", 1], "pelvis": True, "chest": has_chest_refine, "size": args.mask_size, "feather": 8, "hands": False}}
        wf["11p"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11c", 0], "text": anima_pos}}
        wf["11n"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11c", 0], "text": f"{anima_neg}, shorts, denim shorts, panties, underwear, lingerie, briefs, censored"}}
        wf["11e"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 0]}}
        wf["11s"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["11e", 0], "mask": ["11k", 0]}}
        wf["11x"] = {"class_type": "KSampler", "inputs": {"model": ["11m", 0], "positive": ["11p", 0], "negative": ["11n", 0], "latent_image": ["11s", 0], "seed": args.inpaint_seed if args.inpaint_seed is not None else args.seed, "steps": 16, "cfg": 4.5, "sampler_name": "euler", "scheduler": "simple", "denoise": args.anima_denoise}}
        wf["11y"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["11x", 0], "vae": ["4", 0], "tile_size": 256, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
        wf["11r"] = {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["11", 0], "source": ["11y", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["11k", 0]}}
        wf["11b"] = {"class_type": "RMBG", "inputs": {"image": ["11r", 0], "model": "RMBG-2.0", "sensitivity": 1.0, "process_res": 1024, "mask_blur": 0, "mask_offset": 0, "background": "white", "invert_output": False, "optimize": "default"}}
        wf["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11b", 0], "filename_prefix": args.prefix}}
    else:
        wf["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": args.prefix}}
    pid = post("/prompt", {"prompt": wf})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        try:
            hist = get(f"/history/{pid}")
        except Exception:
            time.sleep(4)
            continue
        if pid in hist:
            st = hist[pid]["status"]
            if st.get("status_str") == "error":
                sys.exit(f"GEN ERROR: {json.dumps(st)[:600]}")
            outs = [im for o in hist[pid].get("outputs", {}).values() for im in o.get("images", [])]
            print(f"OK {time.time()-t0:.0f}s -> {[o['filename'] for o in outs]}")
            return
        time.sleep(3)
    try:
        post("/interrupt")
    except Exception:
        pass
    sys.exit(f"TIMEOUT {args.timeout}s — interrupted (likely VRAM paging; check input sizes / restart server)")


if __name__ == "__main__":
    main()
