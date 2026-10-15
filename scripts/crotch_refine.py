#!/usr/bin/env python
"""Region refinement: crop the hip/crotch region, Qwen-Edit it (GGUF + Unchained XXX LoRA,
full recipe 20 steps CFG 3.5), paste back centroid-aligned with an elliptical feather.
Full-body generations draw a smooth censored crotch; the model only draws detail in close-up.

Usage: uv run --with pillow --with numpy python scripts/crotch_refine.py --frame in.png --out out.png --prefix phase0/ref_x
ComfyUI must be running on 127.0.0.1:8188 (lowvram). One frame per call (~2-3 min).
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY_API as API, COMFY_ROOT as COMFY  # noqa: E402
PROMPT = ("Image 1 is a close-up anime illustration of a nude woman's hips and upper thighs with sheer beige thigh-high stockings. "
          "Redraw the exact same image with the same framing, skin tone, lighting and clean anime style, changing only one thing: "
          "the smooth featureless crotch becomes anatomically correct and clearly detailed - a visible vulva with labia and a defined cleft between her thighs, "
          "drawn in the same anime style with soft shading. Keep the navel, hips, thighs and stockings identical.")


def post(path, obj=None):
    req = urllib.request.Request(API + path, data=json.dumps(obj).encode() if obj else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(API + path, timeout=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True, help="ComfyUI output prefix, e.g. phase0/ref_f1")
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--edited", default=None, help="reuse an existing edited crop PNG (skip generation)")
    args = ap.parse_args()

    im = Image.open(args.frame).convert("RGB")
    a = np.asarray(im).astype(int)
    mask = a.min(axis=2) < 242
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bh = y1 - y0
    cx = int((x0 + x1) / 2)
    cl, cr = max(0, cx - 95), min(im.width, cx + 95)
    ct, cb = int(y0 + 0.50 * bh), int(y0 + 0.80 * bh)
    crop = im.crop((cl, ct, cr, cb))
    cw, ch = crop.size
    big = crop.resize((cw * 3, ch * 3), Image.LANCZOS)
    stage = os.path.join(COMFY, "input", "refine_crop.png")
    big.save(stage)
    lw = (big.width // 16) * 16
    lh = (big.height // 16) * 16
    print(f"crop box=({cl},{ct},{cr},{cb}) latent={lw}x{lh}")

    if args.edited:
        ed = Image.open(args.edited).convert("RGB")
        print(f"reusing edited crop {args.edited} crop box=({cl},{ct},{cr},{cb})")
    else:
        wf = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q3_K_M.gguf"}},
            "1c": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Unchained-XXX.safetensors", "strength_model": args.strength}},
            "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1c", 0], "shift": 3.1}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "5": {"class_type": "LoadImage", "inputs": {"image": "refine_crop.png"}},
            "7": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "image1": ["5", 0], "prompt": PROMPT}},
            "7n": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": ""}},
            "9": {"class_type": "EmptyLatentImage", "inputs": {"width": lw, "height": lh, "batch_size": 1}},
            "10": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["7", 0], "negative": ["7n", 0],
                   "latent_image": ["9", 0], "seed": args.seed, "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
            "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": args.prefix}},
        }
        pid = post("/prompt", {"prompt": wf})["prompt_id"]
        t0 = time.time()
        out_name = None
        while time.time() - t0 < 700:
            try:
                h = get(f"/history/{pid}")
            except Exception:
                time.sleep(4); continue
            if pid in h:
                if h[pid]["status"].get("status_str") == "error":
                    sys.exit(f"GEN ERROR: {json.dumps(h[pid]['status'])[:500]}")
                out_name = [im2["filename"] for o in h[pid].get("outputs", {}).values() for im2 in o.get("images", [])][0]
                break
            time.sleep(4)
        if not out_name:
            sys.exit("TIMEOUT")
        print(f"edited in {time.time()-t0:.0f}s -> {out_name}")
        sub = args.prefix.split("/")[0] if "/" in args.prefix else ""
        ed = Image.open(os.path.join(COMFY, "output", sub, out_name)).convert("RGB")

    # detail centroid in edited crop: dark saturated red pixels
    ea = np.asarray(ed).astype(int)
    red = (ea[:, :, 0] - ea[:, :, 2] > 45) & (ea[:, :, 0] > 85) & (ea.mean(axis=2) < 178)
    if red.sum() >= 12:
        eys, exs = np.where(red)
        dcx, dcy = float(np.median(exs)), float(np.median(eys))
    else:
        dcx, dcy = ed.width * 0.5, ed.height * 0.42
        print("WARNING: detail centroid fallback (few red px)")

    # anchor in original: top of the white gap between thighs (count white px near cx per row)
    white = a.min(axis=2) >= 242
    gap_y, run = None, 0
    for y in range(ct + 10, cb):
        if white[y, max(0, cx - 14):cx + 15].sum() >= 8:
            run += 1
            if run >= 5:
                gap_y = y - run + 1
                break
        else:
            run = 0
    if gap_y is not None:
        gx = np.where(white[min(gap_y + 6, cb - 1), max(0, cx - 20):cx + 21])[0]
        ax = int(max(0, cx - 20) + gx.mean()) if len(gx) else cx
        ay = gap_y - 10
    else:
        ax, ay = cx, int(ct + 0.72 * (cb - ct))
    print(f"detail centroid=({dcx:.0f},{dcy:.0f}) anchor=({ax},{ay}) gap_y={gap_y}")

    # scale edited back to crop size, shift so centroid lands on anchor, elliptical feather paste
    ed_s = ed.resize((cw, ch), Image.LANCZOS)
    scx, scy = dcx * cw / ed.width, dcy * ch / ed.height
    dx, dy = int(round(ax - cl - scx)), int(round(ay - ct - scy))
    canvas = Image.new("RGB", im.size, (0, 0, 0))
    canvas.paste(ed_s, (cl + dx, ct + dy))
    yy, xx = np.mgrid[0:im.height, 0:im.width]
    rx, ry, fe = 58.0, 44.0, 16.0
    d = np.sqrt(((xx - ax) / rx) ** 2 + ((yy - ay) / ry) ** 2)
    alpha = np.clip((1.0 - d) * (rx / fe), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    # never paste outside the silhouette (edited crop may place skin over background)
    alpha = alpha * mask
    out = (np.asarray(canvas).astype(float) * alpha[:, :, None] + a.astype(float) * (1 - alpha[:, :, None])).astype(np.uint8)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out} shift=({dx},{dy})")


if __name__ == "__main__":
    main()
