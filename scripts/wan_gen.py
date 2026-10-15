#!/usr/bin/env python
"""Wan2.2 I2V (Winnougan INT8-ConvRot dual-expert) + LightX2V 4-step Distill LoRA -> frames + pixelize.

Usage:
  python scripts/wan_gen.py --image first_frame.png --prompt "..." --prefix phase0/idle \
      --pixelize-out work/idle --assemble output/idle.aseprite --zoom work/idle_zoom.png
"""
import argparse, json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY_API as API, COMFY_OUTPUT as COMFY_OUT  # noqa: E402

HI_LORA = "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
LO_LORA = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
       "camera movement, zoom, morphing, warping, changing clothes")


def post(path, obj=None):
    req = urllib.request.Request(API + path, data=json.dumps(obj).encode() if obj else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(API + path, timeout=60))


def model_chain(unet, nid, attention, theta, lora_name=None, lora_strength=1.0):
    nodes = {
        nid: {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
    }
    prev = nid
    if lora_name:
        lid = nid + "l"
        nodes[lid] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": [prev, 0], "lora_name": lora_name, "strength_model": lora_strength}}
        prev = lid
    sid = nid + "s"
    nodes[sid] = {"class_type": "ModelSamplingSD3", "inputs": {"model": [prev, 0], "shift": 5.0}}
    prev = sid
    if attention != "none":
        rid = nid + "r"
        nodes[rid] = {"class_type": "RISAAttention",
                      "inputs": {"model": [prev, 0], "attention": attention, "theta": theta}}
        prev = rid
    return nodes, prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="start image filename in ComfyUI input/")
    ap.add_argument("--end-image", default=None, help="optional end image in ComfyUI input/ (uses WanFirstLastFrameToVideo)")
    ap.add_argument("--loop", action="store_true", help="pin end_image = start_image via WanFirstLastFrameToVideo for closed forward loops")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--length", type=int, default=9, help="frames (4n+1: 9=half-cycle, 13, 17)")
    ap.add_argument("--distill", action=argparse.BooleanOptionalAction, default=True,
                    help="use LightX2V 4-step Distill LoRAs (default: True; sets steps=4, split=2, cfg=1.0 unless overridden)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--split", type=int, default=None, help="high->low expert switch step")
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--attention", default="sparse_int8_attention",
                    choices=["sparse_int8_attention", "int8_attention", "pytorch_attention", "none"])
    ap.add_argument("--theta", type=float, default=0.99)
    ap.add_argument("--tiled-vae", action=argparse.BooleanOptionalAction, default=True,
                    help="use VAEDecodeTiled to avoid WDDM shared-memory spill on 16GB VRAM (default: True)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--pixelize-out", default=None, help="if set, run pixelize.py on generated frames into this dir")
    ap.add_argument("--subsample", type=int, default=4, help="keyframes to subsample for pixelize.py (default: 4)")
    ap.add_argument("--pingpong", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--canvas", default="128x224")
    ap.add_argument("--assemble", default=None, help="optional output .aseprite path for pixelize.py")
    ap.add_argument("--zoom", default=None, help="optional zoom preview PNG path for pixelize.py")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else (4 if args.distill else 12)
    split = args.split if args.split is not None else (2 if args.distill else 6)
    cfg = args.cfg if args.cfg is not None else (1.0 if args.distill else 3.5)

    wf, hi = model_chain("wan2.2_i2v_high_noise_14B_int8_convrot.safetensors", "1", args.attention, args.theta,
                         HI_LORA if args.distill else None)
    wf2, lo = model_chain("wan2.2_i2v_low_noise_14B_int8_convrot.safetensors", "2", args.attention, args.theta,
                          LO_LORA if args.distill else None)
    wf.update(wf2)
    end_img = args.end_image or (args.image if args.loop else None)
    if end_img:
        wf["7e"] = {"class_type": "LoadImage", "inputs": {"image": end_img}}
        i2v_node = {"class_type": "WanFirstLastFrameToVideo",
                    "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["4", 0],
                               "width": args.width, "height": args.height, "length": args.length,
                               "batch_size": 1, "start_image": ["7", 0], "end_image": ["7e", 0]}}
    else:
        i2v_node = {"class_type": "WanImageToVideo",
                    "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["4", 0],
                               "width": args.width, "height": args.height, "length": args.length,
                               "batch_size": 1, "start_image": ["7", 0]}}
    decode_node = ({"class_type": "VAEDecodeTiled",
                    "inputs": {"samples": ["10", 0], "vae": ["4", 0], "tile_size": 256, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
                   if args.tiled_vae else
                   {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}})
    wf.update({
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": args.prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEG}},
        "7": {"class_type": "LoadImage", "inputs": {"image": args.image}},
        "8": i2v_node,
        "9": {"class_type": "KSamplerAdvanced", "inputs": {"model": [hi, 0], "add_noise": "enable", "noise_seed": args.seed,
              "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["8", 2],
              "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}},
        "10": {"class_type": "KSamplerAdvanced", "inputs": {"model": [lo, 0], "add_noise": "disable", "noise_seed": 0,
               "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
               "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["9", 0],
               "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "11": decode_node,
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": args.fps}},
        "13": {"class_type": "SaveVideo", "inputs": {"video": ["12", 0], "filename_prefix": args.prefix, "format": "auto", "codec": "auto"}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": args.prefix + "_frames/f"}},
    })
    pid = post("/prompt", {"prompt": wf})["prompt_id"]
    print("submitted", pid, flush=True)
    t0, last = time.time(), 0
    while time.time() - t0 < args.timeout:
        try:
            h = get(f"/history/{pid}")
        except Exception:
            time.sleep(2); continue
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                msgs = st.get("messages", [])
                sys.exit("GEN ERROR: " + json.dumps(msgs[-1] if msgs else st)[:1800])
            outs = []
            for o in h[pid].get("outputs", {}).values():
                for k in ("images", "gifs", "videos"):
                    outs += [x.get("filename") for x in o.get(k, [])]
            print(f"OK {time.time()-t0:.1f}s ->", outs[:4], "... total", len(outs))
            if args.pixelize_out:
                f_imgs = h[pid].get("outputs", {}).get("14", {}).get("images", [])
                f_paths = [os.path.join(COMFY_OUT, x.get("subfolder", ""), x["filename"]) for x in f_imgs]
                cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "pixelize.py"),
                       "--frames", *f_paths, "--no-register", "--consensus", "none",
                       "--canvas", args.canvas, "--out", args.pixelize_out]
                if args.subsample:
                    cmd += ["--subsample", str(args.subsample)]
                if args.pingpong and not args.loop:
                    cmd += ["--pingpong"]
                if args.assemble:
                    cmd += ["--assemble", args.assemble]
                if args.zoom:
                    cmd += ["--zoom", args.zoom]
                subprocess.run(cmd, check=True)
            return
        if time.time() - t0 - last > 60:
            last = time.time() - t0
            print(f"...{last:.0f}s still running", flush=True)
        time.sleep(1)
    try:
        post("/interrupt")
    except Exception:
        pass
    sys.exit(f"TIMEOUT {args.timeout}s")


if __name__ == "__main__":
    main()
