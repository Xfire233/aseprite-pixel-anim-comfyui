#!/usr/bin/env python3
"""One-shot installer. Run with the SAME Python that runs ComfyUI, from inside ComfyUI/custom_nodes/<this repo>:

    <ComfyUI python> scripts/install.py              # everything
    <ComfyUI python> scripts/install.py --skip-models
    <ComfyUI python> scripts/install.py --only-models --sfw   # skip the two NSFW LoRAs

Steps (each is idempotent, safe to re-run):
  1. pip install -r requirements.txt into the current interpreter
  2. git clone the third-party custom nodes the workflows need (skips ones already present)
  3. download every weight in models.json into ComfyUI/models/<dir>/ (resumes; verifies sha256 prefix)
  4. apply patches/comfyui_core_bypass_lora.patch to ComfyUI (skips if already applied)
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY_ROOT, PROJECT_ROOT  # noqa: E402

CUSTOM_NODES = [
    # (folder name, git url, why)
    ("comfyui_controlnet_aux", "https://github.com/Fannovel16/comfyui_controlnet_aux", "DWPreprocessor (Stage 1B pelvis mask)"),
    ("ComfyUI-RMBG", "https://github.com/1038lab/ComfyUI-RMBG", "RMBG node (white background)"),
    ("ComfyUI-WD14-Tagger", "https://github.com/pythongosssss/ComfyUI-WD14-Tagger", "charcard extractor (optional)"),
]
# RISA Attention: the ComfyUI node lives in a subfolder of the CUDA package repo.
RISA_REPO = "https://github.com/nukuihayu/RisaAttention"
RISA_NODE_DIR = "comfyui-risa-attention"
# Folder names other installs commonly use for the same nodes (e.g. ComfyUI-aki package)
ALIASES = {"ComfyUI-RMBG": ["comfyui-rmbg"], "ComfyUI-WD14-Tagger": ["comfyui-WD14-Tagger"]}

def run(cmd, cwd=None):
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


def step_requirements():
    print("[1/4] Python requirements")
    req = os.path.join(PROJECT_ROOT, "requirements.txt")
    if run([sys.executable, "-m", "pip", "install", "-r", req]) != 0:
        sys.exit("pip install failed")


def step_custom_nodes():
    print("[2/4] Custom nodes")
    cn = os.path.join(COMFY_ROOT, "custom_nodes")
    for folder, url, why in CUSTOM_NODES:
        present = [d for d in [folder] + ALIASES.get(folder, []) if os.path.isdir(os.path.join(cn, d))]
        if present:
            print(f"  ok   {present[0]:28s} ({why})")
            continue
        print(f"  get  {folder:28s} ({why})")
        if run(["git", "clone", "--depth", "1", url, os.path.join(cn, folder)]) != 0:
            print(f"  !! clone failed for {url}; install it via ComfyUI-Manager instead")
            continue
        req = os.path.join(cn, folder, "requirements.txt")
        if os.path.exists(req):
            run([sys.executable, "-m", "pip", "install", "-r", req])
    _install_risa(cn)


def _risa_works() -> bool:
    code = "import risa_attention as r, sys; sys.exit(0 if r.int8_attention_is_available() else 1)"
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def _install_risa(cn):
    """Node is always installed (pytorch_attention mode works without CUDA build). INT8 kernels are best-effort."""
    node_dst = os.path.join(cn, RISA_NODE_DIR)
    src = os.path.join(PROJECT_ROOT, "work", "_risa_src")
    if not os.path.isdir(node_dst) or not _risa_works():
        if not os.path.isdir(src):
            run(["git", "clone", "--depth", "1", RISA_REPO, src])
        if not os.path.isdir(node_dst) and os.path.isdir(os.path.join(src, RISA_NODE_DIR)):
            shutil.copytree(os.path.join(src, RISA_NODE_DIR), node_dst)
            print(f"  ok   {RISA_NODE_DIR:28s} (RISAAttention node)")
    if _risa_works():
        print("  ok   risa_attention CUDA kernels")
        return
    print("  build risa_attention CUDA kernels (needs CUDA Toolkit 12.8+, CMake, a C++ compiler; ~5-15 min)")
    if os.path.isdir(src) and run([sys.executable, "-m", "pip", "install", src]) == 0 and _risa_works():
        print("  ok   risa_attention built")
        return
    print("  !! risa_attention INT8 build unavailable -> switching shipped workflows to pytorch_attention (slower, same output)")
    _set_attention_mode("pytorch_attention")


def _set_attention_mode(mode):
    import glob
    for p in glob.glob(os.path.join(PROJECT_ROOT, "workflows", "*_web.json")):
        wf = json.load(open(p, encoding="utf-8"))
        for n in wf["nodes"]:
            if n["type"] == "RISAAttention" and isinstance(n.get("widgets_values"), list) and n["widgets_values"]:
                n["widgets_values"][0] = mode
        json.dump(wf, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _sha_prefix(path, n):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _download(url, dst, retries=8):
    """Resumable download. A dropped connection can end the stream early without an error, so completion is
    judged by byte count against the server-reported total, not by the read loop ending."""
    tmp = dst + ".part"
    total = None
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total and have >= total:
            break
        headers = {"User-Agent": "aseprite-workflow-installer"}
        if have:
            headers["Range"] = f"bytes={have}-"
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                resumed = have and r.status == 206
                if not resumed:
                    have = 0
                length = int(r.headers.get("Content-Length") or 0)
                total = have + length if length else total
                done = have
                last_pct = -1
                with open(tmp, "ab" if resumed else "wb") as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(done * 100 / total) if total else -1
                        if pct != last_pct:
                            last_pct = pct
                            print(f"\r       {done / 2**30:6.2f} / {total / 2**30:.2f} GB  {pct:3d}%", end="", flush=True)
            print()
        except Exception as e:
            print(f"\n       connection error ({e}); retry {attempt}/{retries}")
            continue
        if total and os.path.getsize(tmp) >= total:
            break
        print(f"       stream ended early at {os.path.getsize(tmp) / 2**30:.2f} GB; resuming ({attempt}/{retries})")
    if not total or os.path.getsize(tmp) < total:
        sys.exit(f"  download incomplete for {os.path.basename(dst)}; re-run install.py to resume from {tmp}")
    os.replace(tmp, dst)


def step_models(sfw: bool):
    print("[3/4] Models")
    manifest = json.load(open(os.path.join(PROJECT_ROOT, "models.json"), encoding="utf-8"))["models"]
    todo = [m for m in manifest if not (sfw and m.get("nsfw"))]
    need = sum(m["size_gb"] for m in todo if not os.path.exists(os.path.join(COMFY_ROOT, "models", m["directory"], m["name"])))
    free = shutil.disk_usage(COMFY_ROOT).free / 2**30
    print(f"  to download: {need:.1f} GB, free: {free:.1f} GB")
    if need > free - 5:
        sys.exit("  not enough disk space")
    for m in todo:
        dst_dir = os.path.join(COMFY_ROOT, "models", m["directory"])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, m["name"])
        if os.path.exists(dst):
            print(f"  ok   {m['name']}")
            continue
        print(f"  get  {m['name']}  ({m['size_gb']} GB)")
        _download(m["url"], dst)
        got = _sha_prefix(dst, len(m["sha256"]))
        if got != m["sha256"]:
            os.remove(dst)
            sys.exit(f"  sha256 mismatch for {m['name']}: {got} != {m['sha256']} (deleted; re-run to retry)")


def step_core_patch():
    print("[4/4] ComfyUI core patch (multi bypass-LoRA fix)")
    patch = os.path.join(PROJECT_ROOT, "patches", "comfyui_core_bypass_lora.patch")
    if not os.path.isdir(os.path.join(COMFY_ROOT, ".git")):
        print("  ComfyUI is not a git checkout; apply manually: git apply", patch)
        return
    if subprocess.run(["git", "apply", "--reverse", "--check", patch], cwd=COMFY_ROOT, capture_output=True).returncode == 0:
        print("  already applied")
        return
    if subprocess.run(["git", "apply", "--check", patch], cwd=COMFY_ROOT, capture_output=True).returncode != 0:
        print("  !! patch does not apply cleanly to this ComfyUI version; stacked NSFW LoRAs may override each other. See README.")
        return
    run(["git", "apply", patch], cwd=COMFY_ROOT)


def step_inputs():
    print("[5/5] Example inputs & pose library")
    comfy_in = os.path.join(COMFY_ROOT, "input")
    os.makedirs(comfy_in, exist_ok=True)
    import glob
    for p in glob.glob(os.path.join(PROJECT_ROOT, "input", "*.*")):
        dst = os.path.join(comfy_in, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copyfile(p, dst)
    for p in glob.glob(os.path.join(PROJECT_ROOT, "poses", "nsfw_library", "*.png")):
        dst = os.path.join(comfy_in, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copyfile(p, dst)
    print("  inputs and pose skeletons synced to ComfyUI/input/")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--only-models", action="store_true")
    ap.add_argument("--sfw", action="store_true", help="skip NSFW LoRAs (workflows then need those LoRA nodes bypassed)")
    args = ap.parse_args()
    if not COMFY_ROOT or not os.path.exists(os.path.join(COMFY_ROOT, "main.py")):
        sys.exit("ComfyUI not found. Clone this repo into ComfyUI/custom_nodes/ or set ASEPRITE_WF_COMFY_ROOT.")
    print(f"ComfyUI: {COMFY_ROOT}\nPython : {sys.executable}")
    if not args.only_models:
        step_requirements()
        step_custom_nodes()
    if not args.skip_models:
        step_models(args.sfw)
    if not args.only_models:
        step_core_patch()
    step_inputs()
    print("Done. Restart ComfyUI, then drag a file from workflows/ into the browser.")


if __name__ == "__main__":
    main()
