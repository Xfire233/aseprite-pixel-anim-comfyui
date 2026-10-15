"""Machine-specific locations, overridable by environment variables.

Set these once (e.g. in your shell profile) instead of editing scripts:
  ASEPRITE_WF_COMFY_ROOT   ComfyUI directory (contains main.py, models/, input/, output/)
  ASEPRITE_WF_ASEPRITE     Aseprite executable
  ASEPRITE_WF_COMFY_API    ComfyUI HTTP API base (default http://127.0.0.1:8188)
  LLM_BASE_URL / LLM_API_KEY  optional OpenAI-compatible endpoint for the prompt director / charcard extractor
"""
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _detect_comfy_root() -> str:
    # Normal install: <ComfyUI>/custom_nodes/<this repo>/scripts/paths.py
    parent = os.path.dirname(PROJECT_ROOT)
    if os.path.basename(parent) == "custom_nodes" and os.path.exists(os.path.join(os.path.dirname(parent), "main.py")):
        return os.path.dirname(parent)
    known = r"C:\baidunetdiskdownload\comfyui1.6\ComfyUI-aki-v1.6\ComfyUI"
    if os.path.exists(os.path.join(known, "main.py")):
        return known
    return ""


COMFY_ROOT = os.environ.get("ASEPRITE_WF_COMFY_ROOT") or _detect_comfy_root()
COMFY_INPUT = os.path.join(COMFY_ROOT, "input") if COMFY_ROOT else ""
COMFY_OUTPUT = os.path.join(COMFY_ROOT, "output") if COMFY_ROOT else ""
COMFY_API = os.environ.get("ASEPRITE_WF_COMFY_API", "http://127.0.0.1:8188")


def _detect_aseprite() -> str:
    for cand in (
        shutil.which("aseprite"), shutil.which("Aseprite"),
        r"C:/Program Files (x86)/Steam/steamapps/common/Aseprite/Aseprite.exe",
        r"C:/Program Files/Aseprite/Aseprite.exe",
        os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/Aseprite/Aseprite.app/Contents/MacOS/aseprite"),
        "/Applications/Aseprite.app/Contents/MacOS/aseprite",
        os.path.expanduser("~/.steam/steam/steamapps/common/Aseprite/aseprite"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return "aseprite"


ASEPRITE = os.environ.get("ASEPRITE_WF_ASEPRITE") or _detect_aseprite()

# Local, git-ignored secret written when a key is typed into the ComfyUI node.
LLM_KEY_FILE = os.path.join(PROJECT_ROOT, "work", ".llm_api_key.secret")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")


def resolve_llm_key(ui_key: str = "") -> str:
    """UI-entered key > env var name typed into UI > local secret file > LLM_API_KEY env."""
    k = (ui_key or "").strip()
    if k and not set(k) <= set("•* ") and k != "__SAVED__" and not k.startswith("(空"):
        if k in os.environ:
            return os.environ[k]
        return k
    if os.path.exists(LLM_KEY_FILE):
        with open(LLM_KEY_FILE, encoding="utf-8") as f:
            saved = f.read().strip()
        if saved:
            return saved
    return os.environ.get("LLM_API_KEY", "")
