#!/usr/bin/env python3
"""Write ComfyUI's native `properties.models` download metadata into every loader node of workflows/*_web.json.

When a user drags a workflow in and a file is missing, the ComfyUI frontend then lists it with a one-click
download button (same mechanism as the official templates). Source of truth: models.json.

Usage: uv run python scripts/embed_model_urls.py
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    manifest = {m["name"]: m for m in json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))["models"]}
    for path in sorted(glob.glob(os.path.join(ROOT, "workflows", "*_web.json"))):
        wf = json.load(open(path, encoding="utf-8"))
        tagged, unknown = 0, set()
        for n in wf["nodes"]:
            vals = n.get("widgets_values") if isinstance(n.get("widgets_values"), list) else []
            refs = [v for v in vals if isinstance(v, str) and v.endswith((".safetensors", ".gguf"))]
            if not refs:
                continue
            entries = [{"name": v, "url": manifest[v]["url"], "directory": manifest[v]["directory"]} for v in refs if v in manifest]
            unknown.update(v for v in refs if v not in manifest)
            if entries:
                n.setdefault("properties", {})["models"] = entries
                tagged += 1
        json.dump(wf, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"{os.path.basename(path)}: tagged {tagged} loader nodes" + (f"; NOT IN models.json: {sorted(unknown)}" if unknown else ""))


if __name__ == "__main__":
    main()
