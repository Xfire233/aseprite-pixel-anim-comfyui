#!/usr/bin/env python3
"""Deterministic guardrail for ComfyUI Web workflow JSONs (workflows/*_web.json).

Checks:
1. Link table integrity: every link ID referenced on node inputs/outputs exists in `links` with matching node/slot IDs.
2. No untiled `VAEDecode` nodes (must use `VAEDecodeTiled` on 16GB VRAM to avoid WDDM shared-memory paging).
3. No hardcoded character traits ('pink bob hair', 'round glasses', 'white hair bow', 'gold choker') leaked inside
   AsepriteLLMPromptDirector `system_prompt` widgets.
4. Every `LoadImage` node has a clear, non-empty `title`.
5. No LLM endpoint or key stored in AsepriteLLMPromptDirector / AsepriteCharcardExtractor (keep workflows shareable).
"""
import glob
import json
import os
import sys

BANNED_SYS_PROMPT_SUBSTRINGS = [
    "pink bob hair",
    "round glasses",
    "white hair bow",
    "gold choker",
]


def check_workflow(path: str) -> list[str]:
    errs = []
    with open(path, "r", encoding="utf-8") as f:
        wf = json.load(f)

    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    links = {l[0]: l for l in wf.get("links", [])}

    for nid, n in nodes.items():
        ntype = n.get("type", "")
        if ntype == "VAEDecode":
            errs.append(f"Node #{nid} uses untiled VAEDecode (must use VAEDecodeTiled on 16GB VRAM)")
        if ntype == "LoadImage" and not (n.get("title") or "").strip():
            errs.append(f"Node #{nid} (LoadImage) is missing an explicit title")
        if ntype in ("AsepriteLLMPromptDirector", "AsepriteCharcardExtractor"):
            named = n.get("widgets_values_named") or {}
            vals = [v for v in (n.get("widgets_values") or []) if isinstance(v, str)] + [v for v in named.values() if isinstance(v, str)]
            if any(v.startswith(("http://", "https://")) and "127.0.0.1" not in v and "localhost" not in v for v in vals):
                errs.append(f"Node #{nid} ({ntype}) stores an LLM base_url; leave it empty in committed workflows")
            wv = n.get("widgets_values") or []
            pos_key = wv[1] if ntype == "AsepriteLLMPromptDirector" and len(wv) > 1 and isinstance(wv[1], str) else ""
            if (named.get("api_key") or "").strip() or (pos_key.strip() and not set(pos_key.strip()) <= set("•* ")):
                errs.append(f"Node #{nid} ({ntype}) stores an api_key")
        if ntype == "AsepriteLLMPromptDirector":
            wvals = n.get("widgets_values") or []
            if len(wvals) >= 6 and isinstance(wvals[5], str):
                sys_p = wvals[5].lower()
                for banned in BANNED_SYS_PROMPT_SUBSTRINGS:
                    if banned in sys_p:
                        errs.append(f"Node #{nid} system_prompt contains hardcoded character trait '{banned}'")

        for slot_idx, inp in enumerate(n.get("inputs") or []):
            lid = inp.get("link")
            if lid is not None:
                if lid not in links:
                    errs.append(f"Node #{nid} input '{inp.get('name')}' references missing link #{lid}")
                else:
                    l = links[lid]
                    if l[3] != nid or l[4] != slot_idx:
                        errs.append(
                            f"Link #{lid} target mismatch: link says ({l[3]}, slot {l[4]}), "
                            f"node #{nid} input '{inp.get('name')}' is at slot {slot_idx}"
                        )

        for slot_idx, out in enumerate(n.get("outputs") or []):
            for lid in out.get("links") or []:
                if lid not in links:
                    errs.append(f"Node #{nid} output '{out.get('name')}' references missing link #{lid}")
                else:
                    l = links[lid]
                    if l[1] != nid or l[2] != slot_idx:
                        errs.append(
                            f"Link #{lid} origin mismatch: link says ({l[1]}, slot {l[2]}), "
                            f"node #{nid} output '{out.get('name')}' is at slot {slot_idx}"
                        )

    return errs


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, "workflows", "*_web.json")))
    if not paths:
        print("No workflows/*_web.json files found.")
        sys.exit(1)
    failed = False
    for p in paths:
        errs = check_workflow(p)
        if errs:
            failed = True
            print(f"[FAIL] {p}:")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"[PASS] {p}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
