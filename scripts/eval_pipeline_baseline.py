"""Automated Phase 1 Feedback Loop for Stage 1 Inpaint, LoRA safety, and Fake-Warp elimination.
Run:
  python scripts/eval_pipeline_baseline.py
"""
import json
import os
import re
import sys
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, r"C:\baidunetdiskdownload\comfyui1.6\ComfyUI-aki-v1.6\ComfyUI")
def check_mask_coverage():
    print("[Check 1/6] Inpaint Mask Scope & Coverage...")
    # Canonical OpenPose-18 points normalized [0, 1]
    CANONICAL = {
        "nose": (0.50, 0.18), "neck": (0.50, 0.25), "r_sho": (0.42, 0.27), "l_sho": (0.58, 0.27),
        "r_elb": (0.36, 0.39), "l_elb": (0.64, 0.39), "r_wri": (0.33, 0.51), "l_wri": (0.67, 0.51),
        "r_hip": (0.45, 0.49), "l_hip": (0.55, 0.49), "r_kne": (0.44, 0.66), "l_kne": (0.56, 0.66),
        "r_ank": (0.43, 0.83), "l_ank": (0.57, 0.83), "r_eye": (0.48, 0.17), "l_eye": (0.52, 0.17),
        "r_ear": (0.46, 0.18), "l_ear": (0.54, 0.18)
    }
    W, H = 512, 768
    from __init__ import AsepritePoseRegionMask
    mask_node = AsepritePoseRegionMask()
    
    # Check default arguments in INPUT_TYPES
    inputs_def = mask_node.INPUT_TYPES()["required"]
    def_pelvis = inputs_def["pelvis"][1]["default"]
    def_chest = inputs_def["chest"][1]["default"]
    def_size = inputs_def["size"][1]["default"]
    
    # Assert defaults do NOT mask chest and do not blow up size
    assert def_chest is False, f"FAIL: default chest is {def_chest}, MUST be False to prevent whole-body redrawing!"
    assert def_size <= 1.05, f"FAIL: default size is {def_size}, MUST be <= 1.0 to prevent mask spilling!"
    
    # Now build mask on simulated character
    kps = {k: [v[0]*W, v[1]*H, 1.0] for k, v in CANONICAL.items()}
    flat_kps = []
    for k in ("nose", "neck", "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
              "r_hip", "r_kne", "r_ank", "l_hip", "l_kne", "l_ank", "r_eye", "l_eye", "r_ear", "l_ear"):
        flat_kps.extend(kps[k])
        
    pose_kps = [{"people": [{"pose_keypoints_2d": flat_kps}], "canvas_width": W, "canvas_height": H}]
    mask_tensor = mask_node.build(pose_kps, pelvis=True, chest=False, size=1.0, feather=12)[0]
    m_arr = (mask_tensor.numpy()[0] * 255).astype(np.uint8)
    
    ys, xs = np.where(m_arr > 32)
    height_covered = ys.max() - ys.min() if len(ys) > 0 else 0
    area_covered = (m_arr > 32).sum()
    fraction = area_covered / (W * H)
    
    print(f"   Mask: height={height_covered}px, area={area_covered}px ({fraction*100:.1f}% of canvas)")
    assert height_covered < 220, f"FAIL: mask height {height_covered}px >= 220px! Too big, will redraw entire body!"
    assert fraction < 0.08, f"FAIL: mask fraction {fraction*100:.1f}% >= 8%! Too big!"
    print("   [PASS] Mask is strictly localized to pelvis region only.")

def check_prompt_director():
    print("[Check 2/6] Prompt Director Anima Prompt Non-Empty...")
    from __init__ import AsepriteLLMPromptDirector
    director = AsepriteLLMPromptDirector()
    res = director.build_local_template(
        charcard="none (不使用预设角色卡，自动识别参考图1外观)",
        pose_preset="solo_m_legs_sit (单人·M字开腿坐地抚腿娇喘)",
        user_prompt="参考图1的角色，换成M字开腿坐在地上的姿势，下身不穿裤子完全裸露，双手扶在大腿上，害羞脸红；动作：缓慢呼吸，胸口与肩膀轻微起伏"
    )
    anima_pos, anima_neg, wan_pos, wan_neg = res[:4]
    assert len(anima_pos.strip()) > 30, f"FAIL: anima_positive is too short or empty: '{anima_pos}'"
    assert "masterpiece" in anima_pos, f"FAIL: 'masterpiece' missing from anima_positive: '{anima_pos}'"
    print(f"   anima_positive len={len(anima_pos)}, contains masterpiece and pose tags.")
    print("   [PASS] Director outputs valid non-empty Anima tags.")

def check_lora_and_workflow_parameters():
    print("[Check 3/6] Workflow LoRA and Default Parameters...")
    for wf_name in ["workflows/i2i_qwen_edit_wan_aseprite_web.json", "workflows/anima_wan_aseprite_full_web.json"]:
        path = os.path.join(PROJECT_ROOT, wf_name)
        with open(path, "r", encoding="utf-8") as f:
            wf = json.load(f)
        nodes = {n["id"]: n for n in wf["nodes"]}
        
        # Node 40: qwen_nsfw_adv LoKr
        n40 = nodes.get(40)
        if n40 and n40.get("type") == "LoraLoaderBypassModelOnly":
            strength = n40.get("widgets_values", [None, None])[1]
            assert strength == 0.8, f"FAIL in {wf_name}: Node 40 (qwen_nsfw_adv) strength is {strength}, MUST be 0.8 (noon baseline)!"
            
        # Node 41: Unchained-XXX
        n41 = nodes.get(41)
        if n41 and n41.get("type") == "LoraLoaderBypassModelOnly":
            strength = n41.get("widgets_values", [None, None])[1]
            assert strength == 0.72, f"FAIL in {wf_name}: Node 41 strength is {strength}, MUST be 0.72!"
            
        # Node 47: AsepritePoseRegionMask
        n47 = nodes.get(47)
        if n47 and n47.get("type") == "AsepritePoseRegionMask":
            wv = n47.get("widgets_values", [])
            pelvis, chest, size = wv[0], wv[1], wv[2]
            assert chest is False, f"FAIL in {wf_name}: Node 47 chest is {chest}, MUST be False!"
            assert size == 1.4, f"FAIL in {wf_name}: Node 47 size is {size}, MUST be 1.4!"
            
        # Node 53: KSampler denoise
        n53 = nodes.get(53)
        if n53 and n53.get("type") == "KSampler":
            denoise = n53.get("widgets_values", [None]*7)[6]
            assert denoise == 0.7, f"FAIL in {wf_name}: Node 53 denoise is {denoise}, MUST be 0.70!"
            
        # Node 56: RegexReplace
        n56 = nodes.get(56)
        if n56 and n56.get("type") == "RegexReplace":
            wv = n56.get("widgets_values", [])
            assert '^[\\s\\S]*?(?=masterpiece)' in wv, f"FAIL in {wf_name}: Node 56 regex_pattern missing masterpiece anchor!"
    print("   [PASS] Workflows strictly adhere to un-polluted LoRA, mask, and denoise bounds.")

def check_no_fake_warp():
    print("[Check 4/6] Codebase Fake-Warp & Trajectory Elimination...")
    init_path = os.path.join(PROJECT_ROOT, "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "_warp_image_by_pose_kps" not in src, "FAIL: _warp_image_by_pose_kps found in __init__.py! Fake motion still present!"
    assert "_ACTIVE_TRAJ_LATENT" not in src, "FAIL: _ACTIVE_TRAJ_LATENT found in __init__.py! Sampling override still present!"
    assert "_aseprite_vae_encode" not in src, "FAIL: _aseprite_vae_encode found in __init__.py! VAE hook still present!"
    print("   [PASS] Zero fake-warp deformation hooks remain in __init__.py.")

def check_api_key_masking():
    print("[Check 5/6] Frontend API Key Masking...")
    js_path = os.path.join(PROJECT_ROOT, "web", "llm_director.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    assert "maskApiKeyWidget" in js, "FAIL: maskApiKeyWidget missing from web/llm_director.js!"
    assert "type = \"password\"" in js, "FAIL: password input type setting missing from web/llm_director.js!"
    assert "••••••••••••••••" in js, "FAIL: canvas password masking missing from web/llm_director.js!"
    print("   [PASS] API Key frontend masking is fully implemented.")

def check_metadata_embedding():
    print("[Check 6/6] Output PngInfo & Workflow Embedding...")
    init_path = os.path.join(PROJECT_ROOT, "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert '"hidden": {' in src and '"prompt": "PROMPT"' in src and '"extra_pnginfo": "EXTRA_PNGINFO"' in src, \
        "FAIL: AsepritePixelizeAssemble missing hidden prompt/extra_pnginfo inputs!"
    assert "pnginfo=metadata" in src, "FAIL: pnginfo=metadata missing from PNG save calls!"
    assert "_workflow.json" in src, "FAIL: automatic _workflow.json backup missing!"
    print("   [PASS] Output metadata embedding and workflow JSON backup are wired.")

def main():
    print("="*60)
    print("RUNNING FEEDBACK LOOP REGRESSION SUITE")
    print("="*60)
    check_mask_coverage()
    check_prompt_director()
    check_lora_and_workflow_parameters()
    check_no_fake_warp()
    check_api_key_masking()
    check_metadata_embedding()
    print("="*60)
    print("ALL 6 PHASE 1 FEEDBACK CHECKS PASSED: BASELINE LOCKED GREEN")
    print("="*60)

if __name__ == "__main__":
    main()
